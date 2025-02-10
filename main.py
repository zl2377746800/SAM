import torch
from pathlib import Path
from landslide_detection import EnhancedLandslideDetectionModel, create_dataloaders, train_model
from download_sam import download_sam_model
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os
from typing import Tuple
from torch.utils.data import DataLoader, ConcatDataset
from dataset.landslide_dataset import LandslideDataset, custom_collate_fn

def setup_logging():
    """设置日志"""
    # 创建logs文件夹
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 设置日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def visualize_batch(batch, predictions=None, save_dir='visualizations'):
    """可视化一个批次的数据"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取批次中的第一个样本
    image = batch['image'][0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    dem = batch['dem'][0].cpu().numpy()
    mask = batch['mask'][0].cpu().numpy()[0]  # 移除通道维度
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 显示原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image')
    
    # 显示DEM
    dem_plot = axes[0, 1].imshow(dem, cmap='terrain')
    plt.colorbar(dem_plot, ax=axes[0, 1])
    axes[0, 1].set_title('DEM')
    
    # 显示真实标签
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Ground Truth')
    
    # 显示预测结果（如果有）
    if predictions is not None:
        pred = predictions['main'][0].cpu().detach().numpy()[0]
        axes[1, 1].imshow(pred, cmap='gray')
        axes[1, 1].set_title('Prediction')
    
    # 保存图像
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'{save_dir}/visualization_{timestamp}.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置日志
    setup_logging()
    logging.info(f'Using device: {device}')
    
    # 设置路径
    data_root = Path('dataset').absolute()
    model_type = "vit_b"
    sam_checkpoint = str(Path(f'models/sam_{model_type}.pth').absolute())
    
    # 确保文件存在
    if not data_root.exists():
        raise ValueError(f"找不到数据集目录: {data_root}")
    if not Path(sam_checkpoint).exists():
        raise ValueError(f"找不到SAM模型文件: {sam_checkpoint}")
    
    logging.info(f"数据集路径: {data_root}")
    logging.info(f"SAM模型路径: {sam_checkpoint}")
    
    # 优化的训练配置
    config = {
        'batch_size': 8,          # V100可以处理更大的批次
        'num_workers': 8,         # 与CPU核心数匹配
        'num_epochs': 50,
        'accumulation_steps': 2,  # 减少梯度累积步数
        'sam_type': model_type,
        'device': 'cuda'
    }
    
    # 创建模型
    logging.info('Creating model...')
    model = EnhancedLandslideDetectionModel(
        sam_checkpoint=sam_checkpoint,
        model_type=config['sam_type'],
        device=config['device']
    ).to(config['device'])
    
    # 创建数据加载器
    logging.info('Creating data loaders...')
    try:
        train_loader, val_loader = create_dataloaders(
            data_root=str(data_root),
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        logging.info(f"训练集大小: {len(train_loader.dataset)}")
        logging.info(f"验证集大小: {len(val_loader.dataset)}")
    except Exception as e:
        logging.error(f"创建数据加载器失败: {str(e)}")
        raise
    
    # 开始训练
    logging.info('Starting training...')
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            accumulation_steps=config['accumulation_steps']
        )
        logging.info('Training completed successfully!')
        
        # 检查模型文件是否存在
        best_model_path = Path('models/checkpoints/best_model.pth')
        last_model_path = Path('models/checkpoints/last_model.pth')
        
        if best_model_path.exists():
            logging.info(f'最佳模型已保存: {best_model_path}')
        else:
            logging.warning(f'未找到最佳模型文件: {best_model_path}')
            
        if last_model_path.exists():
            logging.info(f'最终模型已保存: {last_model_path}')
        else:
            logging.warning(f'未找到最终模型文件: {last_model_path}')
            
    except Exception as e:
        logging.error(f'Training failed with error: {str(e)}')
        raise
    
    # 可视化一个批次的结果
    logging.info('Visualizing results...')
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        images = batch['image'].to(device)
        dems = batch['dem'].to(device)
        predictions = model(images, dems)
        visualize_batch(batch, predictions)

if __name__ == '__main__':
    main() 