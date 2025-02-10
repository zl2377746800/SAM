import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from typing import Dict, List, Tuple
from models.sam_adapter import SAMAdapter
from dataset.landslide_dataset import LandslideDataset, custom_collate_fn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import os
from pathlib import Path

class DualAttentionFusion(nn.Module):
    """双重注意力特征融合模块"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        
        return x_ca * sa

class DEMEncoder(nn.Module):
    """增强的DEM特征提取网络"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            self._make_layer(1, 32),      # Level 0: [B, 32, 512, 512]
            self._make_layer(32, 64),     # Level 1: [B, 64, 256, 256]
            self._make_layer(64, 128),    # Level 2: [B, 128, 128, 128]
            self._make_layer(128, 256)    # Level 3: [B, 256, 64, 64]
        ])
        
        self.attention_blocks = nn.ModuleList([
            DualAttentionFusion(32),
            DualAttentionFusion(64),
            DualAttentionFusion(128),
            DualAttentionFusion(256)
        ])
        
    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for encoder, attention in zip(self.encoder, self.attention_blocks):
            x = encoder(x)
            x = attention(x)
            features.append(x)
            if len(features) < len(self.encoder):
                x = F.max_pool2d(x, 2)
        return features

class EnhancedLandslideDetectionModel(nn.Module):
    def __init__(self, sam_checkpoint: str, model_type: str = "vit_b", device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        # SAM适配器和DEM编码器
        self.sam_adapter = SAMAdapter(
            checkpoint_path=sam_checkpoint,
            model_type=model_type
        )
        self.dem_encoder = DEMEncoder()
        
        # 添加特征降维层，将拼接后的特征降到512通道
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(768, 512, 1),  # 768 = 512(SAM) + 256(DEM)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合层
        self.fusion_layers = nn.ModuleList([
            self._make_fusion_layer(512, 256),  # 512 -> 256
            self._make_fusion_layer(256, 128),  # 256 -> 128
            self._make_fusion_layer(128, 64)    # 128 -> 64
        ])
        
        # 特征转换层 - 用于跳跃连接
        self.skip_transforms = nn.ModuleList([
            nn.Conv2d(128, 64, 1),  # 转换Level 2特征
            nn.Conv2d(64, 64, 1),   # 转换Level 1特征
            nn.Conv2d(32, 64, 1)    # 转换Level 0特征，输出改为64通道
        ])
        
        # 解码器
        self.decoder = nn.ModuleList([
            self._make_decoder_block(64, 64),   # 64->128
            self._make_decoder_block(64, 64),   # 128->256
            self._make_decoder_block(64, 64)    # 256->512，改为64通道输出
        ])
        
        # 修改最终输出层的输入通道数
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 输入通道改为64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # 辅助头保持不变
        self.aux_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
    
    def _make_fusion_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """创建特征融合层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            DualAttentionFusion(out_channels)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """创建改进的解码器块"""
        return nn.Sequential(
            # 先进行特征处理
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # 然后进行上采样
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, image: torch.Tensor, dem: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        with torch.amp.autocast('cuda'):
            image_features = self.sam_adapter.get_image_embedding(image)
            points, labels = self.sam_adapter.generate_prompt_points(image_features)
            sam_mask = self.sam_adapter.get_mask_prediction(image, points, labels)
            
            dem_features = self.dem_encoder(dem.unsqueeze(1))
            
            # 打印特征形状以进行调试
            print("Image features shape:", image_features.shape)
            print("DEM features shape:", dem_features[-1].shape)
            
            # 特征融合
            x = torch.cat([image_features, dem_features[-1]], dim=1)
            del image_features  # 及时释放内存
            
            # 降维
            x = self.dim_reduction(x)
            
            # 特征融合过程
            for i, fusion in enumerate(self.fusion_layers):
                x = fusion(x)
                if i == 0:
                    aux_features = x
            
            # 解码过程
            for i, (decoder, skip_transform) in enumerate(zip(self.decoder, self.skip_transforms)):
                x = decoder(x)
                if i < len(dem_features) - 1:
                    skip_feat = skip_transform(dem_features[-(i+2)])
                    x = x + skip_feat
            
            del dem_features  # 释放内存
            
            # 调整SAM mask大小并添加
            sam_mask_resized = F.interpolate(
                sam_mask.float(), 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            x = x + sam_mask_resized
            
            # 生成输出
            main_output = self.final_conv(x)
            aux_output = self.aux_head(aux_features)
            
            torch.cuda.empty_cache()  # 清理GPU缓存
            
            return {
                'main': main_output,
                'aux': aux_output,
                'sam_mask': sam_mask
            }

def create_dataloaders(data_root: str,
                      batch_size: int = 8,
                      num_workers: int = 8,
                      train_val_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    # 加载滑坡和非滑坡数据集
    landslide_dataset = LandslideDataset(data_root, split='landslide')
    non_landslide_dataset = LandslideDataset(data_root, split='non-landslide')
    
    # 合并数据集
    full_dataset = ConcatDataset([landslide_dataset, non_landslide_dataset])
    
    # 划分训练集和验证集
    train_size = int(len(full_dataset) * train_val_split)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器，使用自定义的collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,  # 预取2个批次
        persistent_workers=True,  # 保持worker进程存活
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,  # 预取2个批次
        persistent_workers=True,  # 保持worker进程存活
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

def visualize_results(image: torch.Tensor,
                     dem: torch.Tensor,
                     prediction: torch.Tensor,
                     true_label: torch.Tensor,
                     save_path: str = None) -> None:
    """可视化结果
    Args:
        image: [3, H, W] RGB图像
        dem: [H, W] DEM数据
        prediction: [1, H, W] 预测结果
        true_label: [1, H, W] 真实标签
        save_path: 保存路径
    """
    # 转换为numpy数组并调整维度
    image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    dem = dem.cpu().numpy()  # [H, W]
    prediction = prediction.cpu().numpy().squeeze()  # [H, W]
    true_label = true_label.cpu().numpy().squeeze()  # [H, W]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('滑坡检测结果', fontsize=16)
    
    # 显示原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 显示DEM
    dem_plot = axes[0, 1].imshow(dem, cmap='terrain')
    plt.colorbar(dem_plot, ax=axes[0, 1])
    axes[0, 1].set_title('DEM数据')
    axes[0, 1].axis('off')
    
    # 显示预测结果
    axes[1, 0].imshow(prediction, cmap='jet', vmin=0, vmax=1)
    axes[1, 0].set_title('预测结果')
    axes[1, 0].axis('off')
    
    # 显示真实标签
    axes[1, 1].imshow(true_label, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title('真实标签')
    axes[1, 1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存结果
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'visualizations/result_{timestamp}.png'
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """计算准确率和IoU
    Args:
        predictions: 预测结果 [B, 1, H, W]
        targets: 真实标签 [B, 1, H, W]
        threshold: 二值化阈值
    Returns:
        accuracy: 准确率
        iou: IoU分数
    """
    # 转换为二值图
    pred_binary = (predictions > threshold).float()
    
    # 计算准确率
    correct = (pred_binary == targets).float().mean()
    accuracy = correct.item() * 100  # 转换为百分比
    
    # 计算IoU
    intersection = (pred_binary * targets).sum()
    union = pred_binary.sum() + targets.sum() - intersection
    iou = (intersection / (union + 1e-6)).item() * 100
    
    return accuracy, iou

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                num_epochs: int,
                device: str = 'cuda',
                accumulation_steps: int = 2,
                target_accuracy: float = 90.0) -> None: # 目标准确率。达到百分之九十停止训练。修改此处，达到自己想要的准确率停止训练
    
    # 优化学习率和优化器配置
    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=2e-4,              # 略微提高学习率
                                 weight_decay=0.01)    # 添加权重衰减
    
    # 使用OneCycleLR调度器以获得更好的收敛性
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.3,            # 30%时间用于预热
        div_factor=25,            # 初始学习率 = max_lr/25
        final_div_factor=1000,    # 最终学习率 = max_lr/1000
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n开始训练 - {num_epochs}轮 | {len(train_loader.dataset)}训练样本 | {len(val_loader.dataset)}验证样本\n")
    
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    
    # 创建模型保存目录
    try:
        model_dir = Path('models/checkpoints')
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"模型保存目录: {model_dir.absolute()}")
        
        best_model_path = model_dir / 'best_model.pth'
        last_model_path = model_dir / 'last_model.pth'
        
        # 测试目录是否可写
        test_file = model_dir / 'test.txt'
        test_file.write_text('test')
        test_file.unlink()  # 删除测试文件
        
    except Exception as e:
        print(f"创建模型目录失败: {str(e)}")
        raise
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = 0
        optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in enumerate(train_loader):
            if i % 50 == 0:
                print(f"Batch: {i}/{len(train_loader)}", end='\r')
                
            images = batch['image'].to(device)
            dems = batch['dem'].to(device)
            labels = batch['mask'].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images, dems)
                main_loss = criterion(outputs['main'], labels)
                aux_labels = F.interpolate(labels, size=outputs['aux'].shape[2:], mode='nearest')
                aux_loss = criterion(outputs['aux'], aux_labels)
                loss = (main_loss + 0.4 * aux_loss) / accumulation_steps
                
                # 计算训练指标
                acc, iou = calculate_metrics(outputs['main'], labels)
                train_acc += acc
                train_iou += iou
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            
            del outputs, main_loss, aux_loss, loss
            
        avg_train_loss = train_loss/len(train_loader)
        avg_train_acc = train_acc/len(train_loader)
        avg_train_iou = train_iou/len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_acc = 0
        val_iou = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                dems = batch['dem'].to(device)
                labels = batch['mask'].to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images, dems)
                    loss = criterion(outputs['main'], labels)
                    
                    # 计算验证指标
                    acc, iou = calculate_metrics(outputs['main'], labels)
                    val_acc += acc
                    val_iou += iou
                    
                val_loss += loss.item()
        
        avg_val_loss = val_loss/len(val_loader)
        avg_val_acc = val_acc/len(val_loader)
        avg_val_iou = val_iou/len(val_loader)
        
        scheduler.step()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_acc': avg_val_acc,
                'val_iou': avg_val_iou
            }, best_model_path)
            print(f"【保存最佳模型】Loss: {best_val_loss:.4f}, Acc: {avg_val_acc:.2f}%, IoU: {avg_val_iou:.2f}%")
            
            # 可视化最佳模型的预测结果
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                images = val_batch['image'].to(device)
                dems = val_batch['dem'].to(device)
                labels = val_batch['mask'].to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(images, dems)
                
                vis_path = f'visualizations/best_epoch_{epoch+1}.png'
                visualize_results(
                    images[0],
                    dems[0],
                    outputs['main'][0],
                    labels[0],
                    save_path=vis_path
                )
            
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n{patience}轮未改善，停止训练")
                break
        
        # 保存最后一个模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'val_acc': avg_val_acc,
            'val_iou': avg_val_iou
        }, last_model_path)
        
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%, IoU: {avg_train_iou:.2f}%")
        print(f"Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}%, IoU: {avg_val_iou:.2f}%")
        
        # 检查是否达到目标准确率
        if avg_val_acc >= target_accuracy:
            print(f"\n达到目标准确率 {target_accuracy}%，停止训练")
            break
        
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache() 