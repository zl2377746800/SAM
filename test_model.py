import torch
import matplotlib.pyplot as plt
from pathlib import Path
from landslide_detection import EnhancedLandslideDetectionModel
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path: str) -> torch.Tensor:
    """加载图像并进行预处理"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # 添加批次维度

def load_dem(dem_path: str) -> torch.Tensor:
    """加载DEM文件并进行预处理"""
    dem = Image.open(dem_path).convert('L')  # 转换为灰度图
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(dem).unsqueeze(0)  # 添加批次维度

def visualize_results(image: torch.Tensor, dem: torch.Tensor, prediction: torch.Tensor) -> None:
    """可视化结果"""
    image = image.squeeze(0).permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
    dem = dem.squeeze(0).numpy()  # [1, H, W] -> [H, W]
    prediction = prediction.squeeze(0).detach().numpy()  # [1, H, W] -> [H, W]

    # 二值化预测结果
    binary_prediction = (prediction > 0.5).astype(float)  # 将预测结果二值化

    # 创建一个彩色图像用于显示
    colored_image = image.copy()
    colored_image[binary_prediction == 1] = [255, 0, 0]  # 将滑坡区域标记为红色

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(dem, cmap='terrain')
    axes[1].set_title('DEM')
    axes[1].axis('off')

    axes[2].imshow(colored_image)
    axes[2].set_title('Prediction with Landslide Areas')
    axes[2].axis('off')

    plt.show()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model_type = "vit_b"
    sam_checkpoint = str(Path(f'models/sam_{model_type}.pth').absolute())
    model = EnhancedLandslideDetectionModel(sam_checkpoint=sam_checkpoint, model_type=model_type, device=device)
    model.load_state_dict(torch.load('models/checkpoints/best_model.pth')['model_state_dict'])
    model.to(device)
    model.eval()

    # 加载图像和DEM
    image_path = 'D:/python project/graduation project/demo/test/hz026.png'  # 替换为您的图像路径
    dem_path = 'D:/python project/graduation project/demo/test/hz026.png'  # 替换为您的DEM文件路径

    image = load_image(image_path).to(device)
    dem = load_dem(dem_path).to(device)

    # 进行推理
    with torch.no_grad():
        predictions = model(image, dem.squeeze(1))  # 去掉多余的维度
        main_output = predictions['main']

    # 可视化结果
    visualize_results(image.cpu(), dem.cpu().squeeze(0), main_output.cpu().squeeze(0))  # 去掉多余的维度

if __name__ == '__main__':
    main() 