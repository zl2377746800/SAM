import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import rasterio
from pathlib import Path
import logging
import torchvision.transforms.functional as TF
import random

class LandslideDataset(Dataset):
    """滑坡数据集加载器"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'landslide',  # 'landslide' or 'non-landslide'
                 transform: bool = True):
        """
        参数:
            root_dir (str): 数据集根目录
            split (str): 数据集划分 ('landslide' or 'non-landslide')
            transform (bool): 是否使用数据增强
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # 获取所有样本路径
        self.samples = self._get_samples()
        
        # 验证文件是否存在
        self._validate_files()
        
    def _validate_files(self):
        """验证所有文件是否存在"""
        valid_samples = []
        for sample in self.samples:
            is_valid = True
            
            # 检查图像文件
            if not os.path.exists(sample['image']):
                logging.warning(f"找不到图像文件: {sample['image']}")
                is_valid = False
                
            # 检查DEM文件
            if not os.path.exists(sample['dem']):
                logging.warning(f"找不到DEM文件: {sample['dem']}")
                is_valid = False
                
            # 如果是滑坡数据，检查mask文件
            if self.split == 'landslide' and sample['mask'] is not None:
                if not os.path.exists(sample['mask']):
                    logging.warning(f"找不到mask文件: {sample['mask']}")
                    is_valid = False
            
            if is_valid:
                valid_samples.append(sample)
        
        self.samples = valid_samples
        logging.info(f"{self.split} 数据集有效样本数: {len(self.samples)}")
    
    def _get_samples(self) -> List[Dict[str, str]]:
        """获取所有样本的路径"""
        split_dir = self.root_dir / self.split
        
        # 确保目录存在
        if not split_dir.exists():
            raise ValueError(f"找不到数据集目录: {split_dir}")
            
        image_dir = split_dir / 'image'
        if not image_dir.exists():
            raise ValueError(f"找不到图像目录: {image_dir}")
            
        # 获取所有图像文件，现在包括.png文件
        image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.lower().endswith('.png')  # 只查找png文件
        ])
        
        if not image_files:
            raise ValueError(f"在 {image_dir} 中没有找到PNG图像文件")
        
        samples = []
        for img_file in image_files:
            base_name = img_file.rsplit('.', 1)[0]
            
            # 根据数据集类型构建不同的样本字典
            if self.split == 'landslide':
                sample = {
                    'image': str(image_dir / img_file),
                    'dem': str(split_dir / 'dem' / f'{base_name}.png'),  # DEM文件也是png
                    'mask': str(split_dir / 'mask' / f'{base_name}.png')
                    if os.path.exists(split_dir / 'mask' / f'{base_name}.png') else None,
                    'polygon': str(split_dir / 'polygon_coordinate' / f'{base_name}.txt')
                    if os.path.exists(split_dir / 'polygon_coordinate' / f'{base_name}.txt') else None
                }
            else:  # non-landslide
                sample = {
                    'image': str(image_dir / img_file),
                    'dem': str(split_dir / 'dem' / f'{base_name}.png'),  # DEM文件也是png
                    'mask': None,  # 非滑坡数据没有mask
                    'polygon': None  # 非滑坡数据没有polygon
                }
            
            samples.append(sample)
            
        return samples
    
    def _load_image(self, path: str) -> torch.Tensor:
        """加载并预处理图像"""
        image = Image.open(path).convert('RGB')
        # 确保图像大小为512x512
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
        image = np.array(image)
        # 标准化到[0,1]
        image = image.astype(np.float32) / 255.0
        # 转换为CxHxW格式
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    
    def _load_dem(self, path: str) -> torch.Tensor:
        """加载并预处理DEM数据，现在处理PNG格式"""
        try:
            # 使用PIL加载PNG格式的DEM数据
            dem = Image.open(path).convert('L')  # 转换为灰度图
            # 确保DEM大小为512x512
            if dem.size != (512, 512):
                dem = dem.resize((512, 512), Image.Resampling.BILINEAR)
            dem = np.array(dem)
            
            # 标准化DEM数据
            dem = dem.astype(np.float32)
            if dem.max() > dem.min():  # 避免除零
                dem = (dem - dem.min()) / (dem.max() - dem.min())
            
            dem = torch.from_numpy(dem)
            return dem
            
        except Exception as e:
            logging.error(f"加载DEM文件失败 {path}: {str(e)}")
            # 返回零张量作为替代
            return torch.zeros((512, 512), dtype=torch.float32)
    
    def _load_mask(self, path: Optional[str]) -> torch.Tensor:
        """加载并预处理mask"""
        if path is None:
            return torch.zeros((1, 512, 512), dtype=torch.float32)
            
        mask = Image.open(path).convert('L')
        # 确保mask大小为512x512
        if mask.size != (512, 512):
            mask = mask.resize((512, 512), Image.Resampling.NEAREST)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return mask
    
    def _load_polygon(self, path: Optional[str]) -> Optional[torch.Tensor]:
        """加载多边形坐标"""
        if path is None:
            return torch.zeros((1, 2), dtype=torch.float32)
        
        try:
            # 读取文件的所有行
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 跳过头部文本，直接处理坐标数据
            coordinates = []
            for line in lines:
                line = line.strip()
                # 跳过头部文本行
                if 'landslide instance name:' in line or 'coordinates' in line:
                    continue
                
                try:
                    # 分割并转换为浮点数
                    x, y = map(float, line.split())
                    coordinates.append([x, y])
                except ValueError:
                    continue
            
            if not coordinates:  # 如果没有有效坐标
                logging.warning(f"在文件 {path} 中没有找到有效的坐标")
                return torch.zeros((1, 2), dtype=torch.float32)
            
            # 转换为张量
            coordinates = torch.tensor(coordinates, dtype=torch.float32)
            
            # 标准化坐标到[0,1]范围
            if coordinates.shape[0] > 0:
                coordinates_min = coordinates.min(dim=0)[0]
                coordinates_max = coordinates.max(dim=0)[0]
                coordinates = (coordinates - coordinates_min) / (coordinates_max - coordinates_min + 1e-8)
                
                # 缩放到图像大小
                coordinates = coordinates * 512  # 假设图像大小为512x512
            
            return coordinates
            
        except Exception as e:
            logging.error(f"加载多边形文件失败 {path}: {str(e)}")
            return torch.zeros((1, 2), dtype=torch.float32)
    
    def apply_augmentation(self, 
                          image: torch.Tensor,
                          dem: torch.Tensor,
                          mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """应用数据增强"""
        # 将所有张量转换为相同的设备
        device = image.device
        
        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            dem = TF.hflip(dem)
            mask = TF.hflip(mask)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            dem = TF.vflip(dem)
            mask = TF.vflip(mask)
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            # 对于图像（3通道）
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            # 对于DEM（单通道）
            dem = TF.rotate(dem.unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)
            # 对于mask（单通道）
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 随机亮度、对比度调整（仅对图像）
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        
        return image, dem, mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载数据
        image = self._load_image(sample['image'])
        dem = self._load_dem(sample['dem'])
        mask = self._load_mask(sample['mask'])
        polygon = self._load_polygon(sample.get('polygon'))
        
        # 数据增强
        if self.transform:
            image, dem, mask = self.apply_augmentation(image, dem, mask)
        
        # 确保所有返回值都是张量
        return {
            'image': image,  # [3, 512, 512]
            'dem': dem,      # [512, 512]
            'mask': mask,    # [1, 512, 512]
            'polygon': polygon,  # [N, 2] 或 [1, 2]
            'metadata': {
                'image_path': sample['image'],
                'dem_path': sample['dem'],
                'mask_path': sample['mask'],
                'polygon_path': sample.get('polygon')
            }
        }

def custom_collate_fn(batch):
    """自定义的批处理函数，处理不同长度的多边形坐标"""
    # 分离不同类型的数据
    images = torch.stack([item['image'] for item in batch])
    dems = torch.stack([item['dem'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    # 多边形坐标作为列表存储，不进行堆叠
    polygons = [item['polygon'] for item in batch]
    
    # 元数据作为列表存储
    metadata = [item['metadata'] for item in batch]
    
    return {
        'image': images,
        'dem': dems,
        'mask': masks,
        'polygon': polygons,  # 保持为列表形式
        'metadata': metadata
    } 