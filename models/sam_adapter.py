import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, Tuple, Optional
import numpy as np
from pathlib import Path

class SAMAdapter(nn.Module):
    """SAM模型适配器，用于提取和处理SAM特征"""
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b"):
        super().__init__()
        # 确保checkpoint_path是字符串类型
        checkpoint_path = str(checkpoint_path)
        
        # 初始化SAM模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # 加载模型权重
        if torch.cuda.is_available():
            self.sam.to(device='cuda')
        else:
            self.sam.to(device='cpu')
            
        self.predictor = SamPredictor(self.sam)
        
        # 冻结SAM参数
        for param in self.sam.parameters():
            param.requires_grad = False
            
        # 特征转换层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1),
            nn.LayerNorm([512, 64, 64]),
            nn.ReLU(inplace=True)
        )
        
    def _check_image_size(self, image: torch.Tensor) -> torch.Tensor:
        """检查并调整图像大小"""
        if image.shape[-2:] != (512, 512):
            image = F.interpolate(image.unsqueeze(0), size=(512, 512), 
                                mode='bilinear', align_corners=False).squeeze(0)
        return image
        
    def get_image_embedding(self, image_batch: torch.Tensor) -> torch.Tensor:
        """获取并处理SAM图像特征
        Args:
            image_batch: 形状为[B, 3, H, W]的图像批次
        Returns:
            形状为[B, 512, 64, 64]的特征批次
        """
        # 确保图像在正确的设备上
        if image_batch.device != self.sam.device:
            image_batch = image_batch.to(self.sam.device)
        
        batch_size = image_batch.shape[0]
        features_list = []
        
        for i in range(batch_size):
            # 处理单个图像
            image = image_batch[i]
            # 确保图像大小正确
            image = self._check_image_size(image)
            # 转换为HWC格式
            image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            
            # 获取SAM特征
            self.predictor.set_image(image)
            features = self.predictor.get_image_embedding()  # [1, 256, 64, 64]
            features = features.squeeze(0)  # [256, 64, 64]
            features_list.append(features)
        
        # 堆叠所有特征
        features_batch = torch.stack(features_list, dim=0)  # [B, 256, 64, 64]
        assert features_batch.shape[1:] == (256, 64, 64), f"特征维度错误: {features_batch.shape}"
        
        # 应用特征适配器
        adapted_features = self.feature_adapter(features_batch)  # [B, 512, 64, 64]
        return adapted_features
    
    def generate_prompt_points(self, 
                             features: torch.Tensor, 
                             num_points: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成提示点
        Args:
            features: 形状为[B, C, H, W]的特征图
            num_points: 每个图像生成的点数
        Returns:
            points: 形状为[B, N, 2]的坐标点
            labels: 形状为[B, N]的标签
        """
        B, C, H, W = features.shape
        attention_map = torch.mean(features, dim=1)  # [B, H, W]
        
        # 使用注意力图选择最显著的点
        flat_attention = attention_map.view(B, -1)
        _, indices = torch.topk(flat_attention, num_points, dim=1)
        
        # 转换为坐标
        points = torch.zeros(B, num_points, 2, device=features.device)
        for b in range(B):
            points[b, :, 0] = indices[b] % W  # x坐标
            points[b, :, 1] = indices[b] // W  # y坐标
        
        # 缩放坐标到原始图像大小
        points = points * (512 / W)  # 缩放到512x512
        
        # 生成标签（假设所有点都是前景）
        labels = torch.ones(B, num_points, device=features.device)
        
        assert points.shape == (B, num_points, 2), f"点坐标维度错误: {points.shape}"
        assert labels.shape == (B, num_points), f"标签维度错误: {labels.shape}"
        
        return points, labels
    
    def get_mask_prediction(self, 
                          image_batch: torch.Tensor,
                          points: torch.Tensor,
                          labels: torch.Tensor) -> torch.Tensor:
        """使用SAM生成分割mask
        Args:
            image_batch: 形状为[B, 3, H, W]的图像批次
            points: 形状为[B, N, 2]的提示点
            labels: 形状为[B, N]的标签
        Returns:
            形状为[B, 1, H, W]的mask批次
        """
        batch_size = image_batch.shape[0]
        masks_list = []
        
        for i in range(batch_size):
            # 处理单个图像
            image = image_batch[i]
            image = self._check_image_size(image)
            image = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            
            self.predictor.set_image(image)
            
            # 获取当前图像的点和标签
            curr_points = points[i].cpu().numpy()
            curr_labels = labels[i].cpu().numpy()
            
            mask, _, _ = self.predictor.predict(
                point_coords=curr_points,
                point_labels=curr_labels,
                multimask_output=False
            )
            mask = mask[0]  # 只取第一个mask [H, W]
            masks_list.append(torch.from_numpy(mask).to(self.sam.device))
        
        # 堆叠所有mask
        masks_batch = torch.stack(masks_list, dim=0)  # [B, H, W]
        masks_batch = masks_batch.unsqueeze(1)  # [B, 1, H, W]
        
        assert masks_batch.shape[1:] == (1, 512, 512), f"Mask维度错误: {masks_batch.shape}"
        
        return masks_batch.float()
    
    def forward(self, image_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            image_batch: 形状为[B, 3, H, W]的图像批次
        Returns:
            包含以下键的字典:
            - features: [B, 512, 64, 64] 特征图
            - points: [B, N, 2] 提示点坐标
            - labels: [B, N] 提示点标签
            - masks: [B, 1, H, W] 预测的mask
        """
        features = self.get_image_embedding(image_batch)
        points, labels = self.generate_prompt_points(features)
        masks = self.get_mask_prediction(image_batch, points, labels)
        
        return {
            'features': features,  # [B, 512, 64, 64]
            'points': points,      # [B, N, 2]
            'labels': labels,      # [B, N]
            'masks': masks        # [B, 1, H, W]
        } 