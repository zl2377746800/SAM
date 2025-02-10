# 滑坡检测系统技术文档

## 1. 项目概述

本项目是一个基于深度学习的滑坡检测系统，使用 SAM (Segment Anything Model) 作为基础模型，结合 DEM (数字高程模型) 数据进行滑坡区域的精确识别。

## 2. 系统架构

### 2.1 核心组件

- **主程序 (main.py)**: 程序入口，负责训练流程的协调和执行
- **滑坡检测模型 (EnhancedLandslideDetectionModel)**: 基于 SAM 的改进模型
- **数据集处理 (LandslideDataset)**: 自定义数据集类，处理遥感图像和 DEM 数据
- **SAM 模型下载器 (download_sam.py)**: 负责下载和管理 SAM 预训练模型

### 2.2 目录结构

project/ 
├── main.py # 主程序 
├── landslide_detection.py # 检测模型实现 
├── download_sam.py # SAM模型下载器 
├── dataset/ # 数据集目录 
│ └── landslide/ 
│ └── polygon_coordinate/ # 标注数据 
├── logs/ # 日志目录 
└── models/ # 模型存储目录 
└── checkpoints/ # 训练检查点

## 3. 核心功能实现

### 3.1 数据处理

- **批次大小**: 8
- **数据增强**: 支持图像翻转、旋转等增强方法
- **数据加载器**: 多进程数据加载 (num_workers=8)

### 3.2 模型架构

基于 SAM 模型的改进版本，主要特点：
- 集成了 DEM 数据处理能力
- 针对滑坡检测任务进行了优化
- 支持多尺度特征提取

### 3.3 训练配置
python config = 
{ 'batch_size': 8, 
'num_workers': 8, 
'num_epochs': 50, 
'accumulation_steps': 2, 
'sam_type': 'vit_b', 
'device': 'cuda' }

### 3.4 可视化功能

系统提供了完整的可视化支持：
- 原始图像显示
- DEM 数据可视化
- 真实标签展示
- 预测结果对比

## 4. 日志系统

### 4.1 日志配置

- 日志保存位置: `logs/training_[timestamp].log`
- 日志级别: INFO
- 记录内容:
  - 训练过程
  - 错误信息
  - 模型保存状态
  - 数据集信息

### 4.2 日志格式
%(asctime)s - %(levelname)s - %(message)s

## 5. 模型保存

### 5.1 检查点保存

- 最佳模型: `models/checkpoints/best_model.pth`
- 最新模型: `models/checkpoints/last_model.pth`

## 6. 使用说明

### 6.1 环境要求

- Python 3.7+
- PyTorch
- CUDA 支持（推荐）
- 足够的 GPU 内存（针对 batch_size=8）

### 6.2 运行步骤

1. 确保数据集位于正确位置
2. 下载 SAM 预训练模型
3. 运行训练脚本：
python main.py

### 6.3 输出说明

- 训练日志将保存在 `logs` 目录
- 可视化结果将保存在 `visualizations` 目录
- 模型检查点将保存在 `models/checkpoints` 目录

## 7. 注意事项

1. 确保有足够的磁盘空间存储数据集和模型
2. 推荐使用 GPU 进行训练
3. 定期检查日志文件了解训练状态
4. 注意备份重要的模型检查点

## 8. 性能优化建议

1. 根据 GPU 内存调整 batch_size
2. 适当调整 num_workers 以匹配 CPU 核心数
3. 考虑使用混合精度训练
4. 根据实际需求调整 accumulation_steps

## 9. 故障排除

常见问题及解决方案：

1. 内存不足
   - 减小 batch_size
   - 减少 num_workers

2. 模型不收敛
   - 检查学习率设置
   - 验证数据预处理流程
   - 检查损失函数实现

3. 数据加载错误
   - 确认数据集路径正确
   - 验证数据格式是否符合要求

## 10. 使用方法

### 10.1 运行测试模型

要运行测试模型并进行滑坡检测，请按照以下步骤操作：

1. 确保您已安装所需的库，包括 `torch`, `torchvision`, `PIL`, 和 `matplotlib`。
2. 确保您已下载并准备好模型权重文件，路径应与代码中的路径一致。
3. 修改 `test_model.py` 中的 `image_path` 和 `dem_path` 变量，以指向您要测试的图像和 DEM 文件。
4. 在命令行中运行以下命令：

   ```bash
   python test_model.py
   ```

5. 运行后，程序将加载模型，进行推理，并显示原始图像、DEM 和预测结果的可视化。
