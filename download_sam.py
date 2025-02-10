import torch
import os
from segment_anything import sam_model_registry
from tqdm import tqdm
import requests
import hashlib
from pathlib import Path

def calculate_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def verify_model_file(file_path: str, expected_hash: str) -> bool:
    """验证模型文件的完整性"""
    if not os.path.exists(file_path):
        return False
    actual_hash = calculate_file_hash(file_path)
    return actual_hash == expected_hash

def download_file_with_progress(url: str, save_path: str):
    """带进度条的文件下载"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=f"下载 {os.path.basename(save_path)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_sam_model(model_type="vit_b"):
    """下载SAM模型权重"""
    print(f"开始检查 SAM {model_type} 模型...")
    
    # 模型URL和哈希值映射
    model_info = {
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "hash": "4b8939a88964f0f4ff5f5b2642c598a6"
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "hash": "0b3195507c641ddb6910d2bb5adee89c"
        },
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "hash": "01ec64d29a2fca3f0661936605ae66f8"
        }
    }
    
    if model_type not in model_info:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建models目录
    os.makedirs("models", exist_ok=True)
    
    # 设置保存路径
    save_path = f"models/sam_{model_type}.pth"
    
    # 检查文件是否存在且完整
    if os.path.exists(save_path):
        if verify_model_file(save_path, model_info[model_type]["hash"]):
            print(f"模型文件已存在且完整: {save_path}")
            return save_path
        else:
            print(f"发现损坏的模型文件，正在删除: {save_path}")
            os.remove(save_path)
    
    # 下载模型
    url = model_info[model_type]["url"]
    try:
        print(f"开始下载模型: {url}")
        download_file_with_progress(url, save_path)
        
        # 验证下载的文件
        if verify_model_file(save_path, model_info[model_type]["hash"]):
            print(f"模型下载完成并验证通过: {save_path}")
            return save_path
        else:
            print("模型文件验证失败，可能下载不完整")
            os.remove(save_path)
            return None
            
    except Exception as e:
        print(f"下载失败: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return None

if __name__ == "__main__":
    # 清理已有的模型文件
    for file in Path('models').glob('*.pth'):
        file.unlink()
    print("已清理旧文件")
    
    # 下载基础版本的SAM模型
    model_path = download_sam_model("vit_b")
    if model_path:
        print(f"模型准备完成: {model_path}")
    else:
        print("模型下载失败") 