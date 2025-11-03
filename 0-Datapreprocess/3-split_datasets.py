import os
import shutil
import random
from glob import glob

def split_dataset(img_dir, gt_dir, save_root, img_suffix=('.jpg', '.png') , gt_suffix=('.png', '.json') , seed=42) :
    """
    划分图像和GT数据集为train/val/test (8:1:1)
    
    参数:
        img_dir: 原始图像文件夹路径 (如./data/img)
        gt_dir: 对应GT文件夹路径 (如./data/gt)
        save_root: 划分后数据集保存根目录 (如./dataset_split)
        img_suffix: 图像文件后缀 (元组形式，支持多种格式)
        gt_suffix: GT文件后缀 (元组形式，支持多种格式)
        seed: 随机种子 (固定以保证划分结果可复现)
    """
    # 固定随机种子
    random.seed(seed) 
    
    # -------------------------- 1. 检查输入文件夹并获取文件列表 --------------------------
    # 验证源文件夹是否存在
    if not os.path.isdir(img_dir) :
        raise ValueError(f"图像文件夹不存在: {img_dir}") 
    if not os.path.isdir(gt_dir) :
        raise ValueError(f"GT文件夹不存在: {gt_dir}") 
    
    # 获取所有图像文件 (根据后缀过滤)
    img_files = []
    for suffix in img_suffix:
        img_files.extend(glob(os.path.join(img_dir, f'*{suffix}') ) ) 
    img_files = sorted(img_files)   # 排序保证文件顺序一致
    if not img_files:
        raise ValueError(f"在{img_dir}中未找到符合后缀{img_suffix}的图像文件") 
    
    # 提取图像文件名前缀 (用于匹配GT)
    # 例：./img/abc.jpg → 前缀为abc
    img_prefixes = [os.path.splitext(os.path.basename(f) ) [0] for f in img_files]
    
    # 检查GT文件是否与图像一一对应
    gt_files = []
    for prefix in img_prefixes:
        # 查找与图像前缀匹配的GT文件 (支持多种后缀)
        matched_gt = None
        for suffix in gt_suffix:
            gt_candidate = os.path.join(gt_dir, f'{prefix}{suffix}') 
            if os.path.exists(gt_candidate) :
                matched_gt = gt_candidate
                break
        if not matched_gt:
            raise FileNotFoundError(f"未找到与图像{prefix}匹配的GT文件 (检查后缀{gt_suffix})") 
        gt_files.append(matched_gt) 
    
    print(f"找到{len(img_files) }对图像-GT文件, 开始划分...") 
    
    # -------------------------- 2. 按8:1:1比例划分数据集 --------------------------
    total = len(img_files) 
    # 计算各子集数量 (保证整数，val和test数量可能差1)
    train_size = int(total * 0.8) 
    val_size = int(total * 0.1) 
    test_size = total - train_size - val_size
    
    # 生成随机索引并划分
    indices = list(range(total) ) 
    random.shuffle(indices)   # 打乱顺序 (固定seed保证结果一致)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 构建各子集的文件列表
    subsets = {
        'train': {
            'img': [img_files[i] for i in train_indices],
            'gt': [gt_files[i] for i in train_indices]
        },
        'val': {
            'img': [img_files[i] for i in val_indices],
            'gt': [gt_files[i] for i in val_indices]
        },
        'test': {
            'img': [img_files[i] for i in test_indices],
            'gt': [gt_files[i] for i in test_indices]
        }
    }
    
    # -------------------------- 3. 创建目标目录并复制文件 --------------------------
    # 创建目录结构：save_root/[train/val/test]/[img/gt]
    for subset in subsets:
        for dtype in ['img', 'gt']:
            dir_path = os.path.join(save_root, subset, dtype) 
            os.makedirs(dir_path, exist_ok=True) 
            print(f"创建目录: {dir_path}") 
    
    # 复制文件到对应目录
    for subset, data in subsets.items() :
        print(f"\n处理{subset}集 ({len(data['img']) }个样本)...") 
        for img_path, gt_path in zip(data['img'], data['gt']) :
            # 复制图像
            img_dst = os.path.join(save_root, subset, 'img', os.path.basename(img_path) ) 
            shutil.copy2(img_path, img_dst)   # copy2保留文件元数据
            
            # 复制GT
            gt_dst = os.path.join(save_root, subset, 'gt', os.path.basename(gt_path) ) 
            shutil.copy2(gt_path, gt_dst) 
            
            # 打印进度 (每10个文件打印一次)
            if (subsets[subset]['img'].index(img_path)  + 1)  % 10 == 0:
                print(f"已复制{subset}集 {subsets[subset]['img'].index(img_path)  + 1}/{len(data['img']) }个文件") 
    
    print("\n数据集划分完成!") 
    print(f"总样本数: {total}") 
    print(f"训练集: {train_size} 验证集: {val_size} 测试集: {test_size}") 
    print(f"结果保存至: {save_root}") 


if __name__ == '__main__':
    # -------------------------- 请修改以下配置 --------------------------
    IMG_DIR = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset\0-processed\JPEGImages"  # 原始图像文件夹路径
    GT_DIR = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset\0-processed\SegmentationClass"       # 对应GT文件夹路径
    SAVE_ROOT = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset\0-processed\datasets"      # 划分后数据集保存根目录
    IMG_SUFFIX = ('.jpg', '.png')        # 图像文件后缀 (根据实际情况修改)
    GT_SUFFIX = ('.png', '.json')        # GT文件后缀 (根据实际情况修改，如标签图是.png，JSON是.json)
    RANDOM_SEED = 42                    # 固定随机种子，保证划分结果可复现
    # -------------------------------------------------------------------------
    
    # 调用划分函数
    split_dataset(
        img_dir=IMG_DIR,
        gt_dir=GT_DIR,
        save_root=SAVE_ROOT,
        img_suffix=IMG_SUFFIX,
        gt_suffix=GT_SUFFIX,
        seed=RANDOM_SEED
    ) 