import json
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
from labelme import utils  # LabelMe的工具函数，用于解析JSON和生成标签

# -------------------------- 1. 请修改以下3处配置 --------------------------
JSON_DIR = r"./your_json_dir"  # 存放所有JSON文件的文件夹路径（如./data/json/）
SAVE_ROOT = r"./dataset"      # 输出数据集根目录（会自动创建子文件夹）
# 类别映射：背景=0，按您的标注label顺序分配像素值（必须与JSON中的label一致！）
LABEL_MAP = {"_background_": 0, "rusting": 1, "body": 2}
# -------------------------------------------------------------------------

# 自动创建输出目录结构（符合图像分割数据集通用规范）
os.makedirs(SAVE_ROOT, exist_ok=True)
IMG_SAVE_DIR = osp.join(SAVE_ROOT, "JPEGImages")  # 存放原始图像
LABEL_SAVE_DIR = osp.join(SAVE_ROOT, "SegmentationClass")  # 存放单通道标签图
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(LABEL_SAVE_DIR, exist_ok=True)

# 遍历所有JSON文件
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
if len(json_files) == 0:
    print(f"错误：在{JSON_DIR}中未找到JSON文件！")
    sys.exit()

for json_filename in json_files:
    json_path = osp.join(JSON_DIR, json_filename)
    file_prefix = json_filename.replace(".json", "")  # 文件名前缀（如img-00004-00742）
    
    # 1. 读取并解析JSON文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. 提取原始图像（从imageData解码，若没有则从imagePath读取）
    if "imageData" in data and data["imageData"]:
        # 解码Base64格式的图像数据（您的JSON中包含imageData，优先用此方式）
        img = utils.img_b64_to_arr(data["imageData"])  # 转为numpy数组（HWC，RGB）
    else:
        # 若JSON中无imageData，读取imagePath对应的原图（需确保路径正确）
        img_path = osp.join(JSON_DIR, data["imagePath"])
        if not osp.exists(img_path):
            print(f"警告：{json_filename}的imagePath不存在，跳过该文件！")
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
    
    # 3. 生成单通道标签图（核心步骤）
    # 输入：图像尺寸（H,W）、标注shapes、类别映射；输出：H×W的单通道数组
    lbl, _ = utils.shapes_to_label(
        img_shape=img.shape,
        shapes=data["shapes"],
        label_name_to_value=LABEL_MAP,
        background_label=0  # 背景像素值设为0
    )
    
    # 4. 保存原图和标签图（确保文件名对应）
    # 保存原图（转为JPEG格式，符合多数模型输入习惯）
    img_save_path = osp.join(IMG_SAVE_DIR, f"{file_prefix}.jpg")
    Image.fromarray(img).save(img_save_path)
    
    # 保存标签图（单通道8位灰度图，关键！避免压缩导致像素值失真）
    label_save_path = osp.join(LABEL_SAVE_DIR, f"{file_prefix}.png")
    utils.lblsave(label_save_path, lbl)  # LabelMe的lblsave确保标签为单通道无压缩
    
    print(f"已处理：{json_filename} → 原图：{img_save_path}，标签：{label_save_path}")

print(f"\n批量转换完成！共处理{len(json_files)}个JSON文件，输出路径：{SAVE_ROOT}")