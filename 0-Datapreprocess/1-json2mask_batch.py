import json
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
from labelme import utils


def process_json_files(json_dir, save_root, label_map):
    """
    批量处理JSON文件, 转换为图像和标签图
    
    参数:
        json_dir: JSON文件所在文件夹路径
        save_root: 输出数据集根目录
        label_map: 类别映射字典 (如{"_background_label_": 0, "rusting": 2, "body": 1}) 
    """
    # 自动创建输出目录结构
    os.makedirs(save_root, exist_ok=True)
    img_save_dir = osp.join(save_root, "JPEGImages")
    label_save_dir = osp.join(save_root, "SegmentationClass")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    # 遍历所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if len(json_files) == 0:
        print(f"错误：在{json_dir}中未找到JSON文件!")

    for json_filename in json_files:
        json_path = osp.join(json_dir, json_filename)
        file_prefix = json_filename.replace(".json", "")  # 提取文件名前缀
        
        # 1. 读取并解析JSON文件
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 2. 提取原始图像 (优先从imageData解码, 否则从imagePath读取) 
        if "imageData" in data and data["imageData"]:
            img = utils.img_b64_to_arr(data["imageData"])  # Base64解码为数组
        else:
            img_path = osp.join(json_dir, data["imagePath"])
            if not osp.exists(img_path):
                print(f"警告：{json_filename}的imagePath不存在, 跳过该文件！")
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
        
        # 3. 生成单通道标签图
        lbl, _ = utils.shapes_to_label(
            img_shape=img.shape,
            shapes=data["shapes"],
            label_name_to_value=label_map,
        )
        
        # 4. 保存原图和标签图
        img_save_path = osp.join(img_save_dir, f"{file_prefix}.jpg")
        Image.fromarray(img).save(img_save_path)
        
        label_save_path = osp.join(label_save_dir, f"{file_prefix}.png")
        utils.lblsave(label_save_path, lbl)  # 确保标签图为单通道无压缩
        
        print(f"已处理：{json_filename} → 原图：{img_save_path}, 标签：{label_save_path}")
    
    print(f"\n批量转换完成! 共处理{len(json_files)}个JSON文件, 输出路径：{save_root}")


def main():
    # -------------------------- 仅需用户修改以下3处配置 --------------------------
    ROOT_DIR = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset"  # JSON文件所在文件夹
    SAVE_ROOT = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset\0-processed"  # 输出根目录
    LABEL_MAP = {"_background_label_": 0, "rusting": 2, "body": 1}  # 类别映射 (背景+标注类别) 
    # -------------------------- Multi Folder --------------------------
    # for JSON_DIR in os.listdir(ROOT_DIR):
    #     JSON_DIR = ROOT_DIR + '//' + JSON_DIR
    #     # 调用核心处理函数
    #     process_json_files(
    #         json_dir=JSON_DIR,
    #         save_root=SAVE_ROOT,
    #         label_map=LABEL_MAP
    #     )
    # -------------------------- Single Folder --------------------------
    JSON_DIR = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset\20231012-K7XY-152-medium-x-y"
    LABEL_MAP = {"_background_label_": 0, "rusting": 1, "body": 2}  # 类别映射 (背景+标注类别) 
    process_json_files(
    json_dir=JSON_DIR,
    save_root=SAVE_ROOT,
    label_map=LABEL_MAP
    )


if __name__ == '__main__':
    main()