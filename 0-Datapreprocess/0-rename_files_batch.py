import os

def rename_files_in_folder(folder_path):
    # 1. 获取文件夹名称（作为新前缀的核心）
    folder_name = os.path.basename(folder_path)
    print(f"开始处理文件夹：{folder_name}")
    print("-" * 50)

    # 2. 遍历文件夹中的所有文件（跳过子文件夹）
    for filename in os.listdir(folder_path):
        # 拼接完整文件路径
        old_path = os.path.join(folder_path, filename)
        
        # 只处理文件，不处理子文件夹
        if not os.path.isfile(old_path):
            continue
        
        # 3. 分割文件名和后缀（如"img-00006-00069.jpg" → 文件名"img-00006-00069"，后缀".jpg"）
        file_base, file_ext = os.path.splitext(filename)
        
        # 4. 检查原文件名是否符合格式（需包含"img-"且至少两个"-"，确保能提取末尾数字）
        if not file_base.startswith("img-") or file_base.count("-") < 2:
            print(f"跳过不符合格式的文件：{filename}（格式应为img-xxxx-yyyy）")
            continue
        
        # 5. 提取末尾数字（如"img-00006-00069" → 按"-"分割后取最后一个元素"00069"）
        parts = file_base.split("-")
        end_number = parts[-1]  # 末尾数字部分
        
        # 6. 构造新文件名（文件夹名_末尾数字.后缀）
        new_filename = f"{folder_name}_{end_number}{file_ext}"
        new_path = os.path.join(folder_path, new_filename)
        
        # 7. 重命名（避免覆盖已存在的文件）
        if os.path.exists(new_path):
            print(f"跳过已存在的文件：{new_filename}（原文件：{filename}）")
        else:
            os.rename(old_path, new_path)
            print(f"已重命名：{filename} → {new_filename}")
    
    print("-" * 50)
    print(f"文件夹{folder_name}处理完成！")

# -------------------------- 请修改为你的文件夹路径 --------------------------
root_folder = r"E:\BIT\3-research group\1-zht\8-code\CTImgDenoiseMamba\dataset"  # 例如："D:/data/20231212-3pian-wan"
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # 检查文件夹是否存在
    if not os.path.isdir(root_folder):
        print(f"错误：文件夹不存在 → {root_folder}")
    else:
        for target_folder in os.listdir(root_folder):
            target_folder = root_folder + '//' + target_folder
            rename_files_in_folder(target_folder)