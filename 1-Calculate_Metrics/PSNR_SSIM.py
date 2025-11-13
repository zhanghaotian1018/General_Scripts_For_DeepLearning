import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse

def calculate_psnr(image1, image2):
    """
    计算两张图片之间的峰值信噪比(PSNR)
    
    参数:
        image1: 第一张图片(numpy数组)
        image2: 第二张图片(numpy数组)
    
    返回:
        psnr_value: PSNR值(分贝dB)
    """
    # 确保图片尺寸相同[9](@ref)
    if image1.shape != image2.shape:
        raise ValueError("输入图片的尺寸必须相同")
    
    # 计算均方误差(MSE)[9](@ref)
    mse = np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)
    
    # 处理完全相同的图片[9](@ref)
    if mse == 0:
        return float('inf')
    
    # 计算PSNR[1,9](@ref)
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def calculate_ssim(image1, image2):
    """
    计算两张图片之间的结构相似性指数(SSIM)
    
    参数:
        image1: 第一张图片(numpy数组)
        image2: 第二张图片(numpy数组)
    
    返回:
        ssim_value: SSIM值(-1到1之间)
    """
    # 确保图片尺寸相同[9](@ref)
    if image1.shape != image2.shape:
        raise ValueError("输入图片的尺寸必须相同")
    
    # 转换为灰度图像计算SSIM[1](@ref)
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
        gray2 = image2
    
    # 计算SSIM[1,13](@ref)
    ssim_value, _ = ssim(gray1, gray2, full=True)
    return ssim_value

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='计算两张图片的PSNR和SSIM')
    parser.add_argument('image1_path', type=str, help='第一张图片的路径')
    parser.add_argument('image2_path', type=str, help='第二张图片的路径')
    
    args = parser.parse_args()
    
    try:
        # 读取图片[1](@ref)
        image1 = cv2.imread(args.image1_path)
        image2 = cv2.imread(args.image2_path)
        
        # 检查图片是否成功读取
        if image1 is None:
            raise ValueError(f"无法读取图片: {args.image1_path}")
        if image2 is None:
            raise ValueError(f"无法读取图片: {args.image2_path}")
        
        print(f"图片1尺寸: {image1.shape}")
        print(f"图片2尺寸: {image2.shape}")
        
        # 计算PSNR和SSIM
        psnr_value = calculate_psnr(image1, image2)
        ssim_value = calculate_ssim(image1, image2)
        
        # 输出结果
        print("\n图像质量评估结果:")
        print(f"PSNR (峰值信噪比): {psnr_value:.2f} dB")
        print(f"SSIM (结构相似性指数): {ssim_value:.4f}")
        
        # 结果解读[11](@ref)
        print("\n结果解读:")
        if psnr_value == float('inf'):
            print("PSNR: 图片完全相同")
        elif psnr_value > 40:
            print("PSNR: 图像质量极好（肉眼难以分辨差异）")
        elif psnr_value > 30:
            print("PSNR: 图像质量良好")
        elif psnr_value > 20:
            print("PSNR: 图像质量较差（明显失真）")
        else:
            print("PSNR: 图像质量严重失真")
        
        if ssim_value > 0.95:
            print("SSIM: 图像结构非常相似")
        elif ssim_value > 0.8:
            print("SSIM: 图像结构较为相似")
        elif ssim_value > 0.6:
            print("SSIM: 图像结构有差异")
        else:
            print("SSIM: 图像结构差异较大")
            
    except Exception as e:
        print(f"错误: {e}")
        print("请检查图片路径是否正确，且图片格式受支持（如.jpg, .png等）")

if __name__ == "__main__":
    main()