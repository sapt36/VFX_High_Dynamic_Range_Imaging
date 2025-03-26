import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os  # 用於處理檔名

def calculate_image_metrics(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)

    # 將圖片轉換為 RGB 格式 (OpenCV 默認使用 BGR 格式)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 計算亮度 (灰度圖)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 計算亮度直方圖
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # 計算平均亮度
    mean_brightness = np.mean(gray_image)

    # 計算亮度標準差
    std_brightness = np.std(gray_image)

    # 計算色彩飽和度 (Saturation)
    # 轉換到 HLS 色彩空間
    image_hls = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
    saturation = image_hls[:, :, 2]

    # 計算色彩飽和度平均值
    mean_saturation = np.mean(saturation)

    # 計算色彩飽和度標準差
    std_saturation = np.std(saturation)

    # 取得圖像檔名 (不含路徑與副檔名)
    image_name = os.path.basename(image_path)
    image_name_without_extension = os.path.splitext(image_name)[0]

    # 顯示亮度直方圖並保存
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist)
    plt.title('Brightness Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig('brightness_histogram.png')  # 輸出直方圖圖像

    # 顯示原圖並保存
    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title(f'{image_name_without_extension}')  # 顯示檔名
    plt.axis('off')
    plt.savefig('original_image.png')  # 輸出原圖圖像
    plt.close()

    # 返回各項指標
    return {
        'mean_brightness': mean_brightness,
        'std_brightness': std_brightness,
        'mean_saturation': mean_saturation,
        'std_saturation': std_saturation
    }

def open_image_file():
    # 使用 tkinter 打開文件選擇視窗
    root = Tk()  # 創建 Tk 實例
    root.withdraw()  # 隱藏主視窗
    file_path = askopenfilename(title="Select an Image File",
                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])
    return file_path

# 彈出視窗選擇圖片並計算指標
image_path = open_image_file()
if image_path:
    metrics = calculate_image_metrics(image_path)
    print("Average Brightness:", metrics['mean_brightness'])
    print("Brightness Standard Deviation:", metrics['std_brightness'])
    print("Average Saturation:", metrics['mean_saturation'])
    print("Saturation Standard Deviation:", metrics['std_saturation'])
else:
    print("No image selected.")
