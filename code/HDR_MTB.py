import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_grayscale(image):
    """將 BGR 影像轉換成灰階影像"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def median_threshold_bitmap(gray_img):
    """根據灰階影像的中位數進行二值化，生成 Bitmap"""
    med = np.median(gray_img)
    # 將像素大於等於中位數的部分設為 255，其餘設為 0
    bitmap = (gray_img >= med).astype(np.uint8) * 255
    return bitmap


def align_mtb_with_offsets(ref_img, img, levels=5):
    """
    使用 MTB 算法對齊兩幅影像，並記錄每一層金字塔的最佳位移。

    參數：
        ref_img : 參考影像 (BGR 格式)
        img     : 待對齊影像 (BGR 格式)
        levels  : 金字塔層數（預設 5 層）

    返回：
        aligned_img : 對齊後的影像
        offsets     : 一個列表，每個元素為 (level, dx, dy)
    """
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    shift = [0, 0]
    offsets = []

    # 從最低解析度開始，逐層尋找最佳位移
    for level in reversed(range(levels)):
        scale = 2 ** level
        small_ref = cv2.resize(ref_gray, (ref_gray.shape[1] // scale, ref_gray.shape[0] // scale),
                               interpolation=cv2.INTER_AREA)
        small_img = cv2.resize(img_gray, (img_gray.shape[1] // scale, img_gray.shape[0] // scale),
                               interpolation=cv2.INTER_AREA)

        med_ref = np.median(small_ref)
        med_img = np.median(small_img)
        bitmap_ref = (small_ref >= med_ref).astype(np.uint8)
        bitmap_img = (small_img >= med_img).astype(np.uint8)

        best_offset = [0, 0]
        min_diff = np.inf

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                candidate = [shift[0] * 2 + dx, shift[1] * 2 + dy]
                shifted = np.roll(bitmap_img, shift=candidate, axis=(0, 1))
                diff = np.sum(np.abs(bitmap_ref - shifted))
                if diff < min_diff:
                    min_diff = diff
                    best_offset = candidate
        shift = best_offset
        offsets.append((level, best_offset[0], best_offset[1]))

    aligned_img = np.roll(img, shift=tuple(shift), axis=(0, 1))
    return aligned_img, offsets


def main():
    # 輸入影像資料夾路徑
    folder = input("請輸入影像資料夾路徑：")
    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
    images = []
    for f in filenames:
        img = cv2.imread(os.path.join(folder, f))
        if img is not None:
            images.append(img)

    # 檢查是否有圖片
    if len(images) == 0:
        print("找不到影像！")
        return

    # 1. 將每幅影像轉換成灰階圖並儲存
    for idx, img in enumerate(images):
        gray = convert_to_grayscale(img)
        gray_filename = os.path.join(folder, f"gray_{idx}.jpg")
        cv2.imwrite(gray_filename, gray)

        # 2. 根據中位數進行二值化生成 Bitmap 圖像
        bitmap = median_threshold_bitmap(gray)
        bitmap_filename = os.path.join(folder, f"bitmap_{idx}.jpg")
        cv2.imwrite(bitmap_filename, bitmap)

    # 新增一個資料夾，用於存放對齊後的影像（保持原檔名）
    aligned_folder = os.path.join(folder, "aligned")
    if not os.path.exists(aligned_folder):
        os.makedirs(aligned_folder)

    # 以第一張影像作為參考影像
    ref_img = images[0]
    # 將參考影像直接複製到新資料夾
    cv2.imwrite(os.path.join(aligned_folder, filenames[0]), ref_img)

    # 對其他影像進行對齊，並以原檔名儲存
    for idx in range(1, len(images)):
        candidate_img = images[idx]
        aligned_img, offsets = align_mtb_with_offsets(ref_img, candidate_img, levels=5)
        output_path = os.path.join(aligned_folder, filenames[idx])
        cv2.imwrite(output_path, aligned_img)
        print(f"對齊 {filenames[idx]} 完成，儲存至 {output_path}")

    print("全部影像已對齊並儲存於資料夾：", aligned_folder)
    # 4. 將各金字塔層的最佳位移繪製成圖表
    levels_list = [item[0] for item in offsets]
    dx_list = [item[1] for item in offsets]
    dy_list = [item[2] for item in offsets]

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(levels_list, dx_list, marker='o')
    plt.xlabel("Pyramid Level")
    plt.ylabel("Horizontal Displacement (dx)")
    plt.title("Optimal dx of each level")

    plt.subplot(2, 1, 2)
    plt.plot(levels_list, dy_list, marker='o', color='orange')
    plt.xlabel("Pyramid Level")
    plt.ylabel("Vertical Displacement (dy)")
    plt.title("Optimal dy of each level")

    plt.tight_layout()
    offset_chart_filename = os.path.join(folder, "best_offset_chart.jpg")
    plt.savefig(offset_chart_filename)
    plt.show()


if __name__ == "__main__":
    main()

