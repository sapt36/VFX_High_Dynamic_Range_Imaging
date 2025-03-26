import os
import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time


# ==========================================================
# 1. 基礎輸入 / 輸出程式
# ==========================================================
def select_image_folder():
    """
    模擬使用者選擇資料夾的虛擬函式。
    實際情況中，可使用檔案對話框或 argparse 來選擇資料夾。
    """
    folder_path = input("請輸入影像資料夾的路徑：")
    return folder_path


def load_images_and_exposures(folder_path):
    """
    從指定資料夾中讀取影像。
    為簡化處理，我們假設：
      1) 所有影像檔名格式如 'img_<曝光時間>.jpg'，
         或其他可解析出 t_k 的格式。
      2) 所有影像解析度相同、皆為 8-bit 並且已對齊。

    傳回值：
        images (list of np.ndarray): 形狀為 (H, W, 3) 的影像陣列清單 (0~255)。
        times (list of float): 曝光時間清單 (單位秒或其他)。
    """
    file_list = sorted(os.listdir(folder_path))
    images = []
    times = []

    for fname in file_list:
        # 判斷檔名是否是常見影像格式
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            name_part = os.path.splitext(fname)[0]

            # 嘗試從檔名中解析曝光時間；若失敗，使用1.0作為預設
            try:
                exposure_str = name_part.split('_')[-1]
                exposure_time = float(exposure_str)
            except ValueError:
                exposure_time = 1.0

            full_path = os.path.join(folder_path, fname)
            img = np.array(Image.open(full_path), dtype=np.uint8)  # 0~255
            images.append(img)
            times.append(exposure_time)

    # 轉成 numpy array
    times = np.array(times, dtype=np.float32)
    print("已讀取 {} 張影像。".format(len(images)))
    return images, times


def write_hdr_image(hdr_data, filename):
    """
    將 Radiance RGBE 格式的 HDR 檔案 (.hdr) 寫出。
    這是一個簡單示範的實作方式。
    若需要更完善的寫出方法，可考慮使用如 "imageio"、OpenEXR 等函式庫。
    """
    H, W, C = hdr_data.shape
    assert C == 3, "HDR 資料必須為 3 通道(RGB)。"

    with open(filename, 'wb') as f:
        header = f"""#?RADIANCE
FORMAT=32-bit_rle_rgbe
EXPOSURE=1.0

-Y {H} +X {W}
""".encode('ascii')
        f.write(header)

        for i in range(H):
            row_rgbe = bytearray()
            for j in range(W):
                r, g, b = hdr_data[i, j, :]
                r = max(r, 1e-9)
                g = max(g, 1e-9)
                b = max(b, 1e-9)
                max_c = max(r, g, b)
                e = np.floor(np.log2(max_c)) if max_c > 1e-9 else 0
                f_scale = np.power(2.0, -e)

                rr = int(r * f_scale * 256.0)
                gg = int(g * f_scale * 256.0)
                bb = int(b * f_scale * 256.0)
                e = int(e + 128)
                row_rgbe.extend([rr & 0xFF, gg & 0xFF, bb & 0xFF, e & 0xFF])
            f.write(row_rgbe)


# ==========================================================
# 2. Estimation-theoretic HDR 合成示範
# ==========================================================
def weight_function_estimation(I, eps=0.02, sat=0.98):
    """
    I: shape = (N, H, W, C), 已正規化到 [0, 1]
    回傳同形狀陣列 w，元素為0或1：
      - 太暗 (< eps) 或太亮 (> sat) 則權重=0
      - 否則權重=1
    """
    w = np.ones_like(I, dtype=np.float32)
    w[I < eps] = 0.0
    w[I > sat] = 0.0
    return w


def construct_hdr_estimation_theoretic(images, times, eps=0.02, sat=0.98):
    """
    以「估計理論」的角度（簡化版）合成多張曝光影像成 HDR (向量化版本)。

    參數：
        images: list[np.ndarray]，長度 N，每張 shape=(H,W,3)，像素 0~255
        times:  np.ndarray (N, )，曝光時間
        eps, sat: 用於過暗 / 過亮權重判定的閾值
    回傳：
        hdr: np.ndarray, shape = (H, W, 3)，每個通道為浮點數(線性HDR)。
    """
    # 假設 N = len(images)
    N = len(images)
    if N == 0:
        raise ValueError("沒有輸入任何影像。")

    H, W, C = images[0].shape

    # (1) 將所有影像合併成一個4D array: shape = (N, H, W, C)
    #     並一次性正規化到 [0,1]
    stack = np.stack(images, axis=0).astype(np.float32) / 255.0  # shape=(N,H,W,C)

    # (2) 計算每個像素在所有曝光下的權重 w_{n,h,w,c}
    w = weight_function_estimation(stack, eps, sat)  # shape = (N,H,W,C)

    # (3) 針對曝光時間，需做廣播(broadcast)：
    #     times shape = (N,) => (N,1,1,1)
    times_4d = times.reshape(N, 1, 1, 1)  # 方便與 stack 相乘

    # (4) 分子 numerator = Σ_n [ w_n * I_n * t_n ]
    numerator = np.sum(w * stack * times_4d, axis=0)  # shape=(H,W,C)

    # (5) 分母 denominator = Σ_n [ w_n * t_n^2 ]
    denominator = np.sum(w * (times_4d ** 2), axis=0)  # shape=(H,W,C)

    # (6) 避免除以零，將 denominator < 1e-12 的位置視為 0 (或其他fallback)
    hdr = np.zeros((H, W, C), dtype=np.float32)
    mask_valid = (denominator > 1e-12)
    hdr[mask_valid] = numerator[mask_valid] / denominator[mask_valid]

    return hdr


# 當繪製相機反應曲線時，我們對每條曲線進行平滑處理
def smooth_curve(curve, window_length=51, polyorder=3):
    """
    使用 Savitzky-Golay 平滑濾波器對相機反應曲線進行平滑處理。
    window_length: 窗口大小（必須是奇數，越大平滑效果越強）
    polyorder: 多項式的階數
    """
    return savgol_filter(curve, window_length=window_length, polyorder=polyorder)


# 相機反應曲線繪製函式
def plot_camera_response_curves(g_channels, folder_path):
    """
    繪製並同時顯示 4 張圖表：
      1) Red channel
      2) Green channel
      3) Blue channel
      4) RGB 三色疊加
    """
    # x 軸為 0~255 的 pixel value
    x_vals = np.arange(256)

    # 建立 2x2 的子圖 (subplots)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Camera Response Curves', fontsize=16)

    # 第一張子圖: Red
    axs[0, 0].plot(x_vals, g_channels[0], 'r')
    axs[0, 0].set_title("Red")
    axs[0, 0].set_xlabel('log exposure X')
    axs[0, 0].set_ylabel('pixel value Z')
    axs[0, 0].grid(True)

    # 第二張子圖: Green
    axs[0, 1].plot(x_vals, g_channels[1], 'g')
    axs[0, 1].set_title("Green")
    axs[0, 1].set_xlabel('log exposure X')
    axs[0, 1].set_ylabel('pixel value Z')
    axs[0, 1].grid(True)

    # 第三張子圖: Blue
    axs[1, 0].plot(x_vals, g_channels[2], 'b')
    axs[1, 0].set_title("Blue")
    axs[1, 0].set_xlabel('log exposure X')
    axs[1, 0].set_ylabel('pixel value Z')
    axs[1, 0].grid(True)

    # 第四張子圖: RGB 三色疊加
    colors = ['r', 'g', 'b']
    labels = ['red', 'green', 'blue']
    for c in range(3):
        axs[1, 1].plot(x_vals, g_channels[c], colors[c], label=labels[c])
    axs[1, 1].set_title("RGB Combined")
    axs[1, 1].set_xlabel('log exposure X')
    axs[1, 1].set_ylabel('pixel value Z')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 讓各子圖的排版更緊湊
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 預留空間給 suptitle

    # 儲存圖片
    curve_out = os.path.join(folder_path, "camera_response_curve.png")
    plt.savefig(curve_out)
    print(f"相機反應曲線已儲存：{curve_out}")

    # 顯示圖表 (如不需立即顯示，可註解掉)
    # plt.show()


# ==========================================================
# 3. Tone Mapping (與先前程式相同)
# ==========================================================
def tone_mapping_Drago(hdr_data):
    """
    使用 OpenCV 的 Drago Tone Mapping。
    """
    tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=8.0)

    # 先假設 HDR 可能非常大，把它簡單壓到 [0,1] 以避免 overrange
    max_val = np.max(hdr_data)
    scale = 1.0
    if max_val > 0:
        scale = 1.0 / max_val

    hdr_normalized = (hdr_data * scale).astype(np.float32)
    ldr_data = tonemap.process(hdr_normalized)
    ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8)
    return ldr_data


def tone_mapping_dodging_burning(hdr_data):
    """
    使用 OpenCV 的 Mantiuk 方法近似 Dodging and Burning 的局部 Tone Mapping
    """
    tonemap_mantiuk = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.0)

    max_val = np.max(hdr_data)
    scale = 1.0 / max_val if max_val > 0 else 1.0
    hdr_normalized = (hdr_data * scale).astype(np.float32)

    ldr_data = tonemap_mantiuk.process(hdr_normalized)
    ldr_data = np.nan_to_num(ldr_data, nan=0.0, posinf=1.0, neginf=0.0)
    ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8)
    return ldr_data


def tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6):
    """
    使用 Reinhard 全局 Tone Mapping。
    """
    L = 0.2126 * hdr_data[..., 0] + \
        0.7152 * hdr_data[..., 1] + \
        0.0722 * hdr_data[..., 2]

    L_log = np.log(delta + L)
    L_avg = np.exp(np.mean(L_log))

    L_scaled = (key / L_avg) * L
    L_mapped = L_scaled / (1.0 + L_scaled)

    S = L_mapped / (L + delta)
    S = np.expand_dims(S, axis=-1)

    ldr = hdr_data * S
    ldr = np.clip(ldr, 0, 1)
    ldr_data = (ldr * 255).astype(np.uint8)
    return ldr_data


# ==========================================================
# 4. Radiance Map (可視化) 與主程式
# ==========================================================
def save_radiance_map_colormap(hdr_data, output_path):
    """
    輸出 Radiance Map 的可視化(彩色)，以 Log Luminance 做色階。
    """
    L = 0.2126 * hdr_data[:, :, 0] + \
        0.7152 * hdr_data[:, :, 1] + \
        0.0722 * hdr_data[:, :, 2]
    L_log = np.log1p(L)  # log(1+L)

    L_min, L_max = L_log.min(), L_log.max()
    if L_max - L_min < 1e-10:
        L_norm = np.zeros_like(L_log)
    else:
        L_norm = (L_log - L_min) / (L_max - L_min)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(L_norm, cmap='jet', origin='upper')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log Luminance (normalized)")
    ax.set_xlabel("X-axis (width)")
    ax.set_ylabel("Y-axis (height)")
    ax.set_title("Radiance Map (Log Scale)")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Radiance Map 已儲存至：{output_path}")


def main():
    start_time = time.time()

    # 1) 讀取影像與曝光時間
    folder_path = select_image_folder()
    images, times = load_images_and_exposures(folder_path)
    if len(images) < 2:
        print("需要至少兩張不同曝光的影像！")
        return

    # 估測相機反應曲線 (Camera Response Function, CRF)
    print("開始估測相機反應曲線 (Debevec) ...")

    # OpenCV 的 CalibrateDebevec 要求輸入為 list of 8-bit images 以及對應的曝光時間
    # times shape 要為 (N,1)，所以要 reshape 一下
    times_for_opencv = times.reshape(-1, 1)  # shape=(N,1)

    # OpenCV 需要 list of uint8，若 images 已是 np.uint8 則可直接使用
    # 若不確定，可再轉一次
    images_8u = [img.astype(np.uint8) for img in images]

    calibrate = cv2.createCalibrateDebevec()
    crf = calibrate.process(images_8u, times_for_opencv)
    # crf shape=(256,1,3)，轉成 [3,256]
    g_channels = crf[:, 0, :].T

    # 繪製並輸出相機反應曲線
    plot_camera_response_curves(g_channels, folder_path)

    # 2) 使用 Estimation-theoretic 方法合成 HDR
    print("開始進行 Estimation-theoretic HDR 合成...")
    hdr_data = construct_hdr_estimation_theoretic(images, times)
    print("HDR 合成完成。")
    print("HDR 內容範圍: min={}, max={}".format(hdr_data.min(), hdr_data.max()))

    # 3) 輸出 Radiance Map 的可視化
    output_radiance_map = os.path.join(folder_path, "radiance_map_estimation.jpg")
    save_radiance_map_colormap(hdr_data, output_radiance_map)

    # 4) 進行 Tone Mapping (可搭配多種演算法)
    print("進行 Tone Mapping...")
    ldr_data_drago = tone_mapping_Drago(hdr_data)
    ldr_data_mantiuk = tone_mapping_dodging_burning(hdr_data)
    ldr_data_reinhard = tone_mapping_Reinhard(hdr_data, key=0.18)

    # 5) 輸出 HDR 與 Tone-Mapped 結果
    output_hdr = os.path.join(folder_path, "output_estimation_result.hdr")
    write_hdr_image(hdr_data, output_hdr)
    print(f"HDR 檔案輸出至: {output_hdr}")

    out_ldr_drago = os.path.join(folder_path, "output_estimation_ldr_Drago.jpg")
    Image.fromarray(ldr_data_drago).save(out_ldr_drago)
    print(f"LDR(Drago) 輸出至: {out_ldr_drago}")

    out_ldr_mantiuk = os.path.join(folder_path, "output_estimation_ldr_Mantiuk.jpg")
    Image.fromarray(ldr_data_mantiuk).save(out_ldr_mantiuk)
    print(f"LDR(Mantiuk) 輸出至: {out_ldr_mantiuk}")

    out_ldr_reinhard = os.path.join(folder_path, "output_estimation_ldr_Reinhard.jpg")
    Image.fromarray(ldr_data_reinhard).save(out_ldr_reinhard)
    print(f"LDR(Reinhard) 輸出至: {out_ldr_reinhard}")

    end_time = time.time()
    print("程式執行時間：{:.2f} 秒".format(end_time - start_time))


if __name__ == "__main__":
    main()
