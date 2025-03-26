import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import savgol_filter
from tqdm import tqdm
import time


# 輸入影像資料夾路徑
def select_image_folder():
    """
    模擬使用者選擇資料夾的虛擬函式。
    實際情況中，可使用檔案對話框或 argparse 來選擇資料夾。
    """
    folder_path = input("請輸入影像資料夾的路徑：")
    return folder_path


# 透過檔名取得每張相片的曝光時間
def load_images_and_exposures(folder_path):
    """
    從指定資料夾中讀取影像。
    為簡化處理，我們假設：
    1) 所有影像檔名格式如 'img_<曝光時間>.jpg'，
       或其他可解析出 t_k 的格式。
    2) 所有影像解析度相同、皆為 8-bit 並且已對齊。

    傳回值：
        images (list of np.ndarray): 形狀為 (H, W, 3) 的影像陣列清單。
        times (list of float): 曝光時間清單。
    """
    file_list = sorted(os.listdir(folder_path))
    images = []
    times = []

    for fname in file_list:
        if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            # 嘗試從檔名中解析曝光時間，或使用預設值
            # 例如 "image_0.025.jpg" → 曝光時間為 0.025
            name_part = os.path.splitext(fname)[0]
            # 這是一個簡單的解析方法：以底線作為分隔符
            # 否則，讓使用者外部定義曝光時間
            try:
                exposure_str = name_part.split('_')[-1]
                exposure_time = float(exposure_str)
            except ValueError:
                # 若解析失敗，則使用預設值 1.0
                exposure_time = 1.0

            full_path = os.path.join(folder_path, fname)
            img = np.array(Image.open(full_path), dtype=np.uint8)

            images.append(img)
            times.append(exposure_time)

    # 將曝光時間轉換為 numpy 陣列
    times = np.array(times, dtype=np.float32)
    print("已讀取 {} 張影像。".format(len(images)))
    return images, times


# 使用最簡單的像素強度權重函式
def weight_function(z, lower_limit=20, upper_limit=235):
    """
    新的權重函式對過暗與過亮區域賦予較小的權重。
    這裡我們將像素值低於 lower_limit 或高於 upper_limit 的區域視為過暗或過亮，並降低它們的權重。
    """
    w = np.ones_like(z, dtype=np.float32)

    # 設置過暗像素的權重為較低值
    w[z < lower_limit] = 0.1

    # 設置過亮像素的權重為較低值
    w[z > upper_limit] = 0.1

    return w


# 課堂上介紹的 Debevec's method
def solve_g_and_lnE(Z, B, l, w):
    """
    使用 Debevec & Malik 方法的簡化版本解出 g(z) 與 ln(E)。

    參數：
        Z: 形狀 (P, N)，每一列為同一像素在 N 張影像下的取樣值。
        B: 形狀 (N,)，曝光時間的對數值。
        l: 平滑項的 lambda 值。
        w: 像素權重函式 w(z)。

    傳回值：
        g: 形狀 (256,) - 相機反應函數。
        lE: 形狀 (P,) - 每個像素取樣的對數輻射值。
    """
    # P：取樣像素數； N：影像數量
    P, N = Z.shape

    # 我們要求解 z ∈ [0..255] 的 g(z)，共 256 個值，再加上 P 個 ln(E)
    n = 256
    A = lil_matrix((P * N + n - 1, n + P), dtype=np.float32)
    b = np.zeros(P * N + n - 1, dtype=np.float32)

    # 填充方程組 (資料配適方程)
    # K = 0..P*N-1 為資料配適方程
    idx = 0
    for i in range(P):
        for j in range(N):
            z_ij = Z[i, j]
            w_ij = w[z_ij]
            A[idx, z_ij] = w_ij  # g(Z_ij)
            A[idx, n + i] = -w_ij  # -lnE_i
            b[idx] = w_ij * B[j]  # w_ij * ln(t_j)
            idx += 1

    # 加入 g(z) 的平滑性約束
    # 對於 z ∈ [1..254]，我們希望 (g(z+1) - 2g(z) + g(z-1)) 約等於 0
    for z in range(1, 255):
        A[idx, z - 1] = l * 1
        A[idx, z] = l * -2
        A[idx, z + 1] = l * 1
        idx += 1

    # 固定 g(128) = 0 作為參考
    A[idx, 128] = 1.0
    b[idx] = 0.0
    idx += 1

    # 求解方程組
    A_csr = A.tocsr()
    x = lsqr(A_csr, b)[0]  # x = [g(0..255), lnE(0..P-1)]

    g = x[0:n]
    lE = x[n:]
    return g, lE


# 優化過的建構方法
def construct_radiance_map(images, times, g):
    """
    重建每個像素的輻射圖 (以對數域表示)。
    計算方式為：
        lnE(i,j) = ( Σ_n w(Z(i,j,n)) * [ g(Z(i,j,n)) - ln(t_n) ] ) /
                   ( Σ_n w(Z(i,j,n)) )
    最後取指數以得到 E(i,j)。

    參數：
        images: 彩色影像清單，形狀 (H, W, 3)。
        times: 曝光時間陣列，形狀 (N,)。
        g: 相機反應函數，對於灰階或各通道分別計算。
    傳回值：
        hdr: 重建的 HDR 輻射圖，形狀 (H, W, 3)。
    """
    N = len(images)
    H, W, C = images[0].shape

    # 將曝光時間轉換成對數
    ln_t = np.log(times)

    # 準備輸出 HDR 陣列
    hdr = np.zeros((H, W, C), dtype=np.float32)

    # 預先計算所有可能強度的權重值
    w_arr = weight_function(np.arange(256))

    # 對於每個通道進行計算，利用向量化優化
    for c in tqdm(range(C)):
        # 準備每個像素的強度值 (在 N 張影像中該像素的強度)
        vals = np.array([img[:, :, c] for img in images], dtype=np.uint8)

        # 將權重應用到每個像素強度
        wghts = w_arr[vals]

        # 計算各通道的 top 和 bot 值
        top = np.zeros((H, W), dtype=np.float32)
        bot = np.zeros((H, W), dtype=np.float32)

        for n in range(N):
            # g(c) 是相機反應函數
            top += wghts[n] * (g[c][vals[n]] - ln_t[n])
            bot += wghts[n]

        # 防止除以零，將最小值設為 1e-8
        bot = np.maximum(bot, 1e-8)

        # 計算每個像素的輻射值
        channel_map = top / bot
        hdr[:, :, c] = channel_map

    # 取指數以得到線性域的輻射值
    hdr = np.exp(hdr)
    return hdr


def write_hdr_image(hdr_data, filename):
    """
    將 Radiance RGBE 格式的 HDR 檔案 (.hdr) 寫出。
    這是一個簡單示範的實作方式。
    若需要更完善的寫出方法，可考慮使用如 "imageio" 等函式庫。
    """
    # 獲取影像尺寸
    H, W, C = hdr_data.shape
    assert C == 3, "HDR 資料必須為 3 通道。"

    # 寫出 HDR 檔案
    with open(filename, 'wb') as f:
        header = f"""#?RADIANCE
FORMAT=32-bit_rle_rgbe
EXPOSURE=1.0

-Y {H} +X {W}
""".encode('ascii')
        f.write(header)

        # 將浮點數資料轉換為 RGBE 格式
        # E 為指數，共用 RGB 分量
        # 各分量為 [0..255] 的位元組數值
        for i in range(H):
            row_rgbe = bytearray()
            for j in range(W):
                r, g, b = hdr_data[i, j, :]
                # 避免取 log(0)
                r = max(r, 1e-9)
                g = max(g, 1e-9)
                b = max(b, 1e-9)
                max_c = max(r, g, b)
                # 計算指數（以 2 為底的對數）
                e = np.floor(np.log2(max_c)) if max_c > 1e-9 else 0
                f_scale = np.power(2.0, -e)
                # 將數值縮放到 0~1 範圍
                rr = r * f_scale
                gg = g * f_scale
                bb = b * f_scale
                # 轉換到 [0..255] 範圍
                rr = int(rr * 256.0)
                gg = int(gg * 256.0)
                bb = int(bb * 256.0)
                # 調整指數偏移
                e = int(e + 128)
                row_rgbe.extend([rr & 0xFF, gg & 0xFF, bb & 0xFF, e & 0xFF])
            f.write(row_rgbe)


# 使用OpenCV_Drago
def tone_mapping_Drago(hdr_data):
    """
    將 HDR 影像進行 Tone Mapping，這裡使用 OpenCV 的 Drago 方法。

    參數：
        hdr_data: HDR 影像數據，形狀 (H, W, 3)。。

    返回：
        ldr_data: 經過 tone mapping 處理的低動態範圍影像。
    """
    # 使用 OpenCV 的 Drago Tone Mapping 方法
    tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=8.0)

    # 將 HDR 影像從 [0, 255] 範圍轉換為 [0, 1] 範圍
    hdr_normalized = hdr_data.astype(np.float32) / 255.0

    # 進行 tone mapping
    ldr_data = tonemap.process(hdr_normalized)

    # 將結果從 [0, 1] 範圍轉換回 [0, 255] 範圍
    ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8)

    return ldr_data


# 使用OpenCV_Mantiuk
def tone_mapping_dodging_burning(hdr_data):
    """
    使用 OpenCV 的 Mantiuk 方法近似 Dodging and Burning 的局部 Tone Mapping

    參數：
        hdr_data: HDR 影像數據，形狀 (H, W, 3)。。

    返回：
        ldr_data: 經過 tone mapping 處理的低動態範圍影像。。
    """
    tonemap_mantiuk = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.0)
    hdr_normalized = hdr_data.astype(np.float32) / 255.0
    ldr_data = tonemap_mantiuk.process(hdr_normalized)
    ldr_data = np.nan_to_num(ldr_data, nan=0.0, posinf=1.0, neginf=0.0)
    ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8)
    return ldr_data


# 實作Reinhard 方法
def tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6):
    """
    使用 Reinhard 全局 Tone Mapping 算法對 HDR 影像進行 Tone Mapping。

    參數：
        hdr_data: np.ndarray (H, W, 3)，HDR 影像，格式為 float32 或 float64，
                  輻射值可超出 [0,1] 範圍。
        key: 調整曝光的參數，預設為 0.18（可根據場景調整）。
        delta: 用於防止取對數時除零的小值（預設 1e-6）。

    返回：
        ldr_data: np.ndarray (H, W, 3)，LDR 影像，格式為 uint8（0-255）。

    算法步驟：
      1. 計算每個像素的亮度 L = 0.2126R + 0.7152G + 0.0722B。
      2. 計算對數平均亮度 L_avg = exp(mean(log(delta + L)))。
      3. 將亮度按比例縮放：L_scaled = (key / L_avg) * L。
      4. 應用 Reinhard 映射公式：L_mapped = L_scaled / (1 + L_scaled)。
      5. 計算縮放因子 S = L_mapped / (L + delta)，並將其應用到 RGB 各通道上。
      6. 將結果裁剪到 [0,1] 範圍，再轉換為 uint8（[0,255]）。
    """
    # Step 1: 計算亮度
    L = 0.2126 * hdr_data[:, :, 0] + 0.7152 * hdr_data[:, :, 1] + 0.0722 * hdr_data[:, :, 2]

    # Step 2: 計算對數平均亮度
    L_log = np.log(delta + L)
    L_avg = np.exp(np.mean(L_log))

    # Step 3: 縮放亮度
    L_scaled = (key / L_avg) * L

    # Step 4: 應用 Reinhard 映射公式
    L_mapped = L_scaled / (1 + L_scaled)

    # Step 5: 計算縮放因子，並對每個色彩通道應用
    S = L_mapped / (L + delta)
    S = S[:, :, np.newaxis]  # 擴展為 (H, W, 1) 方便廣播
    ldr = hdr_data * S

    # Step 6: 裁剪結果至 [0,1]，轉換為 8 位元圖像
    ldr = np.clip(ldr, 0, 1)
    ldr_data = (ldr * 255).astype(np.uint8)

    return ldr_data


# 當繪製相機反應曲線時，我們對每條曲線進行平滑處理
def smooth_curve(curve, window_length=51, polyorder=3):
    """
    使用 Savitzky-Golay 平滑濾波器對相機反應曲線進行平滑處理。
    window_length: 窗口大小（必須是奇數，越大平滑效果越強）
    polyorder: 多項式的階數
    """
    return savgol_filter(curve, window_length=window_length, polyorder=polyorder)


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

    # 平滑處理每條曲線
    g_channels_smoothed = [smooth_curve(g) for g in g_channels]

    # 第一張子圖: Red
    axs[0, 0].plot(x_vals, g_channels_smoothed[0], 'r')
    axs[0, 0].set_title("Red")
    axs[0, 0].set_xlabel('log exposure X')
    axs[0, 0].set_ylabel('pixel value Z')
    axs[0, 0].grid(True)

    # 第二張子圖: Green
    axs[0, 1].plot(x_vals, g_channels_smoothed[1], 'g')
    axs[0, 1].set_title("Green")
    axs[0, 1].set_xlabel('log exposure X')
    axs[0, 1].set_ylabel('pixel value Z')
    axs[0, 1].grid(True)

    # 第三張子圖: Blue
    axs[1, 0].plot(x_vals, g_channels_smoothed[2], 'b')
    axs[1, 0].set_title("Blue")
    axs[1, 0].set_xlabel('log exposure X')
    axs[1, 0].set_ylabel('pixel value Z')
    axs[1, 0].grid(True)

    # 第四張子圖: RGB 三色疊加
    colors = ['r', 'g', 'b']
    labels = ['red', 'green', 'blue']
    for c in range(3):
        axs[1, 1].plot(x_vals, g_channels_smoothed[c], colors[c], label=labels[c])
    axs[1, 1].set_title("RGB Combined")
    axs[1, 1].set_xlabel('log exposure X')
    axs[1, 1].set_ylabel('pixel value Z')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 讓各子圖的排版更緊湊
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 預留空間給 suptitle

    # 儲存圖片
    curve_out = os.path.join(folder_path, "camera_response_curve_smoothed.png")
    plt.savefig(curve_out)
    print(f"相機反應曲線已儲存：{curve_out}")

    # 顯示圖表
    # plt.show()


def save_radiance_map_colormap(hdr_data, output_path):
    """
    使用 matplotlib 顯示並輸出 Radiance Map。
    - 計算 Log Luminance 後做歸一化，
      並用 matplotlib 的 imshow + colorbar 可視化。
    - 同時在影像中加上橫軸、縱軸、colorbar，便於對照。

    參數：
        hdr_data (np.ndarray): HDR 影像，形狀 (H, W, 3)，float。
        output_path (str): 儲存輸出的檔案路徑 (例如 'radiance_map.png')。
    """
    # Step 1: 計算 luminance
    L = 0.2126 * hdr_data[:, :, 0] + \
        0.7152 * hdr_data[:, :, 1] + \
        0.0722 * hdr_data[:, :, 2]

    # Step 2: 取 log(1 + L) (避免 L=0 取 log 造成 -inf)
    L_log = np.log1p(L)  # log(1 + L)

    # Step 3: 正規化到 [0..1]，方便視覺化
    L_min, L_max = L_log.min(), L_log.max()
    L_log_norm = (L_log - L_min) / (L_max - L_min + 1e-12)

    # Step 4: 使用 matplotlib 繪圖
    #   - imshow 顯示 L_log_norm
    #   - 加入 colorbar、x/y label
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(L_log_norm, cmap='jet', origin='upper')
    # origin='upper' 使得影像左上角是 [0,0]；若想符合一般圖像原點在左上，可用 'upper'

    # 新增 colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log Luminance (normalized)", fontsize=12)

    # 加上軸標籤 (X axis / Y axis)
    ax.set_xlabel("X-axis (width)", fontsize=12)
    ax.set_ylabel("Y-axis (height)", fontsize=12)
    ax.set_title("Radiance Map (Log Scale)", fontsize=14)

    # 使排版更緊湊
    plt.tight_layout()

    # Step 5: 儲存結果
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # 關閉 figure 以免佔記憶體
    print(f"Radiance Map 已儲存至：{output_path}")


def main():
    # ========== 1. 選擇影像資料夾（模擬使用者輸入） ==========
    folder_path = select_image_folder()
    images, times = load_images_and_exposures(folder_path)

    if len(images) < 2:
        print("需要至少兩張不同曝光的影像。")
        return

    # ========== 2. 估計相機反應曲線（每個色彩通道） ==========
    # 隨機取樣影像像素來求解 g(z)
    # 示範中隨機取 100 個點
    N = len(images)
    H, W, C = images[0].shape
    sample_size = 100
    np.random.seed(42)

    # 將每張影像攤平成 (H*W, C)，方便取樣
    images_flat = [img.reshape(-1, C) for img in images]

    # 隨機選擇像素位置
    rand_idx = np.random.choice(H * W, sample_size, replace=False)

    # 為每個通道準備取樣陣列
    Zs_channels = []
    for c in range(C):
        Z = np.zeros((sample_size, N), dtype=np.uint8)
        for i in range(sample_size):
            px_idx = rand_idx[i]
            for k in range(N):
                Z[i, k] = images_flat[k][px_idx, c]
        Zs_channels.append(Z)

    # 將曝光時間取對數
    B = np.log(times)

    # 平滑參數 lambda
    lambda_smooth = 100.0
    # 預先計算權重
    w_arr = weight_function(np.arange(256))

    g_channels = []
    for c in range(C):
        Z = Zs_channels[c]
        g, _ = solve_g_and_lnE(Z, B, lambda_smooth, w_arr)
        g_channels.append(g)
    print("相機反應曲線已求得")
    # ========== 3. 繪製並儲存相機反應曲線 ==========
    plot_camera_response_curves(g_channels, folder_path)

    # ========== 4. 組合 HDR 輻射圖 ==========
    print("利用已求得的 g(z) 合成 HDR 影像 (每個通道分別處理)")
    hdr_data = construct_radiance_map(images, times, g_channels)
    print("HDR min:", np.min(hdr_data), "HDR max:", np.max(hdr_data))

    # ========== 5. 輸出 Radiance Map 成 colormap ==========
    output_radiance_map = os.path.join(folder_path, "radiance_map.jpg")
    save_radiance_map_colormap(hdr_data, output_radiance_map)

    start_time = time.time()
    # ========== 6. 將 HDR 影像轉換為 LDR 影像（進行 Tone Mapping） ==========
    print("將 HDR 影像轉換為 LDR 影像（進行 Tone Mapping）")
    ldr_data_drago = tone_mapping_Drago(hdr_data)
    ldr_data_dodge_burn = tone_mapping_dodging_burning(hdr_data)
    ldr_data_reinhard = tone_mapping_Reinhard(hdr_data)

    # ========== 7. 輸出 HDR 檔案 ==========
    output_hdr = os.path.join(folder_path, "output_result.hdr")
    write_hdr_image(hdr_data, output_hdr)
    print(f"HDR 影像已儲存至 {output_hdr}")

    # ========== 8. 儲存並顯示 LDR 影像 ==========
    output_ldr_drago = os.path.join(folder_path, "output_result_ldr_Drago.jpg")
    Image.fromarray(ldr_data_drago).save(output_ldr_drago)
    print(f"LDR 影像已儲存至 {output_ldr_drago}")

    output_ldr_dodge_burn = os.path.join(folder_path, "output_result_ldr_Mantiuk.jpg")
    Image.fromarray(ldr_data_dodge_burn).save(output_ldr_dodge_burn)
    print(f"LDR 影像已儲存至 {output_ldr_dodge_burn}")

    output_ldr_reinhard = os.path.join(folder_path, "output_result_ldr_Reinhard.jpg")
    Image.fromarray(ldr_data_reinhard).save(output_ldr_reinhard)
    print(f"LDR 影像已儲存至 {output_ldr_reinhard}")

    end_time = time.time()
    execution_time = end_time - start_time
    print("Tone Mapping 執行時間：", execution_time, "秒")

if __name__ == "__main__":
    main()
