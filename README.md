# Digital-Visual-Effects---High-Dynamic-Range-Imaging
@NTUCS

---
這個專案包含四個 Python 程式，主要目的是進行高動態範圍（HDR）影像的處理、生成及評估，流程如下：

### 1. **對齊影像 (HDR_MTB.py)**
   - **目的**: 使用 MTB（Median Threshold Bitmap）算法對不同曝光時間拍攝的影像進行對齊。
   - **輸入**: 一個資料夾，裡面包含多張不同曝光的影像。
   - **輸出**: 輸出對齊後的影像，並保存至名為 "aligned" 的資料夾中。
   - **步驟**:
     - 將每張影像轉換為灰階。
     - 根據灰階影像的中位數進行二值化，生成位圖。
     - 使用 MTB 算法對齊影像，並保存最佳位移。
   
   **使用方法**:
   ```bash
   python HDR_MTB.py
   ```
   輸入資料夾路徑後，對齊後的影像將儲存在 "aligned" 資料夾中。

### 2. **生成 HDR 影像與 Tone Mapping (HDR_Debevec.py 和 HDR_Robertson.py)**
   - **目的**: 使用第一步驟輸出的對齊影像生成 HDR 影像，並應用 Tone Mapping 技術（Drago、Mantiuk 和 Reinhard）進行處理。
   - **輸入**: 來自 HDR_MTB.py 的對齊影像。
   - **輸出**: 生成 HDR 影像，並使用不同的 Tone Mapping 算法（Drago、Mantiuk 和 Reinhard）將其轉換為低動態範圍（LDR）影像。
   - **步驟**:
     - 載入影像及其曝光時間。
     - 使用 Debevec 方法估算相機反應曲線。
     - 合成 HDR 影像並應用 Tone Mapping。

   **使用方法**:
   ```bash
   python HDR_Debevec.py
   python HDR_Robertson.py
   ```
   輸入影像資料夾路徑，程式將生成 HDR 影像並應用 Tone Mapping，輸出 LDR 影像。

### 3. **LDR 影像品質判斷 (HDR_color.py)**
   - **目的**: 評估第二步驟輸出的 LDR 影像品質，根據影像的亮度和飽和度指標進行判斷。
   - **輸入**: 來自 HDR_Debevec.py 或 HDR_Robertson.py 的 LDR 影像。
   - **輸出**: 顯示影像的亮度直方圖及原始影像，並計算並輸出平均亮度、亮度標準差、平均飽和度和飽和度標準差等指標。
   - **步驟**:
     - 計算影像的亮度直方圖。
     - 計算影像的亮度和飽和度指標。

   **使用方法**:
   ```bash
   python HDR_color.py
   ```
   程式將讓使用者選擇一張影像，並顯示亮度直方圖及原始影像，計算並顯示影像品質指標。

這四個程式依照順序運行，第一步驟用於影像對齊，第二步驟進行 HDR 影像生成和 Tone Mapping，第三步驟用來評估生成的 LDR 影像品質。

### 以下將詳細說明這四個程式執行方式。

---

# HDR_MTB.py

## 說明
這個 Python 程式用於對一組影像進行對齊與處理，主要步驟包括將影像轉換為灰階、應用中位數二值化來生成位圖圖像，並使用 MTB (Median Threshold Bitmap) 算法來對齊影像。程式會處理資料夾中的所有影像，並將結果保存到指定的資料夾中，包括灰階影像、位圖影像、對齊後的影像以及每層金字塔的最佳位移圖表。

## 需求
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

您可以使用以下命令安裝所需的庫：
```
pip install opencv-python numpy matplotlib
```

## 使用方法

### 1. 準備影像資料夾
將您要處理的影像放置在一個資料夾中，支持的格式包括 `.jpg`、`.png`、`.jpeg`、`.bmp`、`.tiff`。

### 2. 運行程式
- 輸入影像資料夾的路徑。
- 程式會自動處理資料夾中的影像，並進行以下操作：
  1. 將每幅影像轉換成灰階圖並保存。
  2. 根據中位數進行二值化處理，生成位圖並保存。
  3. 使用 MTB 算法對影像進行對齊，並將對齊後的影像保存到新資料夾。
  4. 繪製每層金字塔的最佳位移（水平與垂直位移）的圖表，並保存為圖片。

### 3. 輸出結果
- 對齊後的影像會保存在新的資料夾 `aligned` 中。
- 每層金字塔的最佳位移會生成一個圖表，並保存為 `best_offset_chart.jpg`。

## 函數說明

### `convert_to_grayscale(image)`
將 BGR 影像轉換為灰階影像。

### `median_threshold_bitmap(gray_img)`
根據灰階影像的中位數進行二值化，生成位圖。

### `align_mtb_with_offsets(ref_img, img, levels=5)`
使用 MTB 算法對兩幅影像進行對齊，並記錄每層金字塔的最佳位移。
- `ref_img`：參考影像 (BGR 格式)
- `img`：待對齊影像 (BGR 格式)
- `levels`：金字塔層數（預設 5 層）

### `main()`
主程式，負責處理影像資料夾中的所有影像，並進行灰階轉換、二值化、影像對齊以及繪製最佳位移圖表。

## 注意事項
- 請確認資料夾內有有效的影像文件。
- 影像對齊會根據金字塔層進行多層次的位移搜索，可能需要較長的處理時間，具體時間取決於影像數量和解析度。

---

# HDR_Debevec.py

本專案示範如何利用多張不同曝光時間的影像，使用 Debevec & Malik 方法估計相機反應曲線並合成 HDR 影像，最後透過各式 Tone Mapping 方法輸出 LDR 影像。本程式包含：

1. **讀取資料夾中的多張影像及其曝光時間**  
2. **估計相機反應函數 (Camera Response Curve)**  
3. **合成 HDR 輻射圖**  
4. **輸出 Radiance Map 圖片 (可視化)**  
5. **套用 Tone Mapping 產生 LDR 圖片**  
6. **輸出最終 HDR (Radiance RGBE) 檔案**

## 1. 環境需求 (Dependencies)

請先安裝下列套件與 Python 版本：
- Python 3.7 以上版本
- [NumPy](https://numpy.org/)  
- [OpenCV (cv2)](https://opencv.org/)  
- [Pillow (PIL)](https://pillow.readthedocs.io/)  
- [Matplotlib](https://matplotlib.org/)  
- [tqdm](https://github.com/tqdm/tqdm) 用於顯示進度條  
- [SciPy](https://scipy.org/) (主要使用 `scipy.sparse` 與 `scipy.sparse.linalg.lsqr`)

若使用 pip 安裝，可執行：
```bash
pip install numpy opencv-python pillow matplotlib tqdm scipy
```

## 2. 程式架構

建議將下列程式碼儲存為 `HDR_Debevec.py`（或其他檔名），並直接執行：

```python
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from tqdm import tqdm
import time

# ----------------------
# 以下為主要程式內容
# ----------------------

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

def weight_function(z):
    """
    簡單的像素強度權重函式 (0~255)。
    這裡採用帽狀函式：若 z <= 128，則 w(z) = z；否則 w(z) = 255 - z。
    """
    return np.where(z <= 128, z, 255 - z)

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

    idx = 0
    # -- (資料配適方程) --
    for i in range(P):
        for j in range(N):
            z_ij = Z[i, j]
            w_ij = w[z_ij]
            A[idx, z_ij] = w_ij
            A[idx, n + i] = -w_ij
            b[idx] = w_ij * B[j]
            idx += 1

    # -- (平滑性約束) --
    for z in range(1, 255):
        A[idx, z - 1] = l
        A[idx, z] = -2 * l
        A[idx, z + 1] = l
        idx += 1

    # -- (固定 g(128) = 0) --
    A[idx, 128] = 1.0
    b[idx] = 0.0
    idx += 1

    # -- (求解線性方程組) --
    A_csr = A.tocsr()
    x = lsqr(A_csr, b)[0]  # x = [g(0..255), lnE(0..P-1)]

    g = x[0:n]
    lE = x[n:]
    return g, lE

def construct_radiance_map(images, times, g):
    """
    根據估計好的 g(z)，重建每個像素的輻射圖 (以對數域表示)。
    計算方式為：
        lnE(i,j) = ( Σ_n w(Z(i,j,n)) * [ g(Z(i,j,n)) - ln(t_n) ] ) /
                   ( Σ_n w(Z(i,j,n)) )
    最後取指數以得到 E(i,j)。

    參數：
        images: 彩色影像清單，形狀 (H, W, 3)。
        times: 曝光時間陣列，形狀 (N,)。
        g: 相機反應函數 (每個通道都有一個 g(z))。

    傳回值：
        hdr: 重建的 HDR 輻射圖，形狀 (H, W, 3)。
    """
    N = len(images)
    H, W, C = images[0].shape
    ln_t = np.log(times)

    hdr = np.zeros((H, W, C), dtype=np.float32)
    w_arr = weight_function(np.arange(256))

    for c in tqdm(range(C), desc="HDR Reconstruction"):
        # (N,H,W) 每一層都是第 n 張影像的第 c 通道
        vals = np.array([img[:, :, c] for img in images], dtype=np.uint8)
        wghts = w_arr[vals]

        top = np.zeros((H, W), dtype=np.float32)
        bot = np.zeros((H, W), dtype=np.float32)

        for n in range(N):
            top += wghts[n] * (g[c][vals[n]] - ln_t[n])
            bot += wghts[n]

        bot = np.maximum(bot, 1e-8)
        channel_map = top / bot
        hdr[..., c] = channel_map

    hdr = np.exp(hdr)
    return hdr

def write_hdr_image(hdr_data, filename):
    """
    將 Radiance RGBE 格式的 HDR 檔案 (.hdr) 寫出。
    這只是簡單示範，可再根據需求使用更完整的工具庫 (如 imageio)。
    """
    H, W, C = hdr_data.shape
    assert C == 3, "HDR 資料必須為 3 通道。"

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

def tone_mapping_Drago(hdr_data):
    """
    使用 OpenCV 的 Drago Tone Mapping。
    """
    tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=8.0)
    hdr_normalized = hdr_data.astype(np.float32) / 255.0
    ldr_data = tonemap.process(hdr_normalized)
    ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8)
    return ldr_data

def tone_mapping_dodging_burning(hdr_data):
    """
    使用 OpenCV 的 Mantiuk (近似 Dodging & Burning)。
    """
    tonemap_mantiuk = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.0)
    hdr_normalized = hdr_data.astype(np.float32) / 255.0
    ldr_data = tonemap_mantiuk.process(hdr_normalized)
    ldr_data = np.nan_to_num(ldr_data, nan=0.0, posinf=1.0, neginf=0.0)
    ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8)
    return ldr_data

def tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6):
    """
    使用 Reinhard 全局 Tone Mapping。
    """
    L = 0.2126 * hdr_data[..., 0] + 0.7152 * hdr_data[..., 1] + 0.0722 * hdr_data[..., 2]
    L_log = np.log(delta + L)
    L_avg = np.exp(np.mean(L_log))
    L_scaled = (key / L_avg) * L
    L_mapped = L_scaled / (1 + L_scaled)
    S = L_mapped / (L + delta)
    S = S[..., np.newaxis]
    ldr = hdr_data * S
    ldr = np.clip(ldr, 0, 1)
    ldr_data = (ldr * 255).astype(np.uint8)
    return ldr_data

def plot_camera_response_curves(g_channels, folder_path):
    """
    繪製並同時顯示 4 張圖表：
      1) Red channel
      2) Green channel
      3) Blue channel
      4) RGB 三色疊加
    """
    x_vals = np.arange(256)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Camera Response Curves', fontsize=16)

    axs[0, 0].plot(x_vals, g_channels[0], 'r')
    axs[0, 0].set_title("Red")
    axs[0, 0].set_xlabel('log exposure X')
    axs[0, 0].set_ylabel('pixel value Z')
    axs[0, 0].grid(True)

    axs[0, 1].plot(x_vals, g_channels[1], 'g')
    axs[0, 1].set_title("Green")
    axs[0, 1].set_xlabel('log exposure X')
    axs[0, 1].set_ylabel('pixel value Z')
    axs[0, 1].grid(True)

    axs[1, 0].plot(x_vals, g_channels[2], 'b')
    axs[1, 0].set_title("Blue")
    axs[1, 0].set_xlabel('log exposure X')
    axs[1, 0].set_ylabel('pixel value Z')
    axs[1, 0].grid(True)

    colors = ['r', 'g', 'b']
    labels = ['red', 'green', 'blue']
    for c in range(3):
        axs[1, 1].plot(x_vals, g_channels[c], colors[c], label=labels[c])
    axs[1, 1].set_title("RGB Combined")
    axs[1, 1].set_xlabel('log exposure X')
    axs[1, 1].set_ylabel('pixel value Z')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    curve_out = os.path.join(folder_path, "camera_response_curve.png")
    plt.savefig(curve_out)
    print(f"相機反應曲線已儲存：{curve_out}")

def save_radiance_map_colormap(hdr_data, output_path):
    """
    以 log scale 顯示 HDR 圖的 luminance 並輸出。
    """
    L = 0.2126 * hdr_data[..., 0] + \
        0.7152 * hdr_data[..., 1] + \
        0.0722 * hdr_data[..., 2]
    L_log = np.log1p(L)
    L_min, L_max = L_log.min(), L_log.max()
    L_log_norm = (L_log - L_min) / (L_max - L_min + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(L_log_norm, cmap='jet', origin='upper')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log Luminance (normalized)", fontsize=12)
    ax.set_xlabel("X-axis (width)", fontsize=12)
    ax.set_ylabel("Y-axis (height)", fontsize=12)
    ax.set_title("Radiance Map (Log Scale)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Radiance Map 已儲存至：{output_path}")

def main():
    start_time = time.time()
    folder_path = select_image_folder()
    images, times = load_images_and_exposures(folder_path)

    if len(images) < 2:
        print("需要至少兩張不同曝光的影像。")
        return

    # -- 估計相機反應曲線 --
    N = len(images)
    H, W, C = images[0].shape
    sample_size = 100
    np.random.seed(42)

    images_flat = [img.reshape(-1, C) for img in images]
    rand_idx = np.random.choice(H * W, sample_size, replace=False)

    Zs_channels = []
    for c in range(C):
        Z = np.zeros((sample_size, N), dtype=np.uint8)
        for i in range(sample_size):
            px_idx = rand_idx[i]
            for k in range(N):
                Z[i, k] = images_flat[k][px_idx, c]
        Zs_channels.append(Z)

    B = np.log(times)
    lambda_smooth = 100.0
    w_arr = weight_function(np.arange(256))

    g_channels = []
    for c in range(C):
        Z = Zs_channels[c]
        g, _ = solve_g_and_lnE(Z, B, lambda_smooth, w_arr)
        g_channels.append(g)
    print("相機反應曲線已求得")

    plot_camera_response_curves(g_channels, folder_path)

    # -- 重建 HDR --
    print("利用已求得的 g(z) 合成 HDR 影像 (每個通道分別處理)")
    hdr_data = construct_radiance_map(images, times, g_channels)
    print("HDR min:", np.min(hdr_data), "HDR max:", np.max(hdr_data))

    output_radiance_map = os.path.join(folder_path, "radiance_map.jpg")
    save_radiance_map_colormap(hdr_data, output_radiance_map)

    start_time = time.time()

    # -- Tone Mapping --
    print("將 HDR 影像轉換為 LDR 影像（進行 Tone Mapping）")
    ldr_data_drago = tone_mapping_Drago(hdr_data)
    ldr_data_dodge_burn = tone_mapping_dodging_burning(hdr_data)
    ldr_data_reinhard = tone_mapping_Reinhard(hdr_data)

    output_hdr = os.path.join(folder_path, "output_result.hdr")
    write_hdr_image(hdr_data, output_hdr)
    print(f"HDR 影像已儲存至 {output_hdr}")

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
```

## 3. 執行方式

1. **確認安裝環境**  
   - Python 3.7+  
   - 以及上述套件：numpy, opencv-python, pillow, matplotlib, tqdm, scipy  

2. **準備輸入影像**  
   - 創建一個資料夾，放入多張對齊過(相同構圖)的影像檔。  
   - 檔名結尾最好包含曝光時間，例如：`img_0.005.jpg`, `img_0.025.jpg`，程式會嘗試自動解析最後一段`_0.025`作為曝光時間。

3. **執行程式**  
   - 在終端機 (Terminal / CMD) 中切換到程式所在位置，執行：
     ```bash
     python hdr_pipeline.py
     ```
   - 執行後，程式會要求輸入影像資料夾路徑。請輸入放置影像的資料夾絕對路徑或相對路徑。  
   - 例如：`C:\Users\MyName\Desktop\HDRImages`（Windows）或 `./HDRImages`（Mac / Linux）。

4. **程式流程**  
   - 讀取所有影像與其曝光時間  
   - 估計相機反應曲線 (需要抽樣像素)  
   - 組合 HDR  
   - 輸出一張 `radiance_map.jpg` (log-luminance 可視化)  
   - 進行 Tone Mapping，輸出多種 LDR 版本 (Drago, Mantiuk, Reinhard)  
   - 寫出最終的 `.hdr` 檔案與 LDR 結果 (jpg)

5. **輸出結果**  
   - `camera_response_curve.png`：相機反應曲線 (RGB 三通道)  
   - `radiance_map.jpg`：可視化的 HDR Luminance (log scale)  
   - `output_result.hdr`：HDR (Radiance) 格式檔案  
   - `output_result_ldr_Drago.jpg`：Drago Tone Mapping 結果  
   - `output_result_ldr_Mantiuk.jpg`：Mantiuk Tone Mapping 結果  
   - `output_result_ldr_Reinhard.jpg`：Reinhard Tone Mapping 結果  

## 4. 注意事項

- **相機反應函數求解**  
  本程式實作為傳統 Debevec & Malik 方法的簡化版（需平滑參數 `lambda_smooth`）。若影像數量較少或曝光間隔不明顯，曲線估計可能不準。  
- **影像對齊**  
  程式假設所有輸入影像已對齊 (無相對位移)。如果實際拍攝時有移動，需先進行對齊 (Alignment) 或使用帶對齊功能的 HDR 工具。  
- **輸入影像之動態範圍**  
  本程式以 `8-bit` 影像作為輸入。若有條件，可使用 RAW 檔或更高位元深度（如 16-bit TIFF）取得更佳品質。  
- **對應 Tone Mapping 參數**  
  Drago, Mantiuk, Reinhard 都有不同參數可調整，例如 `gamma`, `bias`, `scale` 等，可根據實際顯示需求做調整。  

## 5. 版權或授權

本程式屬示範性質，可自由修改、學術或個人研究使用。如需商業應用，請先確定所使用的第三方函式庫條款（OpenCV、Pillow、NumPy 等）的授權規範。  

## 6. 常見問題 (FAQ)

1. **Q**：程式執行後跳出 `ModuleNotFoundError: No module named 'cv2'`？  
   **A**：請先確定已安裝 `opencv-python`，或嘗試 `pip install opencv-python`。

2. **Q**：輸出的 `.hdr` 檔案無法在某些軟體中正常顯示？  
   **A**：目前採用的是基本 Radiance RGBE 格式寫出，部分軟體需要能讀 `.hdr`(Radiance)格式。如需 EXR 等格式，可改用 `imageio.imwrite("output.exr", hdr_data, format='EXR')`。

3. **Q**：曝光時間若從檔名無法解析，會怎樣？  
   **A**：程式會預設該曝光時間為1.0；因此若有多張影像都沒成功讀到正確曝光時間，對於合成結果的意義將大幅降低。請務必確認檔名能解析出正確的 `_數值`。

4. **Q**：程式執行後跳出 ` RuntimeWarning: invalid value encountered in cast ldr_data = np.clip(ldr_data * 255, 0, 255).astype(np.uint8) `？
   **A**：別擔心，不影響執行。

---

# HDR_Robertson.py

## 說明
此 Python 程式用於合成高動態範圍影像（HDR）並進行色調映射。它從一組不同曝光時間的影像中，利用估計理論方法合成 HDR，並使用多種色調映射方法（如 Drago、Mantiuk 和 Reinhard）來處理 HDR 影像，最後生成視覺化的輻射圖（Radiance Map）。此外，程式也會輸出相機反應曲線並進行平滑處理。

## 需求
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- SciPy (`scipy`)
- Matplotlib

您可以使用以下命令安裝所需的庫：
```
pip install opencv-python numpy pillow scipy matplotlib
```

## 使用方法

### 1. 準備影像資料夾
將一組不同曝光時間的影像（.jpg、.png、.tiff等格式）放置在一個資料夾中，檔名格式應為 `img_<曝光時間>.jpg`，如 `img_0.01.jpg`。

### 2. 運行程式
- 輸入影像資料夾的路徑，程式會從該資料夾讀取影像與對應的曝光時間。
- 程式將進行以下操作：
  1. 估測相機反應曲線（Camera Response Function, CRF），並繪製曲線。
  2. 使用「估計理論」方法合成 HDR 影像。
  3. 輸出輻射圖（Radiance Map）的可視化結果。
  4. 進行色調映射，並使用 Drago、Mantiuk 和 Reinhard 方法生成 LDR 影像。
  5. 輸出 HDR 與 LDR 結果，並儲存至資料夾。

### 3. 輸出結果
- **相機反應曲線圖**：顯示紅、綠、藍通道的相機反應曲線，以及三色疊加的曲線，儲存為 `camera_response_curve_smoothed.png`。
- **HDR 檔案**：將合成的 HDR 影像儲存為 `.hdr` 格式，儲存路徑為 `output_estimation_result.hdr`。
- **LDR 影像**：分別使用不同的色調映射方法生成 LDR 影像，並儲存為 `.jpg` 格式：
  - Drago 方法：`output_estimation_ldr_Drago.jpg`
  - Mantiuk 方法：`output_estimation_ldr_Mantiuk.jpg`
  - Reinhard 方法：`output_estimation_ldr_Reinhard.jpg`
- **輻射圖可視化**：以 Log Luminance 顯示輻射圖，並儲存為 `radiance_map_estimation.jpg`。

## 函數說明

### `select_image_folder()`
模擬選擇影像資料夾的函式。實際情況中可用檔案對話框或命令行參數選擇資料夾。

### `load_images_and_exposures(folder_path)`
從指定資料夾讀取影像及對應的曝光時間。影像檔名格式需包含曝光時間，程式會解析檔名並將曝光時間提取出來。

### `write_hdr_image(hdr_data, filename)`
將合成的 HDR 影像資料寫入 `.hdr` 檔案，使用 Radiance RGBE 格式。

### `weight_function_estimation(I, eps=0.02, sat=0.98)`
為每個像素計算權重，過暗或過亮的像素將被賦予權重 0，其他像素賦予權重 1。

### `construct_hdr_estimation_theoretic(images, times, eps=0.02, sat=0.98)`
使用簡化版的估計理論方法合成 HDR 影像。

### `smooth_curve(curve, window_length=51, polyorder=3)`
對相機反應曲線進行平滑處理，使用 Savitzky-Golay 濾波器。

### `plot_camera_response_curves(g_channels, folder_path)`
繪製並儲存相機反應曲線圖，包括紅、綠、藍通道的曲線，以及三色疊加的曲線。

### `tone_mapping_Drago(hdr_data)`
使用 OpenCV 的 Drago 方法進行色調映射。

### `tone_mapping_dodging_burning(hdr_data)`
使用 OpenCV 的 Mantiuk 方法進行色調映射，適用於模擬 Dodging and Burning 效果。

### `tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6)`
使用 Reinhard 方法進行全局色調映射。

### `save_radiance_map_colormap(hdr_data, output_path)`
將輻射圖（Log Luminance）以彩色地圖形式儲存，幫助可視化 HDR 影像。

## 注意事項
- 請確保影像檔案符合命名規範，並且有至少兩張影像具有不同的曝光時間。
- 影像應該具有相同解析度，且已對齊。
- 色調映射處理可能需要較長的時間，具體時間取決於影像數量與解析度。

## 執行範例
```bash
$ python hdr_tonemapping.py
請輸入影像資料夾的路徑：/path/to/images
開始估測相機反應曲線 (Robertson) ...
開始進行 Estimation-theoretic HDR 合成...
HDR 合成完成。
HDR 內容範圍: min=0.0001, max=0.9753
進行 Tone Mapping...
LDR(Drago) 輸出至: /path/to/images/output_estimation_ldr_Drago.jpg
LDR(Mantiuk) 輸出至: /path/to/images/output_estimation_ldr_Mantiuk.jpg
LDR(Reinhard) 輸出至: /path/to/images/output_estimation_ldr_Reinhard.jpg
```

---

# HDR_color.py

## 說明
這個 Python 程式用於計算一張圖片的亮度與飽和度指標，並將結果顯示和保存為圖片檔案。程式可以選擇一張圖片，計算該圖片的亮度直方圖、平均亮度、亮度標準差、色彩飽和度的平均值與標準差，並將結果以圖表的形式保存。

## 需求
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Tkinter（用於檔案選擇對話框）

您可以使用以下命令安裝所需的庫：
```
pip install opencv-python numpy matplotlib
```

## 功能概述
1. **讀取圖片**：選擇一張圖片進行處理。
2. **計算亮度指標**：計算並顯示該圖片的亮度直方圖，並計算平均亮度與亮度的標準差。
3. **計算飽和度指標**：計算並顯示圖片的色彩飽和度平均值與標準差。
4. **圖像保存**：保存亮度直方圖和原圖。
5. **結果顯示**：在終端顯示平均亮度、亮度標準差、飽和度平均值和飽和度標準差。

## 使用方法

1. **運行程式**：
   - 執行程式後，程式會彈出檔案選擇對話框，您可以選擇要處理的圖片。
   - 程式會計算該圖片的亮度直方圖、亮度指標、飽和度指標，並保存結果。

2. **輸出結果**：
   - **亮度直方圖**：程式會保存名為 `brightness_histogram.png` 的亮度直方圖圖像。
   - **原圖**：程式會保存名為 `original_image.png` 的原圖圖像。
   - **終端輸出**：程式會顯示圖片的平均亮度、亮度標準差、飽和度平均值、飽和度標準差。

## 函數說明

### `calculate_image_metrics(image_path)`
該函數用於計算並返回以下圖像指標：
- 平均亮度 (`mean_brightness`)
- 亮度標準差 (`std_brightness`)
- 平均飽和度 (`mean_saturation`)
- 飽和度標準差 (`std_saturation`)

它還會生成並保存：
- 亮度直方圖（`brightness_histogram.png`）
- 顯示原圖並保存（`original_image.png`）

### `open_image_file()`
使用 `Tkinter` 彈出檔案選擇視窗，讓用戶選擇要處理的圖片檔案。

## 範例

```bash
$ python image_metrics.py
```

當運行程式後，會彈出一個文件選擇視窗，選擇您要分析的圖片檔案後，程式會顯示結果並生成相關圖像檔案。終端輸出的範例如下：

```bash
Average Brightness: 112.45
Brightness Standard Deviation: 42.17
Average Saturation: 0.45
Saturation Standard Deviation: 0.12
```

生成的圖片檔案會保存在程式運行的當前目錄中。
