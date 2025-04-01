# Digital-Visual-Effects---High-Dynamic-Range-Imaging
@NTUCS

---
這個專案包含四個 Python 程式，主要目的是進行高動態範圍（HDR）影像的處理、生成及評估，流程如下：

### 1. **對齊影像 (HDR_MTB.py)**
   - **目的**: 使用 MTB（Median Threshold Bitmap）算法對不同曝光時間拍攝的影像進行對齊。
   - **輸入**: 一個資料夾，裡面包含多張不同曝光的影像。
   ![image](https://github.com/user-attachments/assets/aa48cc92-6a9d-46a1-9430-cf167756976e)
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
   ![image](https://github.com/user-attachments/assets/b0e5b989-137e-4e0f-953e-657a3e9fc50a)
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

### 結果圖
![image](https://github.com/user-attachments/assets/e941c7e9-06f3-48d8-a65c-416624177b9d)


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

## 說明
這個 Python 程式實現了高動態範圍影像（HDR）的重建與色調映射，能夠從一組具有不同曝光時間的影像中合成HDR影像，並進行三種不同的色調映射（Drago、Mantiuk、Reinhard）。程式使用 Debevec & Malik 方法來估算相機反應函數，並根據此反應函數重建每個像素的輻射值，最後進行色調映射處理。

## 需求
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- PIL (Pillow)
- SciPy
- tqdm

安裝必要的庫：
```
pip install opencv-python numpy matplotlib pillow scipy tqdm
```

## 功能概述
1. **讀取影像與曝光時間**：從資料夾讀取一組影像並解析出每張影像的曝光時間。
2. **計算相機反應曲線**：使用 Debevec & Malik 方法估算相機反應函數 (g(z))，並根據曝光時間進行 HDR 合成。
3. **HDR 輻射圖重建**：使用估算出的相機反應曲線重建每個像素的輻射圖（Log 形式）。
4. **色調映射**：進行 HDR 到 LDR（低動態範圍）的色調映射，支持 Drago、Mantiuk 和 Reinhard 三種方法。
5. **輸出結果**：輸出 HDR 影像（.hdr 格式）和色調映射後的 LDR 影像（.jpg 格式）。

## 使用方法

1. **運行程式**：
   - 運行程式後，程式會要求您輸入影像資料夾的路徑。
   - 程式將自動讀取資料夾中的影像，並根據影像的檔名解析曝光時間。

2. **輸出結果**：
   - **相機反應曲線**：程式會計算每個色彩通道的相機反應曲線並繪製，保存為 `camera_response_curve_smoothed.png`。
   - **HDR 影像**：將計算出的 HDR 影像保存為 `.hdr` 格式，儲存路徑為 `output_result.hdr`。
   - **LDR 影像**：將經過色調映射處理的 LDR 影像保存為 `.jpg` 格式，包括：
     - 使用 Drago 方法的結果：`output_result_ldr_Drago.jpg`
     - 使用 Mantiuk 方法的結果：`output_result_ldr_Mantiuk.jpg`
     - 使用 Reinhard 方法的結果：`output_result_ldr_Reinhard.jpg`
   - **輻射圖可視化**：生成並儲存輻射圖的可視化，保存為 `radiance_map.jpg`。

3. **終端輸出**：
   - 顯示每個步驟的處理狀況，包括讀取影像數量、相機反應曲線的估算、HDR 影像的生成、色調映射結果等。

## 函數說明

### `select_image_folder()`
讓使用者輸入影像資料夾的路徑。

### `load_images_and_exposures(folder_path)`
從指定資料夾讀取影像並解析出曝光時間。影像檔名應包含曝光時間，程式會自動解析。

### `weight_function(z, lower_limit=20, upper_limit=235)`
根據像素強度對每個像素進行加權，過暗或過亮的像素會有較低的權重。

### `solve_g_and_lnE(Z, B, l, w)`
使用 Debevec & Malik 方法來解算相機反應函數 `g(z)` 及每個像素的對數輻射值 `ln(E)`。

### `construct_radiance_map(images, times, g)`
根據估算出的相機反應函數 `g(z)`，重建每個像素的輻射圖。

### `write_hdr_image(hdr_data, filename)`
將重建的 HDR 影像儲存為 `.hdr` 格式，使用 Radiance RGBE 格式。

### `tone_mapping_Drago(hdr_data)`
使用 OpenCV 的 Drago 方法對 HDR 影像進行色調映射。

### `tone_mapping_dodging_burning(hdr_data)`
使用 OpenCV 的 Mantiuk 方法進行局部色調映射，模擬 Dodging 和 Burning 效果。

### `tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6)`
使用 Reinhard 方法對 HDR 影像進行全局色調映射。

### `smooth_curve(curve, window_length=51, polyorder=3)`
使用 Savitzky-Golay 濾波器對相機反應曲線進行平滑處理。

### `plot_camera_response_curves(g_channels, folder_path)`
繪製並儲存相機反應曲線，包括紅、綠、藍三個通道及其疊加。

### `save_radiance_map_colormap(hdr_data, output_path)`
將輻射圖（Log Luminance）顯示為彩色地圖並儲存，幫助可視化 HDR 影像。

## 範例

```bash
$ python hdr_tonemapping.py
```

程式運行後，會提示您輸入影像資料夾路徑，並開始處理影像。處理結果會儲存在同一資料夾中。

## 注意事項
- 確保影像檔名包含曝光時間（例如 `img_0.025.jpg`），並且所有影像的解析度相同。
- 程式支援的影像格式包括 `.jpg`、`.png`、`.jpeg`、`.bmp`、`.tiff`。
- 若影像數量少於 2 張，程式將無法繼續運行。

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
