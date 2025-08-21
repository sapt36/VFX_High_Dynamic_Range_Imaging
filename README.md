# Digital-Visual-Effects---High-Dynamic-Range-Imaging
@NTU CSIE

Four Python programs for HDR image processing:

## Requirements
```
pip install opencv-python numpy matplotlib pillow scipy tqdm
```

### 1. **Image Alignment (HDR_MTB.py)**
   Aligns images with different exposures using MTB algorithm.
   ![image](https://github.com/user-attachments/assets/aa48cc92-6a9d-46a1-9430-cf167756976e)
   ```bash
   python HDR_MTB.py
   ```

### 2. **HDR Generation (HDR_Debevec.py / HDR_Robertson.py)**
   Creates HDR images and applies tone mapping (Drago, Mantiuk, Reinhard).
   ![image](https://github.com/user-attachments/assets/b0e5b989-137e-4e0f-953e-657a3e9fc50a)
   ```bash
   python HDR_Debevec.py
   python HDR_Robertson.py
   ```

### 3. **Quality Assessment (HDR_color.py)**
   Evaluates LDR image quality using brightness and saturation metrics.
   ```bash
   python HDR_color.py
   ```

### Result Images
![image](https://github.com/user-attachments/assets/e941c7e9-06f3-48d8-a65c-416624177b9d)

---

# HDR_MTB.py

Image alignment using MTB (Median Threshold Bitmap) algorithm.

Enter folder path. Outputs aligned images to `aligned/` folder and displacement chart as `best_offset_chart.jpg`.

## Key Functions
- `convert_to_grayscale()` - Convert BGR to grayscale
- `median_threshold_bitmap()` - Generate bitmap using median threshold  
- `align_mtb_with_offsets()` - Align images using MTB algorithm

---

# HDR_Debevec.py

HDR reconstruction using Debevec & Malik method with tone mapping.

Enter image folder path. Outputs:
- HDR image: `output_result.hdr`
- LDR images: `output_result_ldr_Drago.jpg`, `output_result_ldr_Mantiuk.jpg`, `output_result_ldr_Reinhard.jpg`
- Camera response curves: `camera_response_curve_smoothed.png`
- Radiance map: `radiance_map.jpg`

## Key Functions
- `load_images_and_exposures()` - Load images and parse exposure times
- `solve_g_and_lnE()` - Estimate camera response curves using Debevec method
- `construct_radiance_map()` - Reconstruct HDR radiance map
- `tone_mapping_Drago/Reinhard/dodging_burning()` - Apply tone mapping

---

# HDR_Robertson.py

HDR synthesis using estimation-theoretic methods with tone mapping.

Enter image folder path. Outputs:
- HDR image: `output_estimation_result.hdr`
- LDR images: `output_estimation_ldr_Drago.jpg`, `output_estimation_ldr_Mantiuk.jpg`, `output_estimation_ldr_Reinhard.jpg`
- Camera response curves: `camera_response_curve_smoothed.png`  
- Radiance map: `radiance_map_estimation.jpg`

## Key Functions
- `load_images_and_exposures()` - Load images and parse exposure times
- `construct_hdr_estimation_theoretic()` - Synthesize HDR using estimation method
- `weight_function_estimation()` - Calculate pixel weights
- `tone_mapping_Drago/Reinhard/dodging_burning()` - Apply tone mapping

---

# HDR_color.py

Image quality assessment using brightness and saturation metrics.

Select an image via file dialog. Outputs:
- Brightness histogram: `brightness_histogram.png`
- Original image: `original_image.png`
- Terminal metrics: average brightness, brightness std, average saturation, saturation std

## Key Functions
- `calculate_image_metrics()` - Calculate brightness and saturation metrics
- `open_image_file()` - File selection dialog using Tkinter
