# Digital-Visual-Effects---High-Dynamic-Range-Imaging
@NTUCS

---
This project contains four Python programs designed for processing, generating, and evaluating High Dynamic Range (HDR) images. The workflow is as follows:

### 1. **Image Alignment (HDR_MTB.py)**
   - **Purpose**: Align images captured with different exposure times using the MTB (Median Threshold Bitmap) algorithm.
   - **Input**: A folder containing multiple images with different exposures.
   ![image](https://github.com/user-attachments/assets/aa48cc92-6a9d-46a1-9430-cf167756976e)
   - **Output**: Outputs aligned images and saves them to a folder named "aligned".
   - **Steps**:
     - Convert each image to grayscale.
     - Apply binary thresholding based on the median of the grayscale image to generate bitmaps.
     - Align images using the MTB algorithm and save the optimal displacement.
   
   **Usage**:
   ```bash
   python HDR_MTB.py
   ```
   After entering the folder path, aligned images will be saved in the "aligned" folder.

### 2. **HDR Image Generation and Tone Mapping (HDR_Debevec.py and HDR_Robertson.py)**
   - **Purpose**: Generate HDR images using aligned images from step 1 and apply tone mapping techniques (Drago, Mantiuk, and Reinhard) for processing.
   - **Input**: Aligned images from HDR_MTB.py.
   - **Output**: Generate HDR images and convert them to Low Dynamic Range (LDR) images using different tone mapping algorithms (Drago, Mantiuk, and Reinhard).
   ![image](https://github.com/user-attachments/assets/b0e5b989-137e-4e0f-953e-657a3e9fc50a)
   - **Steps**:
     - Load images and their exposure times.
     - Estimate camera response curves using the Debevec method.
     - Synthesize HDR images and apply tone mapping.

   **Usage**:
   ```bash
   python HDR_Debevec.py
   python HDR_Robertson.py
   ```
   Enter the image folder path, and the program will generate HDR images, apply tone mapping, and output LDR images.

### 3. **LDR Image Quality Assessment (HDR_color.py)**
   - **Purpose**: Evaluate the quality of LDR images output from step 2 based on brightness and saturation metrics.
   - **Input**: LDR images from HDR_Debevec.py or HDR_Robertson.py.
   - **Output**: Display the brightness histogram and original image, calculate and output metrics including average brightness, brightness standard deviation, average saturation, and saturation standard deviation.
   - **Steps**:
     - Calculate the brightness histogram of the image.
     - Calculate brightness and saturation metrics of the image.

   **Usage**:
   ```bash
   python HDR_color.py
   ```
   The program will allow users to select an image and display the brightness histogram and original image, calculate and display image quality metrics.

These four programs are executed in sequence: the first step is for image alignment, the second step performs HDR image generation and tone mapping, and the third step evaluates the quality of the generated LDR images.

### Result Images
![image](https://github.com/user-attachments/assets/e941c7e9-06f3-48d8-a65c-416624177b9d)


### The following sections provide detailed explanations of how to execute these four programs.

---

# HDR_MTB.py

## Description
This Python program is used for aligning and processing a set of images. The main steps include converting images to grayscale, applying median threshold binarization to generate bitmap images, and using the MTB (Median Threshold Bitmap) algorithm to align images. The program processes all images in a folder and saves the results to specified folders, including grayscale images, bitmap images, aligned images, and optimal displacement charts for each pyramid level.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

You can install the required libraries using the following command:
```
pip install opencv-python numpy matplotlib
```

## Usage

### 1. Prepare Image Folder
Place the images you want to process in a folder. Supported formats include `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`.

### 2. Run the Program
- Enter the path to the image folder.
- The program will automatically process the images in the folder and perform the following operations:
  1. Convert each image to grayscale and save.
  2. Apply binary thresholding based on the median to generate bitmaps and save.
  3. Align images using the MTB algorithm and save aligned images to a new folder.
  4. Plot the optimal displacement (horizontal and vertical displacement) for each pyramid level and save as an image.

### 3. Output Results
- Aligned images will be saved in a new folder named `aligned`.
- The optimal displacement for each pyramid level will generate a chart and be saved as `best_offset_chart.jpg`.

## Function Descriptions

### `convert_to_grayscale(image)`
Converts a BGR image to grayscale.

### `median_threshold_bitmap(gray_img)`
Applies binary thresholding based on the median of the grayscale image to generate a bitmap.

### `align_mtb_with_offsets(ref_img, img, levels=5)`
Aligns two images using the MTB algorithm and records the optimal displacement for each pyramid level.
- `ref_img`: Reference image (BGR format)
- `img`: Image to be aligned (BGR format)
- `levels`: Number of pyramid levels (default 5 levels)

### `main()`
Main program responsible for processing all images in the image folder, performing grayscale conversion, binarization, image alignment, and plotting optimal displacement charts.

## Notes
- Please ensure there are valid image files in the folder.
- Image alignment performs multi-level displacement search based on pyramid levels, which may require longer processing time depending on the number of images and their resolution.

---

# HDR_Debevec.py

## Description
This Python program implements High Dynamic Range (HDR) image reconstruction and tone mapping. It can synthesize HDR images from a set of images with different exposure times and perform three different tone mapping methods (Drago, Mantiuk, Reinhard). The program uses the Debevec & Malik method to estimate camera response functions and reconstructs radiance values for each pixel based on this response function, followed by tone mapping processing.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- PIL (Pillow)
- SciPy
- tqdm

Install the necessary libraries:
```
pip install opencv-python numpy matplotlib pillow scipy tqdm
```

## Functionality Overview
1. **Load Images and Exposure Times**: Read a set of images from a folder and parse the exposure time for each image.
2. **Calculate Camera Response Curves**: Use the Debevec & Malik method to estimate camera response functions (g(z)) and perform HDR synthesis based on exposure times.
3. **HDR Radiance Map Reconstruction**: Reconstruct the radiance map (in Log form) for each pixel using the estimated camera response curve.
4. **Tone Mapping**: Perform tone mapping from HDR to LDR (Low Dynamic Range), supporting Drago, Mantiuk, and Reinhard methods.
5. **Output Results**: Output HDR images (.hdr format) and tone-mapped LDR images (.jpg format).

## Usage

1. **Run the Program**:
   - After running the program, it will ask you to enter the path to the image folder.
   - The program will automatically read the images in the folder and parse exposure times from the image filenames.

2. **Output Results**:
   - **Camera Response Curves**: The program calculates and plots camera response curves for each color channel, saved as `camera_response_curve_smoothed.png`.
   - **HDR Images**: Save the calculated HDR images in `.hdr` format to `output_result.hdr`.
   - **LDR Images**: Save tone-mapped LDR images in `.jpg` format, including:
     - Drago method result: `output_result_ldr_Drago.jpg`
     - Mantiuk method result: `output_result_ldr_Mantiuk.jpg`
     - Reinhard method result: `output_result_ldr_Reinhard.jpg`
   - **Radiance Map Visualization**: Generate and save radiance map visualization as `radiance_map.jpg`.

3. **Terminal Output**:
   - Display processing status for each step, including number of images read, camera response curve estimation, HDR image generation, tone mapping results, etc.

## Function Descriptions

### `select_image_folder()`
Allows users to input the path to the image folder.

### `load_images_and_exposures(folder_path)`
Read images from the specified folder and parse exposure times. Image filenames should contain exposure times, which the program will automatically parse.

### `weight_function(z, lower_limit=20, upper_limit=235)`
Weight each pixel based on pixel intensity; pixels that are too dark or too bright will have lower weights.

### `solve_g_and_lnE(Z, B, l, w)`
Use the Debevec & Malik method to solve for camera response function `g(z)` and log radiance values `ln(E)` for each pixel.

### `construct_radiance_map(images, times, g)`
Reconstruct the radiance map for each pixel based on the estimated camera response function `g(z)`.

### `write_hdr_image(hdr_data, filename)`
Save the reconstructed HDR image in `.hdr` format using Radiance RGBE format.

### `tone_mapping_Drago(hdr_data)`
Use OpenCV's Drago method for tone mapping HDR images.

### `tone_mapping_dodging_burning(hdr_data)`
Use OpenCV's Mantiuk method for local tone mapping, simulating dodging and burning effects.

### `tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6)`
Use the Reinhard method for global tone mapping of HDR images.

### `smooth_curve(curve, window_length=51, polyorder=3)`
Smooth camera response curves using Savitzky-Golay filter.

### `plot_camera_response_curves(g_channels, folder_path)`
Plot and save camera response curves, including red, green, blue channels and their overlay.

### `save_radiance_map_colormap(hdr_data, output_path)`
Display and save the radiance map (Log Luminance) as a color map to help visualize HDR images.

## Example

```bash
$ python hdr_tonemapping.py
```

After running the program, it will prompt you to enter the image folder path and start processing images. The processing results will be saved in the same folder.

## Notes
- Ensure image filenames contain exposure times (e.g., `img_0.025.jpg`) and all images have the same resolution.
- Supported image formats include `.jpg`, `.png`, `.jpeg`, `.bmp`, `.tiff`.
- If there are fewer than 2 images, the program will not be able to continue.

---

# HDR_Robertson.py

## Description
This Python program is used for synthesizing High Dynamic Range (HDR) images and performing tone mapping. It synthesizes HDR from a set of images with different exposure times using estimation-theoretic methods and uses multiple tone mapping methods (such as Drago, Mantiuk, and Reinhard) to process HDR images, finally generating visualized radiance maps. Additionally, the program outputs camera response curves and performs smoothing processing.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- SciPy (`scipy`)
- Matplotlib

You can install the required libraries using the following command:
```
pip install opencv-python numpy pillow scipy matplotlib
```

## Usage

### 1. Prepare Image Folder
Place a set of images with different exposure times (.jpg, .png, .tiff, etc. formats) in a folder. The filename format should be `img_<exposure_time>.jpg`, such as `img_0.01.jpg`.

### 2. Run the Program
- Enter the path to the image folder, and the program will read images and corresponding exposure times from that folder.
- The program will perform the following operations:
  1. Estimate camera response curves (Camera Response Function, CRF) and plot the curves.
  2. Synthesize HDR images using the "estimation-theoretic" method.
  3. Output visualization results of the radiance map.
  4. Perform tone mapping and generate LDR images using Drago, Mantiuk, and Reinhard methods.
  5. Output HDR and LDR results and save them to the folder.

### 3. Output Results
- **Camera Response Curve Chart**: Display camera response curves for red, green, and blue channels, as well as the three-color overlay curve, saved as `camera_response_curve_smoothed.png`.
- **HDR File**: Save the synthesized HDR image in `.hdr` format to `output_estimation_result.hdr`.
- **LDR Images**: Generate LDR images using different tone mapping methods and save them in `.jpg` format:
  - Drago method: `output_estimation_ldr_Drago.jpg`
  - Mantiuk method: `output_estimation_ldr_Mantiuk.jpg`
  - Reinhard method: `output_estimation_ldr_Reinhard.jpg`
- **Radiance Map Visualization**: Display the radiance map using Log Luminance and save as `radiance_map_estimation.jpg`.

## Function Descriptions

### `select_image_folder()`
Simulates the function for selecting an image folder. In actual situations, file dialog boxes or command line parameters can be used to select folders.

### `load_images_and_exposures(folder_path)`
Read images and corresponding exposure times from the specified folder. Image filenames need to contain exposure times, and the program will parse the filenames to extract exposure times.

### `write_hdr_image(hdr_data, filename)`
Write synthesized HDR image data to `.hdr` files using Radiance RGBE format.

### `weight_function_estimation(I, eps=0.02, sat=0.98)`
Calculate weights for each pixel; pixels that are too dark or too bright will be assigned weight 0, while other pixels are assigned weight 1.

### `construct_hdr_estimation_theoretic(images, times, eps=0.02, sat=0.98)`
Synthesize HDR images using a simplified estimation-theoretic method.

### `smooth_curve(curve, window_length=51, polyorder=3)`
Smooth camera response curves using Savitzky-Golay filter.

### `plot_camera_response_curves(g_channels, folder_path)`
Plot and save camera response curve charts, including curves for red, green, and blue channels, as well as the three-color overlay curve.

### `tone_mapping_Drago(hdr_data)`
Perform tone mapping using OpenCV's Drago method.

### `tone_mapping_dodging_burning(hdr_data)`
Perform tone mapping using OpenCV's Mantiuk method, suitable for simulating dodging and burning effects.

### `tone_mapping_Reinhard(hdr_data, key=0.18, delta=1e-6)`
Perform global tone mapping using the Reinhard method.

### `save_radiance_map_colormap(hdr_data, output_path)`
Save the radiance map (Log Luminance) as a color map to help visualize HDR images.

## Notes
- Please ensure image files conform to naming conventions and have at least two images with different exposure times.
- Images should have the same resolution and be aligned.
- Tone mapping processing may require longer time, depending on the number of images and their resolution.

## Execution Example
```bash
$ python hdr_tonemapping.py
Please enter the image folder path: /path/to/images
Starting camera response curve estimation (Robertson)...
Starting estimation-theoretic HDR synthesis...
HDR synthesis completed.
HDR content range: min=0.0001, max=0.9753
Performing tone mapping...
LDR(Drago) output to: /path/to/images/output_estimation_ldr_Drago.jpg
LDR(Mantiuk) output to: /path/to/images/output_estimation_ldr_Mantiuk.jpg
LDR(Reinhard) output to: /path/to/images/output_estimation_ldr_Reinhard.jpg
```

---

# HDR_color.py

## Description
This Python program is used to calculate brightness and saturation metrics for an image, displaying and saving the results as image files. The program can select an image, calculate the brightness histogram, average brightness, brightness standard deviation, average color saturation, and saturation standard deviation for that image, and save the results in chart format.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Tkinter (for file selection dialog)

You can install the required libraries using the following command:
```
pip install opencv-python numpy matplotlib
```

## Functionality Overview
1. **Read Images**: Select an image for processing.
2. **Calculate Brightness Metrics**: Calculate and display the brightness histogram of the image, and calculate average brightness and brightness standard deviation.
3. **Calculate Saturation Metrics**: Calculate and display the average color saturation and standard deviation of the image.
4. **Image Saving**: Save brightness histogram and original image.
5. **Result Display**: Display average brightness, brightness standard deviation, average saturation, and saturation standard deviation in the terminal.

## Usage

1. **Run the Program**:
   - After executing the program, a file selection dialog will pop up, allowing you to select the image to process.
   - The program will calculate the brightness histogram, brightness metrics, and saturation metrics of the image and save the results.

2. **Output Results**:
   - **Brightness Histogram**: The program will save a brightness histogram image named `brightness_histogram.png`.
   - **Original Image**: The program will save the original image as `original_image.png`.
   - **Terminal Output**: The program will display the image's average brightness, brightness standard deviation, average saturation, and saturation standard deviation.

## Function Descriptions

### `calculate_image_metrics(image_path)`
This function calculates and returns the following image metrics:
- Average brightness (`mean_brightness`)
- Brightness standard deviation (`std_brightness`)
- Average saturation (`mean_saturation`)
- Saturation standard deviation (`std_saturation`)

It also generates and saves:
- Brightness histogram (`brightness_histogram.png`)
- Display and save original image (`original_image.png`)

### `open_image_file()`
Use `Tkinter` to pop up a file selection window, allowing users to select the image file to process.

## Example

```bash
$ python image_metrics.py
```

When running the program, a file selection window will pop up. After selecting the image file you want to analyze, the program will display the results and generate related image files. Example terminal output:

```bash
Average Brightness: 112.45
Brightness Standard Deviation: 42.17
Average Saturation: 0.45
Saturation Standard Deviation: 0.12
```

The generated image files will be saved in the current directory where the program is running.
