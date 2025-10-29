# Cell 2: Comparative CV-Only Image Processing Code with CORRECTED Biharmonic Inpainting

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import urllib.request
import torch # To check for CUDA and set device
from PIL import Image # For displaying images easily

# Import the advanced inpainting function and image conversion utilities
from skimage.restoration import inpaint_biharmonic
from skimage import img_as_float, img_as_ubyte # For correct type conversions

print("All libraries imported successfully.")

# --- Configuration for Pixelation ---
PIXELATION_FACTOR = 16 # Lower value = more blocky

# --- Helper Functions ---
def show_multiple_images(images_dict, main_title="Comparative Anonymization Results"):
    """Displays multiple PIL or OpenCV images in a grid."""
    num_images = len(images_dict)
    if num_images == 0: print("No images to display."); return
    num_cols = min(num_images, 2)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows * 6))
    if num_images == 1: axes = np.array([axes])
    axes = axes.flatten()
    for i, (title, img) in enumerate(images_dict.items()):
        ax = axes[i]
        display_img = None
        if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: display_img = img
        ax.imshow(display_img); ax.set_title(title, fontsize=14); ax.axis('off')
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    fig.suptitle(main_title, fontsize=20, y=1.03); plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.show()

def pixelate_region(image_cv, mask_resized_bool):
    """Applies pixelation to the area defined by the mask (OpenCV image)."""
    rows, cols = np.any(mask_resized_bool, axis=1), np.any(mask_resized_bool, axis=0)
    if not np.any(rows) or not np.any(cols): return image_cv.copy()
    rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
    if rmin >= rmax or cmin >= cmax: return image_cv.copy()
    output_image = image_cv.copy(); roi = output_image[rmin:rmax+1, cmin:cmax+1]
    roi_mask = mask_resized_bool[rmin:rmax+1, cmin:cmax+1]
    if roi.size == 0: return output_image
    h, w, _ = roi.shape
    small_roi = cv2.resize(roi, (max(1, w // PIXELATION_FACTOR), max(1, h // PIXELATION_FACTOR)), interpolation=cv2.INTER_NEAREST)
    pixelated_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
    roi_mask_3c = np.stack([roi_mask]*3, axis=-1)
    np.copyto(roi, pixelated_roi, where=roi_mask_3c); return output_image

# --- 2. LOAD YOLO MODEL AND TEST IMAGE ---
print("Loading YOLOv8x-seg model..."); device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8x-seg.pt'); print(f"YOLOv8x-seg loaded. Will run on {device}.")
img_url = "https://ultralytics.com/images/bus.jpg"; img_path = "bus.jpg"
if not os.path.exists(img_path): print(f"Downloading test image from {img_url}..."); urllib.request.urlretrieve(img_url, img_path)
else: print(f"Test image '{img_path}' already exists.")
original_image_cv = cv2.imread(img_path)
if original_image_cv is None: print(f"Error: Could not read image at {img_path}"); exit()
print(f"Successfully loaded image: {img_path}")

# --- 3. RUN YOLO SEGMENTATION ONCE ---
print("Running YOLO segmentation to get masks...")
results = yolo_model(original_image_cv, device=device, verbose=False)
combined_bystander_mask = None

if results[0].boxes is not None and results[0].masks is not None and results[0].masks.data is not None:
    boxes_data = results[0].boxes; masks_data = results[0].masks.data
    areas = []; people_indices = []
    (orig_h, orig_w) = original_image_cv.shape[:2]
    for i in range(len(boxes_data)):
        cls = int(boxes_data.cls[i].cpu().numpy())
        if cls == 0:
            people_indices.append(i); box = boxes_data.xyxy[i].cpu().numpy().astype(int)
            areas.append((box[2] - box[0]) * (box[3] - box[1]))
    if people_indices:
        subject_local_idx = np.argmax(areas); subject_original_idx = people_indices[subject_local_idx]
        print(f"Found {len(people_indices)} people. Subject is at original detection index {subject_original_idx}.")
        combined_bystander_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for i in range(len(masks_data)):
             if i in people_indices and i != subject_original_idx:
                mask_segment = masks_data[i].cpu().numpy()
                mask_resized = cv2.resize(mask_segment, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                mask_resized = (mask_resized > 0.5).astype(np.uint8)
                combined_bystander_mask = np.maximum(combined_bystander_mask, mask_resized).astype(np.uint8)
        if np.any(combined_bystander_mask): print("Combined mask for all bystanders created.")
        else: print("Subject identified, but no other bystanders found."); combined_bystander_mask = None
    else: print("No people found in the image.")
else: print("No objects or masks detected by YOLOv8-seg.")

# --- 4. APPLY CLASSICAL ANONYMIZATION TECHNIQUES ---
results_images = {"Original": original_image_cv.copy()}

if combined_bystander_mask is not None:
    combined_bystander_mask_uint8_255 = (combined_bystander_mask * 255).astype(np.uint8)
    combined_bystander_mask_bool = combined_bystander_mask.astype(bool)

    # Method 1: Inpainting (Navier-Stokes)
    print("\nApplying Classical Inpainting (Navier-Stokes)...")
    ns_inpainted_cv = cv2.inpaint(original_image_cv.copy(), combined_bystander_mask_uint8_255, 5, cv2.INPAINT_NS)
    results_images["Inpainting (Navier-Stokes)"] = ns_inpainted_cv

    # Method 2: Inpainting (Telea)
    print("Applying Classical Inpainting (Telea)...")
    telea_inpainted_cv = cv2.inpaint(original_image_cv.copy(), combined_bystander_mask_uint8_255, 5, cv2.INPAINT_TELEA)
    results_images["Inpainting (Telea)"] = telea_inpainted_cv

    # Method 3: Biharmonic Inpainting (skimage) - CORRECTED
    print("Applying Biharmonic Inpainting (skimage)...")
    # --- Start Correction ---
    # 1. Convert original image to float [0, 1]
    image_float = img_as_float(original_image_cv)

    # 2. Apply inpainting channel by channel on the float image
    biharmonic_inpainted_float = image_float.copy()
    for channel_idx in range(image_float.shape[2]): # Iterate B, G, R
        biharmonic_inpainted_float[:, :, channel_idx] = inpaint_biharmonic(
            image_float[:, :, channel_idx],
            combined_bystander_mask_bool, # Boolean mask needed here
            channel_axis=None # Process single channel
        )

    # 3. Convert the resulting float image back to uint8 [0, 255]
    # Use img_as_ubyte which handles clipping and scaling correctly
    biharmonic_inpainted_cv = img_as_ubyte(biharmonic_inpainted_float)
    # --- End Correction ---
    results_images["Inpainting (Biharmonic)"] = biharmonic_inpainted_cv

    # Method 4: Pixelation
    print("Applying Pixelation...")
    pixelated_cv = pixelate_region(original_image_cv.copy(), combined_bystander_mask_bool)
    results_images["Pixelation"] = pixelated_cv

    print("\nAll anonymization techniques applied.")
else:
    print("\nSkipping anonymization as no bystanders were identified or detected.")

# --- 5. DISPLAY ALL RESULTS ---
show_multiple_images(results_images, "Comparison of Classical CV Anonymization Techniques")
