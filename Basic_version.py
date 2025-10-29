# Cell 2: Comparative CV-Only Image Processing Code (Fully Weighted Heuristic)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import urllib.request
import torch # To check for CUDA and set device
from PIL import Image # For displaying images easily
import time # For basic timing

# Import the advanced inpainting function and image conversion utilities
from skimage.restoration import inpaint_biharmonic
from skimage import img_as_float, img_as_ubyte # For correct type conversions

print("All libraries imported successfully.")

# --- Configuration ---
PIXELATION_FACTOR = 16
HIERARCHICAL_LEVELS = 2
NS_RADIUS_SPSF = 5
BLUR_KERNEL_SPSF = (51, 51)
HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"

# --- Fully Weighted Heuristic Weights (Adjust these) ---
WEIGHT_FACE = 0.40      # Priority for having a detected face
WEIGHT_AREA = 0.20      # Importance of size
WEIGHT_SALIENCY = 0.15  # Importance of visual attention
WEIGHT_CENTRALITY = 0.15# Importance of being near the center
WEIGHT_HOG_MAG = 0.10   # Importance of HOG feature magnitude (detail proxy)

# --- Helper Functions ---
def show_multiple_images(images_dict, main_title="Comparative Anonymization Results"):
    """Displays multiple PIL or OpenCV images in a grid."""
    num_images = len(images_dict);
    if num_images == 0: print("No images to display."); return
    num_cols = min(num_images, 3); num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
    if num_images == 1: axes = np.array([axes])
    axes = axes.flatten()
    for i, (title, img) in enumerate(images_dict.items()):
        ax = axes[i]; display_img = None
        if isinstance(img, np.ndarray) and len(img.shape) == 3 and img.shape[2] == 3: display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else: display_img = img
        ax.imshow(display_img); ax.set_title(title, fontsize=11); ax.axis('off')
    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    fig.suptitle(main_title, fontsize=16, y=1.01); plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.show()

def pixelate_region(image_cv, mask_resized_bool):
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

def spsf_inpaint(image_cv, mask_uint8_255, mask_bool, ns_radius, blur_kernel):
    inpainted_ns = cv2.inpaint(image_cv.copy(), mask_uint8_255, ns_radius, cv2.INPAINT_NS)
    blurred_ns = cv2.GaussianBlur(inpainted_ns, blur_kernel, 0)
    final_image = np.where(np.stack([mask_bool]*3, axis=-1), blurred_ns, inpainted_ns).astype(np.uint8)
    return final_image

def hierarchical_inpaint(image_cv, mask_uint8_255, levels, core_inpaint_flag=cv2.INPAINT_TELEA, radius=5):
    current_image = image_cv.copy(); current_mask = mask_uint8_255.copy()
    pyramid_images = [current_image]; pyramid_masks = [current_mask]
    actual_levels = 0
    for k in range(levels):
        img_down = cv2.pyrDown(pyramid_images[0])
        if img_down.shape[0] < 1 or img_down.shape[1] < 1: break
        mask_down = cv2.pyrDown(pyramid_masks[0], dstsize=(img_down.shape[1], img_down.shape[0]))
        mask_down = ((mask_down > 127) * 255).astype(np.uint8)
        pyramid_images.insert(0, img_down); pyramid_masks.insert(0, mask_down)
        actual_levels += 1
    if actual_levels < 0: return image_cv
    inpainted_coarse = cv2.inpaint(pyramid_images[0], pyramid_masks[0], radius, core_inpaint_flag)
    current_inpainted = inpainted_coarse
    for i in range(1, actual_levels + 1):
        upsampled_image = cv2.pyrUp(current_inpainted, dstsize=(pyramid_images[i].shape[1], pyramid_images[i].shape[0]))
        level_mask = pyramid_masks[i]
        current_inpainted = np.where(np.stack([level_mask]*3, axis=-1) == 0, pyramid_images[i], upsampled_image)
    return current_inpainted.astype(np.uint8)

# --- Heuristic Helpers ---
def calculate_normalized_area(box, frame_shape): x1, y1, x2, y2 = box; area = (x2 - x1) * (y2 - y1); total_area = frame_shape[0] * frame_shape[1]; return area / total_area if total_area > 0 else 0
def calculate_avg_saliency(saliency_map, mask): saliency_map_float = saliency_map.astype(np.float32) / 255.0; mask_bool = mask.astype(bool); return np.mean(saliency_map_float[mask_bool]) if np.any(mask_bool) else 0.0
def calculate_centrality_score(box, frame_shape): frame_h, frame_w = frame_shape[:2]; frame_center_x, frame_center_y = frame_w / 2, frame_h / 2; x1, y1, x2, y2 = box; box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2; dist_x, dist_y = abs(box_center_x - frame_center_x), abs(box_center_y - frame_center_y); max_dist = np.sqrt((frame_w/2)**2 + (frame_h/2)**2); dist = np.sqrt(dist_x**2 + dist_y**2); return max(0.0, 1.0 - (dist / max_dist)) if max_dist > 0 else 1.0
def calculate_hog_magnitude(roi_gray, hog_descriptor):
    """Calculates normalized HOG descriptor magnitude."""
    if roi_gray.size == 0: return 0.0
    # Resize ROI to standard HOG size
    roi_resized = cv2.resize(roi_gray, (64, 128))
    hog_features = hog_descriptor.compute(roi_resized)
    if hog_features is None: return 0.0
    # Calculate L2 norm (magnitude) and normalize heuristically
    # Max possible HOG magnitude depends on gradient strengths, hard to set a fixed max.
    # We can normalize based on the *potential* maximum in a region, or just use raw magnitude scaled down.
    magnitude = np.linalg.norm(hog_features)
    # Simple scaling heuristic (adjust 10000 based on typical values observed)
    normalized_magnitude = min(1.0, magnitude / 10000.0)
    return normalized_magnitude


# --- 2. LOAD MODELS AND TEST IMAGE ---
print("Loading YOLOv8x-seg model..."); device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8x-seg.pt'); print(f"YOLOv8x-seg loaded. Will run on {device}.")
print("Loading Haar Cascade face detector...");
if not os.path.exists(HAAR_CASCADE_FILE): print(f"ERROR: Haar file not found."); exit()
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE); print("Face detector loaded.")
# Initialize HOG descriptor object ONCE
hog = cv2.HOGDescriptor()
print("HOG descriptor initialized.")
img_url = "https://ultralytics.com/images/bus.jpg"; img_path = "bus.jpg"
if not os.path.exists(img_path): print(f"Downloading test image..."); urllib.request.urlretrieve(img_url, img_path)
else: print(f"Test image exists.")
original_image_cv = cv2.imread(img_path)
if original_image_cv is None: print(f"Error reading image."); exit()
gray_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
print(f"Successfully loaded image.")

# --- 3. RUN YOLO, EXTRACT FEATURES, IDENTIFY SUBJECT ---
print("Running YOLO, features, face detection, and subject ID using Fully Weighted Heuristic...")
start_yolo = time.time(); results = yolo_model(original_image_cv, device=device, verbose=False); end_yolo = time.time()
print(f"  YOLO Time: {end_yolo - start_yolo:.2f}s")
saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliency_map) = saliency_detector.computeSaliency(original_image_cv)
saliency_map_norm = (saliency_map * 255).astype(np.uint8) if success else np.zeros(original_image_cv.shape[:2], dtype=np.uint8)
combined_bystander_mask = None; people_features = []

if results[0].boxes is not None and results[0].masks is not None and results[0].masks.data is not None:
    boxes_data = results[0].boxes; masks_data = results[0].masks.data; (orig_h, orig_w) = original_image_cv.shape[:2]
    for i in range(len(boxes_data)):
        cls = int(boxes_data.cls[i].cpu().numpy())
        if cls == 0: # Person
            box = boxes_data.xyxy[i].cpu().numpy().astype(int); x1, y1, x2, y2 = box
            mask_segment = masks_data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_segment, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask_resized = (mask_resized > 0.5).astype(np.uint8)
            # --- Face Detection ---
            person_roi_gray = gray_image_cv[max(0,y1):min(orig_h,y2), max(0,x1):min(orig_w,x2)]; face_score = 0.0
            if person_roi_gray.size > 0:
                min_face_h, min_face_w = max(20, int((y2 - y1) * 0.1)), max(20, int((x2 - x1) * 0.1))
                faces = face_cascade.detectMultiScale(person_roi_gray, 1.1, 5, minSize=(min_face_w, min_face_h))
                if len(faces) > 0:
                    face_score = 0.5; largest_face_area = 0 # Base score 0.5 for face presence
                    for (fx, fy, fw, fh) in faces: largest_face_area = max(largest_face_area, fw * fh)
                    person_roi_area = (x2 - x1) * (y2 - y1)
                    if person_roi_area > 0: face_score += min(0.5, (largest_face_area / person_roi_area) * 1.0) # Bonus up to +0.5
            # --- Other Features ---
            norm_area = calculate_normalized_area(box, original_image_cv.shape)
            avg_sal = calculate_avg_saliency(saliency_map_norm, mask_resized)
            centrality = calculate_centrality_score(box, original_image_cv.shape)
            hog_mag = calculate_hog_magnitude(person_roi_gray, hog) # Use HOG magnitude

            # --- Combined Weighted Score ---
            combined_score = (WEIGHT_FACE * face_score +
                              WEIGHT_AREA * norm_area +
                              WEIGHT_SALIENCY * avg_sal +
                              WEIGHT_CENTRALITY * centrality +
                              WEIGHT_HOG_MAG * hog_mag)

            people_features.append({'original_index': i, 'mask': mask_resized, 'score': combined_score,
                                    'details': f"Face={face_score:.2f}, A={norm_area:.2f}, S={avg_sal:.2f}, C={centrality:.2f}, H={hog_mag:.2f}"})

    if people_features:
        people_features.sort(key=lambda x: x['score'], reverse=True)
        subject_original_idx = people_features[0]['original_index']
        print(f"Found {len(people_features)} people. Subject at index {subject_original_idx} (Score: {people_features[0]['score']:.3f}). Details: {people_features[0]['details']}")
        combined_bystander_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for person in people_features[1:]: combined_bystander_mask = np.maximum(combined_bystander_mask, person['mask']).astype(np.uint8)
        if np.any(combined_bystander_mask): print("Combined mask created.")
        else: print("Subject found, no bystanders."); combined_bystander_mask = None
    else: print("No people found.")
else: print("No objects/masks detected.")

# --- 4. APPLY CLASSICAL ANONYMIZATION TECHNIQUES ---
results_images = {"Original": original_image_cv.copy()}

if combined_bystander_mask is not None:
    combined_bystander_mask_uint8_255 = (combined_bystander_mask * 255).astype(np.uint8)
    combined_bystander_mask_bool = combined_bystander_mask.astype(bool)

    methods_to_run = {
        "Inpainting (NS)": lambda img, mask_u8, mask_b: cv2.inpaint(img, mask_u8, 5, cv2.INPAINT_NS),
        "Inpainting (Telea)": lambda img, mask_u8, mask_b: cv2.inpaint(img, mask_u8, 5, cv2.INPAINT_TELEA),
        "Inpainting (Biharmonic)": lambda img, mask_u8, mask_b: img_as_ubyte(inpaint_biharmonic(img_as_float(img), mask_b, channel_axis=-1)),
        "Inpainting (Hierarchical Telea)": lambda img, mask_u8, mask_b: hierarchical_inpaint(img, mask_u8, HIERARCHICAL_LEVELS, cv2.INPAINT_TELEA, 5),
        "Inpainting (SPSF - Custom)": lambda img, mask_u8, mask_b: spsf_inpaint(img, mask_u8, mask_b, NS_RADIUS_SPSF, BLUR_KERNEL_SPSF),
        "Pixelation": lambda img, mask_u8, mask_b: pixelate_region(img, mask_b),
    }

    for name, func in methods_to_run.items():
        print(f"\nApplying {name}...")
        start_time = time.time()
        inpainted_img = func(original_image_cv.copy(), combined_bystander_mask_uint8_255, combined_bystander_mask_bool)
        end_time = time.time()
        print(f"  Time: {end_time - start_time:.2f}s")
        results_images[name] = inpainted_img

    print("\nAll anonymization techniques applied.")
else:
    print("\nSkipping anonymization - no bystanders identified/detected.")

# --- 5. DISPLAY ALL RESULTS ---
show_multiple_images(results_images, "Comparison of CV Anonymization (Fully Weighted Heuristic)")
