# Two-stage wire detection pipeline running entirely locally:
#   1. RF-DETR bounding box detection (local checkpoint)
#   2. MobileNetV2 color classification (local checkpoint)
#
# Usage:
#   python color_filtered_bbox_video_local.py --input video.mp4 --color red --display
#   python color_filtered_bbox_video_local.py --input video.mp4 --output output.mp4 --color red
#
# Available colors: green, red, white, yellow

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from rfdetr import RFDETRBase


# ============ CONFIGURATION ============
BBOX_MODEL_PATH = '/Users/elisd/Desktop/vult/models/trained_models/bounding_box_rfdetr_jan7/rfdetr_jan7/checkpoint_best_total.pth'
COLOR_MODEL_PATH = '/Users/elisd/Desktop/vult/models/trained_models/wire_color_detection_imagenet_jan6/imagenet_wire_color_detection_model_jan6.pt'

BBOX_CONFIDENCE_THRESHOLD = 0.7
COLOR_CONFIDENCE_THRESHOLD = 0.8
FRAME_STRIDE = 2  # process every Nth frame (1 = all frames, 2 = every 2nd frame)
BBOX_PADDING_PIXELS = 10  # expand bounding box by this many pixels on each side to avoid object occlusion
IMG_SIZE = 224
AVAILABLE_COLORS = ['green', 'red', 'white', 'yellow']
# =======================================


parser = argparse.ArgumentParser(description='Process video with color-filtered bounding box detection (local models)')
parser.add_argument('--input', '-i', required=True, help='Input video file path')
parser.add_argument('--output', '-o', default=None, help='Output video file path (optional, only saves if provided)')
parser.add_argument('--color', '-c', required=True, nargs='+',
                    help=f'Target color(s) to filter ({", ".join(AVAILABLE_COLORS)})')
parser.add_argument('--display', '-d', action='store_true', help='Show live video preview')
parser.add_argument('--bbox-threshold', type=float, default=BBOX_CONFIDENCE_THRESHOLD,
                    help=f'Bounding box confidence threshold (default: {BBOX_CONFIDENCE_THRESHOLD})')
parser.add_argument('--color-threshold', type=float, default=COLOR_CONFIDENCE_THRESHOLD,
                    help=f'Color confidence threshold (default: {COLOR_CONFIDENCE_THRESHOLD})')
args = parser.parse_args()

input_video_path = args.input
output_video_path = args.output
target_colors = [c.lower() for c in args.color]
bbox_threshold = args.bbox_threshold
color_threshold = args.color_threshold
display_video = args.display
save_video = output_video_path is not None

for c in target_colors:
    if c not in AVAILABLE_COLORS:
        parser.error(f"Invalid color '{c}'. Choose from: {', '.join(AVAILABLE_COLORS)}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS (Apple Silicon)')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Image transform for color classification
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_bbox_model():
    """Load the trained RF-DETR bounding box model."""
    print(f"Loading bbox model from: {BBOX_MODEL_PATH}")
    model = RFDETRBase(pretrain_weights=BBOX_MODEL_PATH)
    print("Loaded RF-DETR Base")
    return model


def load_color_model():
    """Load the trained color classification model."""
    print(f"Loading color model from: {COLOR_MODEL_PATH}")
    checkpoint = torch.load(COLOR_MODEL_PATH, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    # Create mobilenet with same architecture as training
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Color classes: {classes}")
    return model, classes


def predict_color(image_crop, model, classes):
    """Predict color for a cropped image (numpy array BGR format)."""
    # Convert BGR numpy array to RGB PIL Image
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    
    image_tensor = eval_transform(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return {
        'class': classes[predicted.item()],
        'confidence': confidence.item()
    }


def process_video_file(input_video_path, output_video_path, bbox_model, color_model, classes, target_colors):
    """Process video with color-filtered bounding box detection."""
    
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # only create a new video with the predictions if an output path is provided
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps:.2f} fps ({total_frames} frames)")
    print(f"Filtering for color(s): {', '.join(target_colors)}")
    if save_video:
        print(f"Saving to: {output_video_path}")
    
    frame_count = 0
    total_boxes = 0
    matched_boxes = 0
    prev_time = time.time()
    cached_boxes = []  # cache detected boxes for frame skipping
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # only run inference every FRAME_STRIDE frames
        run_inference = (frame_count % FRAME_STRIDE == 0)
        
        if run_inference:
            # Convert BGR to RGB for RF-DETR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # bounding box detection with RF-DETR
            detections = bbox_model.predict(pil_image, threshold=bbox_threshold)
            cached_boxes = []  # reset cache
        
        display_frame = frame.copy()
        
        # Process detections (run inference or use cached boxes)
        if run_inference:
            for i in range(len(detections.xyxy)):
                total_boxes += 1
                
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                bbox_confidence = detections.confidence[i]
                
                # get the cropped image with padding
                padding = 10 # to avoid cropping too much of the wire in case of imperfect bounding box
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(width, x2 + padding)
                y2 = min(height, y2 + padding)
            
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Predict color
                color_result = predict_color(crop, color_model, classes)
                predicted_color = color_result['class']
                color_confidence = color_result['confidence']
                
                # Check if matches target color with sufficient confidence
                if predicted_color in target_colors and color_confidence >= color_threshold:
                    matched_boxes += 1
                    
                    # Calculate padded box for drawing
                    px1 = max(0, x1 - BBOX_PADDING_PIXELS)
                    py1 = max(0, y1 - BBOX_PADDING_PIXELS)
                    px2 = min(width, x2 + BBOX_PADDING_PIXELS)
                    py2 = min(height, y2 + BBOX_PADDING_PIXELS)
                    
                    cached_boxes.append((px1, py1, px2, py2, predicted_color, bbox_confidence))
                    
                    # Draw bounding box (always green)
                    cv2.rectangle(display_frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    
                    # labeling
                    label = f"{predicted_color} | bbox:{bbox_confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(display_frame, (px1, py1 - label_size[1] - 10), 
                                 (px1 + label_size[0], py1), (0, 255, 0), -1)
                    cv2.putText(display_frame, label, (px1, py1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            # Use cached boxes for skipped frames
            for (x1, y1, x2, y2, predicted_color, bbox_confidence) in cached_boxes:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{predicted_color} | bbox:{bbox_confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Calculate and display fps
        current_time = time.time()
        fps_display = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if save_video:
            out.write(display_frame)
        frame_count += 1
        
        if display_video:
            cv2.imshow("Live video", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if out:
        out.release()
    
    return frame_count, total_boxes, matched_boxes


if __name__ == "__main__":
    # Load models
    bbox_model = load_bbox_model()
    color_model, classes = load_color_model()
    
    # Create output directory if saving
    if save_video:
        output_dir = os.path.dirname(output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing: {input_video_path}")
    if save_video:
        print(f"Output: {output_video_path}")
    else:
        print("Output: (not saving)")
    print(f"Target color(s): {', '.join(target_colors)}")
    print(f"Bbox threshold: {bbox_threshold}")
    print(f"Color threshold: {color_threshold}")
    print()
    
    start_time = time.time()
    frame_count, total_boxes, matched_boxes = process_video_file(
        input_video_path, output_video_path, bbox_model, color_model, classes, target_colors
    )
    end_time = time.time()
    
    if display_video:
        cv2.destroyAllWindows()
    
    processing_time = end_time - start_time
    minutes = int(processing_time // 60)
    seconds = processing_time % 60
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Total boxes detected: {total_boxes}")
    print(f"Boxes matching {target_colors}: {matched_boxes}")
    print(f"Match rate: {100*matched_boxes/total_boxes:.1f}%" if total_boxes > 0 else "N/A")
    print(f"Time: {minutes}m {seconds:.1f}s")
    if save_video:
        print(f"Output: {output_video_path}")
    print(f"{'='*60}")
