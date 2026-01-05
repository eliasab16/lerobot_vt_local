# This script draws bounding boxes on a recorded video using local inference
# To run this script with mps acceleration:
#   cd inference
#   uvicorn cpu_http:app --port 9001 --host 0.0.0.0
#
# Make sure that the server is running on localhost:9001
#
# Usage:
#   python bounding_box_recorded_video_local.py --input path/to/input.mp4 --output path/to/output.mp4

import cv2
import os
import time
import argparse
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "wires-uoqnx/1"

parser = argparse.ArgumentParser(description='Process recorded video with bounding box detection')
parser.add_argument('--input', '-i', required=True, help='Input video file path')
parser.add_argument('--output', '-o', required=True, help='Output video file path')
parser.add_argument('--display', '-d', action='store_true', help='Show live video preview while processing')
args = parser.parse_args()

input_video_path = args.input
output_video_path = args.output
display_video = args.display

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=ROBOFLOW_API_KEY
)

config = InferenceConfiguration(
    confidence_threshold=0.8,
    iou_threshold=0.5
)
client.configure(config)
client.select_model(MODEL_ID)

def process_video_file(input_video_path, output_video_path):
    """Process a single video file with bounding box detection"""
    
    cap = cv2.VideoCapture(input_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {width}x{height} @ {fps:.2f} fps")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # local inference
        result = client.infer(frame)
        
        # Draw bounding boxes
        display_frame = frame.copy()
        if result and 'predictions' in result:
            for pred in result['predictions']:
                # Get bounding box coordinates
                x = int(pred.get('x', 0))
                y = int(pred.get('y', 0))
                w = int(pred.get('width', 0))
                h = int(pred.get('height', 0))
                
                # Calculate corner points (x, y is center point)
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2
                
                # Draw bounding box rectangle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with class name and confidence
                class_name = pred.get('class', 'unknown')
                confidence = pred.get('confidence', 0)
                label = f"{class_name}: {confidence:.2f}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        out.write(display_frame)
        
        frame_count += 1
        
        if display_video:
            cv2.imshow("Bounding Box Output", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    
    return frame_count

os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

print(f"Processing: {input_video_path}")
print(f"Output: {output_video_path}")

start_time = time.time()
frame_count = process_video_file(input_video_path, output_video_path)
end_time = time.time()

if display_video:
    cv2.destroyAllWindows()

processing_time = end_time - start_time
minutes = int(processing_time // 60)
seconds = processing_time % 60

print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Frames: {frame_count}")
print(f"Time: {minutes}m {seconds:.1f}s")
print(f"Output: {output_video_path}")
print(f"{'='*60}")
