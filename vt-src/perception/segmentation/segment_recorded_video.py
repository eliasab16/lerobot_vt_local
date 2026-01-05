# This is used to add segmentation masks to a recorded video
from inference import InferencePipeline
import cv2
import os
import av
import time
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

input_dir = os.getenv("INPUT_VIDEO_PATH")
output_dir = os.getenv("OUTPUT_VIDEO_PATH")

file_names = []

fps = 30
display_video = False  # set to True to show a live video preview while processing (slower), False for faster processing

output_container = None
output_stream = None
frame_count = 0
current_output_video = None

def process_segmented_frame(result, video_frame):
    global output_container, output_stream, frame_count, current_output_video
    
    mask_viz = result.get("mask_visualization")
    
    if mask_viz:
        frame_with_mask = mask_viz.numpy_image
        
        # We want to stay consistent with the lerobot settings+
        if output_container is None:
            height, width = frame_with_mask.shape[:2]
            
            # Define video codec options (matching lerobot defaults)
            video_options = {
                'crf': '30',      # Quality
                'g': '2',         # Keyframe interval
                'preset': '12'    # Speed preset
            }
            
            output_container = av.open(current_output_video, mode='w')
            
            output_stream = output_container.add_stream('libsvtav1', rate=fps, options=video_options)
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = 'yuv420p'
            
            print(f"Initialized PyAV encoder: {width}x{height} using libsvtav1 codec (crf=30, g=2, preset=12)")
        
        frame_rgb = cv2.cvtColor(frame_with_mask, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(frame_rgb)
        av_frame = av.VideoFrame.from_image(pil_image)
        
        # Encode and mux the frame
        packet = output_stream.encode(av_frame)
        if packet:
            output_container.mux(packet)
        
        frame_count += 1
        
        if display_video:
            cv2.imshow("Segmentation Output", frame_with_mask)
            cv2.waitKey(1)

total_start_time = time.time()
total_frames = 0

for i, file_name in enumerate(file_names, 1):
    # Reset state for each file
    output_container = None
    output_stream = None
    frame_count = 0
    
    input_video = os.path.join(input_dir, file_name)
    output_video = os.path.join(output_dir, file_name)
    current_output_video = output_video
    
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    print(f"\n[{i}/{len(file_names)}] Processing: {file_name}")
    
    pipeline = InferencePipeline.init_with_workflow(
        api_key=ROBOFLOW_API_KEY,
        workspace_name="lerobotvt-jk4j0",
        workflow_id="background-removal",
        video_reference=input_video,
        max_fps=30,
        on_prediction=process_segmented_frame
    )
    
    # Process video
    file_start_time = time.time()
    pipeline.start()
    pipeline.join()
    file_end_time = time.time()
    
    # Flush and close encoder
    if output_stream is not None:
        packet = output_stream.encode()
        if packet:
            output_container.mux(packet)
    
    if output_container is not None:
        output_container.close()
    
    # file summary
    file_processing_time = file_end_time - file_start_time
    file_minutes = int(file_processing_time // 60)
    file_seconds = file_processing_time % 60
    total_frames += frame_count
    print(f"  Frames: {frame_count} | Time: {file_minutes}m {file_seconds:.1f}s | Output: {output_video}")

if display_video:
    cv2.destroyAllWindows()

total_processing_time = time.time() - total_start_time
total_minutes = int(total_processing_time // 60)
total_seconds = total_processing_time % 60

print(f"\n{'='*60}")
print(f"Batch processing complete!")
print(f"Files processed: {len(file_names)}")
print(f"Total frames: {total_frames}")
print(f"Total time: {total_minutes}m {total_seconds:.1f}s")
print(f"{'='*60}")
