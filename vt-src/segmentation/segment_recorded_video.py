# 1. Import the InferencePipeline library
from inference import InferencePipeline
import cv2
import os
import av
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

input_dir = os.getenv("INPUT_VIDEO_PATH")
output_dir = os.getenv("OUTPUT_VIDEO_PATH")
input_video=os.path.join(input_dir, "video1.mp4")
output_video = os.path.join(output_dir, "video1_with_mask.mp4")

os.makedirs(output_dir, exist_ok=True)
fps = 30
output_container = None
output_stream = None
frame_count = 0

def my_sink(result, video_frame):
    global output_container, output_stream, frame_count
    
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
            
            output_container = av.open(output_video, mode='w')
            
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
        
        cv2.imshow("Segmentation Output", frame_with_mask)
        cv2.waitKey(1)

pipeline = InferencePipeline.init_with_workflow(
    api_key=ROBOFLOW_API_KEY,
    workspace_name="lerobotvt-jk4j0",
    workflow_id="background-removal",
    video_reference=input_video,
    max_fps=30,
    on_prediction=my_sink
)

print(f"Starting video processing: {input_video}")
pipeline.start()
pipeline.join()
if output_stream is not None:
    # Flush remaining packets from encoder
    packet = output_stream.encode()
    if packet:
        output_container.mux(packet)

if output_container is not None:
    output_container.close()

cv2.destroyAllWindows()

print(f"\nProcessing complete")
print(f"Total frames processed: {frame_count}")
print(f"Output saved to: {output_video}")
