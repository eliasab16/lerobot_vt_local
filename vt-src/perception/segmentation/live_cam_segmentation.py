import cv2
import os
import time

# Not working (used to force CoreML execution provider for mps)
# os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CoreMLExecutionProvider]"

from dotenv import load_dotenv
from inference import InferencePipeline
import supervision as sv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

prev_time = time.time()

mask_annotator = sv.MaskAnnotator()

def segment_frame(result, video_frame):
    global prev_time
    frame = video_frame.image.copy()
    
    # Draw segmentation masks
    if result.get("predictions"):
        detections = sv.Detections.from_inference(result)
        frame = mask_annotator.annotate(scene=frame, detections=detections)
    
    # Fps calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

pipeline = InferencePipeline.init(
    model_id=MODEL_ID,
    video_reference=0,
    max_fps=30,
    on_prediction=segment_frame,
    api_key=ROBOFLOW_API_KEY
)

print("Starting segmentation, press 'q' to quit.")
pipeline.start()
pipeline.join()
cv2.destroyAllWindows()
