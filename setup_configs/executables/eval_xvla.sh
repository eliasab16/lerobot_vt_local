#!/bin/bash

# xVLA evaluation with wire detection (lerobot-record)
# Similar config to interactive_eval.sh but records data
#   
# Camera names mapping:
#   wrist_top -> image
#   overhead_top -> image2
#   wrist_bottom -> empty_camera_0

# Configuration
DATASET_REPO_ID="eliasab16/eval_xvla_wire_pickup"

# Clean up existing dataset to avoid conflicts
DATASET_PATH="$HOME/.cache/huggingface/lerobot/$DATASET_REPO_ID"
if [ -d "$DATASET_PATH" ]; then
    echo "Removing existing dataset: $DATASET_PATH"
    rm -rf "$DATASET_PATH"
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680122821 \
  --robot.id=right_follower \
  --robot.cameras='{"empty_camera_0": {"type": "opencv", "index_or_path": 0, "width": 800, "height": 600, "fps": 30}, "image": {"type": "opencv", "index_or_path": 1, "width": 800, "height": 600, "fps": 30}, "image2": {"type": "opencv", "index_or_path": 2, "width": 800, "height": 600, "fps": 30}}' \
  --robot.max_relative_target=15 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5A680135321 \
  --teleop.id=right_leader \
  --policy.path=eliasab16/xvla_wire_pickup_green_bbox_jan13_20k \
  --dataset.repo_id=$DATASET_REPO_ID \
  --dataset.single_task="Pick up the wire inside the green bounding box" \
  --dataset.episode_time_s=250 \
  --dataset.num_episodes=1 \
  --frame_processor.type=wire_detection \
  --frame_processor.config='{"target_colors": ["yellow"], "frame_stride": 2, "bbox_threshold": 0.7, "color_threshold": 0.7}' \
  --display_data=true