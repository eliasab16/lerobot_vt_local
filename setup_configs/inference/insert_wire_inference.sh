#!/bin/bash

# Insert wire inference with breaker segmentation
# Uses lerobot-interactive for VLA-guided wire insertion with visual breaker segmentation
#   
# Camera names mapping:
#   wrist_top -> image
#   overhead_top -> image2
#   wrist_bottom -> empty_camera_0

# Configuration
DATASET_REPO_ID="eliasab16/eval_xvla_wire_insert"
BREAKER_MODEL="/Users/elisd/Desktop/vult/models.nosync/trained_models/breaker_segmentation/jan19/best.pt"

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
  --teleop.type=so101_leader \
  --robot.max_relative_target=10 \
  --teleop.port=/dev/tty.usbmodem5A680135321 \
  --teleop.id=right_leader \
  --policy.path=eliasab16/xvla_merged_pick_up_insert_wire_v1_50k \
  --dataset.repo_id=$DATASET_REPO_ID \
  --dataset.single_task="Insert the wire into the red target on the magenta-outlined component" \
  --dataset.episode_time_s=400 \
  --dataset.num_episodes=2 \
  --frame_processor.config='{"model_path": "'"$BREAKER_MODEL"'", "conf_threshold": 0.85, "border_thickness": 6, "border_color": [255, 0, 255], "frame_stride": 1}' \
  --frame_processor.type=breaker_segmentation \
  --display_data=true


