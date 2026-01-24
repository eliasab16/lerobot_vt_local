#!/bin/bash

# xVLA evaluation with wire detection (lerobot-record)
# Similar config to interactive_eval.sh but records data
#
# INTERVENTION MODE:
#   Press SPACE to toggle between policy and manual control
#   Interventions are saved to a separate dataset for training
#
# Camera names use descriptive hardware names (wrist_top, overhead_top, wrist_bottom).
# The policy_camera_name field in each camera config maps them to
# what the policy expects (image, image2, empty_camera_0).
# The intervention dataset retains descriptive names.

# Configuration
DATASET_REPO_ID="eliasab16/eval_xvla_wire_pickup"
INTERVENTION_REPO_ID="eliasab16/intervention_eval"

# Clean up existing datasets to avoid conflicts
DATASET_PATH="$HOME/.cache/huggingface/lerobot/$DATASET_REPO_ID"
if [ -d "$DATASET_PATH" ]; then
    echo "Removing existing dataset: $DATASET_PATH"
    rm -rf "$DATASET_PATH"
fi

INTERVENTION_PATH="$HOME/.cache/huggingface/lerobot/$INTERVENTION_REPO_ID"
if [ -d "$INTERVENTION_PATH" ]; then
    echo "Removing existing dataset: $INTERVENTION_PATH"
    rm -rf "$INTERVENTION_PATH"
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680122821 \
  --robot.id=right_follower \
  --robot.cameras='{"wrist_bottom": {"type": "opencv", "index_or_path": 0, "width": 800, "height": 600, "fps": 30, "policy_camera_name": "empty_camera_0"}, "wrist_top": {"type": "opencv", "index_or_path": 1, "width": 800, "height": 600, "fps": 30, "policy_camera_name": "image"}, "overhead_top": {"type": "opencv", "index_or_path": 2, "width": 800, "height": 600, "fps": 30, "policy_camera_name": "image2"}}' \
  --robot.max_relative_target=15 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5A680135321 \
  --teleop.id=right_leader \
  --policy.path=eliasab16/xvla_merged_pick_up_insert_wire_v1_50k \
  --dataset.repo_id=$DATASET_REPO_ID \
  --dataset.intervention_repo_id=$INTERVENTION_REPO_ID \
  --dataset.single_task="Pick up the wire inside the magenta bounding box" \
  --dataset.episode_time_s=400 \
  --dataset.num_episodes=5 \
  --dataset.video_encoding_batch_size=5 \
  --frame_processor.type=wire_detection \
  --frame_processor.config='{"target_colors": ["yellow"], "frame_stride": 2, "bbox_threshold": 0.7, "color_threshold": 0.7}' \
  --display_data=true

# Keyboard Controls (Intervention Mode):
#
#   During POLICY mode (autopilot):
#     SPACE      - Enter intervention mode (disable leader torque, start recording)
#     → (Right)  - Restart current run (no save, no reset)
#     ← (Left)   - Same as right arrow (restart run)
#     ESC        - Stop recording session
#
#   During INTERVENTION mode (manual control):
#     SPACE      - DISCARD intervention, return to policy (same run continues)
#     → (Right)  - SAVE intervention episode, reset environment, start new run
#     ← (Left)   - RESTART recording (clear buffer, stay in intervention mode)
#     ESC        - Stop recording session (discard unsaved intervention)
#
#   During RESET phase (after right arrow saved an intervention):
#     → (Right)  - End reset early, start next run
#     ← (Left)   - UNDO last saved intervention, restart run
#
# Usage Notes:
#   - Each run starts with policy running autonomously (leader mirrors follower)
#   - Press SPACE when policy fails to take manual control
#   - Only RIGHT ARROW during intervention saves the episode
#   - LEFT ARROW during intervention restarts recording (useful if arm drops on entry)
#   - SPACE during intervention discards and returns to policy
#   - num_episodes counts saved intervention episodes, not inference runs
#   - Main dataset is never saved (only used for policy features reference)