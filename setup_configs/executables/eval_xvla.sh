#!/bin/bash

# "rename_map": {
#   "observation.images.wrist_top": "observation.images.image",
#   "observation.images.overhead_top": "observation.images.image2",
#   "observation.images.wrist_bottom": "observation.images.empty_camera_0"
# }

lerobot-record \
  --segmentation.enabled=false \
  --segmentation.api_url=http://localhost:9001 \
  --segmentation.model_id=switches-wymit/2 \
  --segmentation.confidence_threshold=0.8 \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680122821 \
  --robot.id=right_follower \
  --robot.cameras='{"image": {"type": "opencv", "index_or_path": 0, "width": 800, "height": 600, "fps": 30, "enable_segmentation": true}, "image2": {"type": "opencv", "index_or_path": 1, "width": 800, "height": 600, "fps": 30, "enable_segmentation": true}, "empty_camera_0": {"type": "opencv", "index_or_path": 2, "width": 800, "height": 600, "fps": 30, "enable_segmentation": true}}' \
  --robot.max_relative_target=4. \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5A680110941 \
  --teleop.id=right_leader \
  --display_data=true \
  --dataset.repo_id=eliasab16/eval_xvlm_mask_dec21_10k_steps \
  --dataset.single_task="Insert tip into device highlighted in green from below" \
  --dataset.episode_time_s=250 \
  --dataset.num_episodes=1 \
  --policy.path=eliasab16/xvlm_mask_dec21_15k_steps \