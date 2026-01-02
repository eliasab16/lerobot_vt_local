#!/bin/bash

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5A680122821 \
  --robot.id=right_follower \
  --robot.cameras='{"image": {"type": "opencv", "index_or_path": 0, "width": 800, "height": 600, "fps": 30}, "image2": {"type": "opencv", "index_or_path": 1, "width": 800, "height": 600, "fps": 30}, "empty_camera_0": {"type": "opencv", "index_or_path": 3, "width": 800, "height": 600, "fps": 30}}' \
  --display_data=true \
  --dataset.repo_id=eliasab16/eval_xvla-3-cameras-dec16 \
  --dataset.single_task="Insert tip into mounted white device from below" \
  --policy.path=eliasab16/xvla-3-cameras-dec16