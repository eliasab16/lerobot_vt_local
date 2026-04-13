#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import cached_property

from lerobot.teleoperators.so110_leader import SO110Leader, SO110LeaderTeleopConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_so110_leader import BiSO110LeaderConfig

logger = logging.getLogger(__name__)


class BiSO110Leader(Teleoperator):
    """Bimanual SO-110 leader: composes two SO110Leader arms with per-arm freeze control."""

    config_class = BiSO110LeaderConfig
    name = "bi_so110_leader"

    def __init__(self, config: BiSO110LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO110LeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            use_degrees=config.left_arm_config.use_degrees,
            full_turn_motors=config.left_arm_config.full_turn_motors,
        )

        right_arm_config = SO110LeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            use_degrees=config.right_arm_config.use_degrees,
            full_turn_motors=config.right_arm_config.full_turn_motors,
        )

        self.left_arm = SO110Leader(left_arm_config)
        self.right_arm = SO110Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    # --- Per-arm freeze control ---

    @property
    def is_left_frozen(self) -> bool:
        return self.left_arm.is_frozen

    @property
    def is_right_frozen(self) -> bool:
        return self.right_arm.is_frozen

    def freeze_left(self) -> None:
        self.left_arm.freeze()

    def unfreeze_left(self) -> None:
        self.left_arm.unfreeze()

    def toggle_freeze_left(self) -> bool:
        return self.left_arm.toggle_freeze()

    def freeze_right(self) -> None:
        self.right_arm.freeze()

    def unfreeze_right(self) -> None:
        self.right_arm.unfreeze()

    def toggle_freeze_right(self) -> bool:
        return self.right_arm.toggle_freeze()

    # --- Teleoperator interface ---

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        action_dict = {}

        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
