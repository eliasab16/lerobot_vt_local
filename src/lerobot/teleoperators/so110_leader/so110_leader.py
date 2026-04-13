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
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_so110_leader import SO110LeaderTeleopConfig

logger = logging.getLogger(__name__)


class SO110Leader(Teleoperator):
    """
    SO-110 leader teleoperator arm with 8 DOF (STS3215 servos).
    Supports freeze/unfreeze to toggle torque for holding arm position.
    """

    config_class = SO110LeaderTeleopConfig
    name = "so110_leader"

    def __init__(self, config: SO110LeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        self._frozen = False
        self._action_offset: dict[str, float] = {}  # additive offset for takeover alignment
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_flex": Motor(1, "sts3215", norm_mode_body),
                "shoulder_pan": Motor(2, "sts3215", norm_mode_body),
                "upper_arm": Motor(3, "sts3215", norm_mode_body),
                "elbow_flex": Motor(4, "sts3215", norm_mode_body),
                "lower_arm": Motor(5, "sts3215", norm_mode_body),
                "wrist_flex": Motor(6, "sts3215", norm_mode_body),
                "wrist_pan": Motor(7, "sts3215", norm_mode_body),
                "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = self.config.full_turn_motors or []
        unknown_range_motors = [m for m in self.bus.motors if m not in full_turn_motors]

        if unknown_range_motors:
            motors_str = ", ".join(f"'{m}'" for m in unknown_range_motors)
            print(
                f"Move joints {motors_str} sequentially through their "
                "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
            )
            range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        else:
            range_mins, range_maxes = {}, {}

        for m in full_turn_motors:
            range_mins[m] = 0
            range_maxes[m] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def freeze(self) -> None:
        """Enable torque to hold current position. Read present positions first, then lock."""
        if self._frozen:
            return
        positions = self.bus.sync_read("Present_Position", num_retry=3)
        self.bus.sync_write("Goal_Position", positions, num_retry=3)
        self.bus.enable_torque(num_retry=3)
        self._frozen = True
        logger.info(f"{self} frozen.")

    def unfreeze(self) -> None:
        """Disable torque so the arm is backdrivable again."""
        if not self._frozen:
            return
        self.bus.disable_torque(num_retry=3)
        self._frozen = False
        logger.info(f"{self} unfrozen.")

    def toggle_freeze(self) -> bool:
        """Toggle freeze state. Returns the new frozen state."""
        if self._frozen:
            self.unfreeze()
        else:
            self.freeze()
        return self._frozen

    def set_takeover_offset(self, follower_pos: dict[str, float]) -> None:
        """Compute offset so follower doesn't jump when human takes over.

        Call this at the moment the human presses the takeover key.
        follower_pos: the follower's position when the policy was paused (keys like 'motor.pos').
        """
        leader_pos = self.bus.sync_read("Present_Position")
        self._action_offset = {}
        for motor, leader_val in leader_pos.items():
            key = f"{motor}.pos"
            if key in follower_pos:
                self._action_offset[key] = follower_pos[key] - leader_val
        logger.info(f"{self} takeover offset set:")
        for k, v in self._action_offset.items():
            fpos = follower_pos.get(k, "?")
            lpos = leader_pos.get(k.removesuffix(".pos"), "?")
            logger.info(f"  {k}: follower={fpos} leader={lpos} offset={v}")

    def clear_offset(self) -> None:
        """Clear the takeover offset (call at episode reset)."""
        self._action_offset = {}

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        # Apply takeover offset if set
        if self._action_offset:
            action = {k: v + self._action_offset.get(k, 0.0) for k, v in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Write follower positions to the leader as goal positions (for tracing).

        Only writes if torque is enabled (frozen/tracing mode). Safe to call at any time.
        """
        if not self._frozen:
            return
        goal_pos = {key.removesuffix(".pos"): val for key, val in feedback.items() if key.endswith(".pos")}
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos, num_retry=0)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._frozen:
            self.unfreeze()
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
