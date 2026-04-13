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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class SO110LeaderConfig:
    """Base configuration class for SO-110 Leader teleoperators (8-DOF arm)."""

    # Port to connect to the arm
    port: str

    # SO-110 uses normalized range by default (not degrees)
    use_degrees: bool = False

    # Motors that do 360-degree continuous rotation (use full 0-4095 range during calibration)
    full_turn_motors: list[str] = field(
        default_factory=lambda: ["shoulder_pan", "lower_arm", "wrist_pan"]
    )


@TeleoperatorConfig.register_subclass("so110_leader")
@dataclass
class SO110LeaderTeleopConfig(TeleoperatorConfig, SO110LeaderConfig):
    pass
