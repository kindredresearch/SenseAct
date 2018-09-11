"""MX-64 definitions"""

#pylint: disable=invalid-name

from .dxl_unit_conv import (Affine,
                            AngleVelRange,
                            SignedPercentRange,
                            BaudConversion,
                            BooleanFlag,
                            RawValue,
                            ReturnTimeDelayConversion,
                            CurrentConversion,
                            GoalTorqueConversion)
from .dxl_reg import Reg, ContiguousRegisters
from .utils import Bits8, Bits10, Bits12
from math import pi

# N.B.: Angle range of (-180, 180) is inconsistent with Robotis docs which
# are (0, 360) but consistent with our historic use.
# AngleRange = Affine(-180, 180)
AngleRange = Affine(-pi, pi)
# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# XXX should be N/m, but chosen to match PyPot
PercentRange = Affine(0, 100)
VoltRange = Affine(0, 25.5)
TempRange = Affine(0, 255)
# SpeedRange = Affine(0, 114 * 360 / 60.)  # 114 rot/min -> deg/sec
SpeedRange = Affine(0, 114 * 2 * pi / 60.)  # 114 rot/min -> rad/sec


MX64 = ContiguousRegisters(
    # 0
    Reg('version_0', 54, w=False, unit=RawValue()),
    Reg('version_1', 1, w=False, unit=RawValue()),
    Reg('firmware', 0, w=False, unit=RawValue()),
    Reg('bus_id', 1, w=True, unit=RawValue()),
    # 4
    Reg('baud', 34, x_lim=Bits8, unit=BaudConversion(), w=True),
    Reg('rtd', 250, x_lim=Bits8, unit=ReturnTimeDelayConversion(), w=True),
    Reg('angle_limit_cw', (0, 0), x_lim=Bits12, unit=AngleRange, w=True),
    # 8
    Reg('angle_limit_ccw', (255, 15), x_lim=Bits12, unit=AngleRange, w=True),
    Reg('reserved_0', 0, w=False, unit=RawValue()),
    Reg('temp_limit_high', 80, x_lim=Bits8, unit=TempRange, w=False),
    # 12
    Reg('voltage_limit_low', 60, x_lim=Bits8, unit=VoltRange, w=False),
    Reg('voltage_limit_high', 160, x_lim=Bits8, unit=VoltRange, w=False),
    Reg('max_torque', (255, 3), x_lim=Bits10, unit=PercentRange, w=True),
    # 16
    Reg('status_return', 2, w=True, unit=RawValue()),
    Reg('alarm_led', 36, w=True, unit=RawValue()),
    Reg('alarm_shutdown', 36, w=True, unit=RawValue()),
    Reg('reserved_19', 0, w=False, unit=RawValue()),
    # 20
    Reg('multi_turn_offset', (0, 0), x_lim=Bits12, w=True, unit=RawValue()),
    Reg('resolution_divider', 1, x_lim=Bits8, w=True, unit=RawValue()),
    Reg('reserved_23', 0, w=False, unit=RawValue()),
    # 24
    Reg('torque_enable', 0, w=True, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('led', 0, w=True, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('derivative_gain', 0, w=True, x_lim=Bits8, unit=RawValue()),
    Reg('integral_gain', 0, w=True, x_lim=Bits8, unit=RawValue()),
    # 28
    Reg('proportional_gain', 32, w=True, x_lim=Bits8, unit=RawValue()),
    Reg('reserved_29', 0, w=False, x_lim=Bits8, unit=RawValue()),
    # 30
    Reg('goal_pos', (0, 0), x_lim=Bits12, unit=AngleRange, w=True),
    # 32
    Reg('moving_speed', (0, 0), x_lim=(0, 2047),
        unit=AngleVelRange(114 * 360 / 60.),
        w=True),
    Reg('torque_limit', (255, 3), x_lim=Bits10, unit=PercentRange, w=True),
    # 36
    Reg('present_pos', (0, 0), x_lim=Bits12, unit=AngleRange, w=False),
    Reg('present_speed',
        (0, 0),
        x_lim=(0, 2047),
        unit=AngleVelRange(114 * 360 / 60.),
        w=False),
    # 40
    Reg('present_load',
        (0, 0),
        x_lim=(0, 2047),
        unit=SignedPercentRange(), w=False),
    Reg('voltage', 0, x_lim=Bits8, unit=VoltRange, w=False),
    Reg('temperature', 0, x_lim=Bits8, unit=TempRange, w=False),
    # 44
    Reg('registered', 0, w=False, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('reserved_45', 0, w=False, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('moving', 0, w=True, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('lock', 0, w=True, x_lim=(0, 1), unit=BooleanFlag()),
    # 48
    Reg('punch', (0, 0), x_lim=(0x00, 0x3FF), w=True, unit=RawValue()),
    Reg('reserved_50_51', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_52_53', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_54_55', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_56_57', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_58_59', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_60_61', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_62_63', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_64_65', (0, 0), w=False, unit=RawValue()),
    Reg('reserved_66_67', (0, 0), w=False, unit=RawValue()),
    # 68
    Reg('current', (0, 0), w=False, unit=CurrentConversion()),
    # 70
    Reg('torque_control_mode_enable', 0, w=True, x_lim=(0, 1), unit=BooleanFlag()),
    # 71
    Reg('goal_torque', (0, 0), w=True, unit=GoalTorqueConversion()),
    # 73
    Reg('goal_acceleration', 0, x_lim=Bits8, w=True, unit=RawValue())
)

