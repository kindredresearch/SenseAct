"""
AX-12 definitions
"""

#pylint: disable=invalid-name

from .dxl_unit_conv import (
    Affine,
    AngleVelRange,
    SignedPercentRange,
    BaudConversion,
    ReturnTimeDelayConversion,
    BooleanFlag,
    ComplianceSlope)
from .dxl_reg import Reg, ContiguousRegisters
from math import pi

EightBits = (0, 255)
TenBits = (0, 1023)
# N.B.: Angle range of (-150, 150) is inconsistent with Robotis docs which
# are (0, 300) but consistent with our historic use.
AngleRange = Affine(-150 * pi / 180., 150 * pi / 180.) # Angles in radians
PercentRange = Affine(0, 100)
VoltRange = Affine(0, 25.5)
TempRange = Affine(0, 255)
SpeedRange = Affine(0, 114 * 360 * pi / 180. / 60.)  # 114 rot/min -> rad/sec


AX12 = ContiguousRegisters(
    Reg('version_0', 12),
    Reg('version_1', 0),
    Reg('firmware', 0),
    Reg('bus_id', 1),
    Reg('baud', 1, x_lim=EightBits, unit=BaudConversion()),
    Reg('rtd', 250, x_lim=EightBits, unit=ReturnTimeDelayConversion()),
    Reg('angle_limit_cw', (0, 0), x_lim=TenBits, unit=AngleRange),
    Reg('angle_limit_ccw', (255, 3), x_lim=TenBits, unit=AngleRange),
    Reg('reserved_0', 0),
    Reg('temp_limit_high', 70, x_lim=EightBits, unit=TempRange),
    Reg('voltage_limit_low', 60, x_lim=EightBits, unit=VoltRange),
    Reg('voltage_limit_high', 140, x_lim=EightBits, unit=VoltRange),
    Reg('max_torque', (255, 3), x_lim=TenBits, unit=PercentRange),
    Reg('status_return', 2),
    Reg('alarm_led', 36),
    Reg('alarm_shutdown', 36),
    Reg('reserved_1', (0, 0)),
    Reg('reserved_2', (0, 0)),
    Reg('reserved_3', 0),
    Reg('torque_enable', 0, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('led', 0, x_lim=(0, 1), unit=BooleanFlag()),
    Reg('cw_compliance_margin', 1, x_lim=EightBits, unit=AngleRange),
    Reg('ccw_compliance_margin', 1, x_lim=EightBits, unit=AngleRange),
    Reg('cw_compliance_slope', 32, x_lim=EightBits, unit=ComplianceSlope()),
    Reg('ccw_compliance_slope', 32, x_lim=EightBits, unit=ComplianceSlope()),
    Reg('goal_pos', (0, 0), x_lim=TenBits, unit=AngleRange),
    Reg('moving_speed', (0, 0), x_lim=TenBits, unit=SpeedRange),
    Reg('torque_limit', (255, 3), x_lim=TenBits, unit=PercentRange),
    Reg('present_pos', (0, 0), x_lim=TenBits, unit=AngleRange),
    Reg('present_speed', (0, 0),
        x_lim=(0, 2047),
        unit=AngleVelRange(114 * 360 * pi / 180. / 60.)),
    Reg('present_load', (0, 0), x_lim=(0, 2047), unit=SignedPercentRange()),
    Reg('voltage', 0, x_lim=EightBits, unit=VoltRange),
    Reg('temperature', 0, x_lim=EightBits, unit=TempRange),
    Reg('registered', 0),
    Reg('reserved_4', 0),
    Reg('moving', 0),
    Reg('lock', 0),
    Reg('punch', (32, 0), x_lim=(0x20, 0x3FF)))
