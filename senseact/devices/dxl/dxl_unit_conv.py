# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Unit conversion
"""
#pylint: disable=too-many-arguments,invalid-name,unused-argument
#pylint: disable=missing-docstring

import numpy as np

from .dxl_exceptions import UnitConversionNotImplemented


class UnitConversion(object):
    """Base class for converting values to physics metrics.

    Attributes:
        y_min: A float representing lower limit of the converted value
        y_max: A float representing upper limit of the converted value
    """

    def __init__(self):
        self.y_min = None
        self.y_max = None

    def set_x_lim(self, x_lim):
        pass

    @staticmethod
    def fwd(x):
        raise UnitConversionNotImplemented()

    @staticmethod
    def inv(y):
        raise UnitConversionNotImplemented()


class Affine(UnitConversion):
    """Class representing affine transformations to the raw register values."""

    def __init__(self, y_min, y_max):
        """See the base class for arguments description"""
        UnitConversion.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = None
        self.x_max = None
        self.coefs = None

    def set_x_lim(self, x_lim):
        self.x_min, self.x_max = x_lim
        Y = float(self.y_max - self.y_min)
        X = float(self.x_max - self.x_min)
        a = Y / X
        b = -self.x_min * Y / X + self.y_min
        self.coefs = float(a), float(b)

    def fwd(self, x):
        a, b = self.coefs
        y = a * x + b
        return y

    def inv(self, y):
        assert self.y_min <= y <= self.y_max, (
            self.y_min, y, self.y_max)
        a, b = self.coefs
        x = int(round((float(y) - b) / a))
        return x


class AngleVelRange(UnitConversion):
    """Class provides methods for velocity convertion.

    Velocity registers on AX12 and AX18 servos encode present_speed.
    """

    def __init__(self, y_max):
        """See the base class for arguments description"""
        UnitConversion.__init__(self)
        # Degrees per second
        self.y_min = -y_max
        self.y_max = y_max
        self.x_min = None
        self.x_max = None

    def set_x_lim(self, x_lim):
        assert x_lim == (0, 1024 + 1023)
        self.x_min, self.x_max = x_lim

    def fwd(self, x):
        if x >= 1024:
            a = self.y_min
            xx = x - 1024
        else:
            a = self.y_max
            xx = x
        return a * xx / 1023.

    def inv(self, y):
        #
        # The inverse of this UnitConversion would normally be required
        # except that it is used to simulate the dynamics of a dynamixel
        # in unit tests and simulations.
        #
        # y = a * xx / 1023.
        assert -self.y_max <= y <= self.y_max
        if y < 0:
            # int or round?
            return 1024 + int(1023. * abs(y) / self.y_max)
        else:
            return int(1023. * y / self.y_max)


class SignedPercentRange(UnitConversion):
    """Description //TODO"""
    # This is the encoding used for present_load

    def __init__(self):
        UnitConversion.__init__(self)
        self.y_min = -100.
        self.y_max = 100.
        self.x_min = None
        self.x_max = None

    def set_x_lim(self, x_lim):
        assert x_lim == (0, 1024 + 1023)
        self.x_min, self.x_max = x_lim

    def fwd(self, x):
        x -= 1023
        return self.y_max * x / 1023.


class BaudConversion(UnitConversion):
    """Description //TODO"""
    table = {1000000: 1,
             500000: 3,
             400000: 4,
             250000: 7,
             200000: 9,
             117647: 16,  # 115200
             57124: 34,   # 57600
             19230: 103,  # 19200
             9615: 207,   # 9600
             9569: 208}   # 9600

    def __init__(self):
        UnitConversion.__init__(self)
        self.y_min = 0
        self.y_max = 10 * 1000 * 1000
        self.x_min = None
        self.x_max = None

    def set_x_lim(self, x_lim):
        assert x_lim == (0, 255)
        self.x_min, self.x_max = x_lim

    def fwd(self, x):
        # x is sensor reading
        for baud, code in self.table.items():
            if code == x:
                break
        else:
            raise ValueError(x)
        return baud  # pylint: disable=undefined-loop-variable

    def inv(self, y):
        return self.table[y]


class CurrentConversion(UnitConversion):
    """Description //TODO

    Unit in mA
    """
    def __init__(self):
        UnitConversion.__init__(self)
        self.unit = 4.5
        self.x_min = 0
        self.x_max = 4095
        self.y_min = self.fwd(self.x_min)
        self.y_max = self.fwd(self.x_max)

    def fwd(self, x):
        return self.unit*(x - 2048)

    def inv(self, y):
        raise NotImplementedError


class GoalTorqueConversion(UnitConversion):
    """ Description //TODO

    Unit in mA
    """

    def __init__(self):
        UnitConversion.__init__(self)
        self.unit = 4.5
        self.x_min = 0
        self.x_max = 2047
        self.y_min = self.fwd(2047)
        self.y_max = self.fwd(1023)
        a = 0

    def fwd(self, x):
        sign = +1 if x < 1024 else -1
        value = x % 1024
        return sign * value * self.unit

    def inv(self, y):
        offset = 1024 if np.sign(y) == -1 else 0
        value = int(np.round(np.abs(y)/self.unit))
        return value + offset


class ReturnTimeDelayConversion(UnitConversion):
    """Description //TODO

    Returns time delay register."""
    # table maps microseconds -> int code
    table = dict((2 * ii, ii) for ii in range(0, 254))

    def __init__(self):
        UnitConversion.__init__(self)
        self.y_min = min(self.table)
        self.y_max = max(self.table)
        self.x_min = None
        self.x_max = None

    def set_x_lim(self, x_lim):
        assert x_lim == (0, 255)
        self.x_min, self.x_max = x_lim

    def fwd(self, x):
        # x is sensor reading
        for us, code in self.table.items():
            if code == x:
                break
        else:
            raise ValueError(x)
        return us  # pylint: disable=undefined-loop-variable

    def inv(self, us):
        return self.table[us]


class DriveModeConversion(UnitConversion):
    """Description //TODO"""
    table = {'normal, master': 0,
             'reverse, master': 1,
             'normal, slave': 2,
             'reverse, slave': 3}

    def __init__(self):
        UnitConversion.__init__(self)
        self.y_min = 0
        self.y_max = 255

    def set_x_lim(self, x_lim):
        assert x_lim == (0, 255)

    def fwd(self, x):
        # x is sensor reading
        for name, code in self.table.items():
            if code == x:
                break
        else:
            raise ValueError(x)
        return name  # pylint: disable=undefined-loop-variable

    def inv(self, name):
        return self.table[name]


class BooleanFlag(UnitConversion):
    """Represents unit conversion for boolean flags (e.g. torque_enable)."""

    def __init__(self):
        UnitConversion.__init__(self)

    @staticmethod
    def fwd(x):
        return x

    @staticmethod
    def inv(y):
        return y


class ComplianceSlope(UnitConversion):
    """Represents unit conversion for slope compliance registers."""

    def __init__(self):
        UnitConversion.__init__(self)

    @staticmethod
    def fwd(x):
        return x

    @staticmethod
    def inv(y):
        thresh = 128
        while thresh > 2:
            if y >= thresh:
                return thresh
            thresh >>= 1
        return thresh


class RawValue(UnitConversion):
    """Represents dummy unit conversion object that is a pass-through."""

    @staticmethod
    def fwd(x):
        return x

    @staticmethod
    def inv(y):
        return y
