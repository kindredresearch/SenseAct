# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Exceptions raised by DXL device interface code."""


class NoSuchServo(Exception):
    """No such servo with given ID on this USB."""


class CommError(Exception):
    """An unexpected communication error occurred."""


class UnitConversionNotImplemented(Exception):
    """Unit Conversion not set for given registers."""


class MalformedStatus(Exception):
    """Internal Exception for bus retry."""
