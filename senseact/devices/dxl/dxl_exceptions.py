"""Exceptions raised by DXL device interface code."""


class NoSuchServo(Exception):
    """No such servo with given ID on this USB."""


class CommError(Exception):
    """An unexpected communication error occurred."""


class UnitConversionNotImplemented(Exception):
    """Unit Conversion not set for given registers."""


class MalformedStatus(Exception):
    """Internal Exception for bus retry."""
