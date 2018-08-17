from enum import IntEnum

Bits8 = (0, (1 << 8) - 1)
Bits10 = (0, (1 << 10) - 1)
Bits12 = (0, (1 << 12) - 1)

class MalformedStatus(Exception):
    """Internal Exception for bus retry"""


class Instructions(IntEnum):
    """Instruction types for dynamixel command packets."""

    Ping = 0x01
    ReadData = 0x02
    WriteData = 0x03
    RegWrite = 0x04
    Action = 0x05
    Reset = 0x06
    SyncWrite = 0x83