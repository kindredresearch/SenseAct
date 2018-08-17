"""
Constants for Dynamixel CommV1 protocol
Protocol 1.0 reference - http://support.robotis.com/en/product/actuator/dynamixel/dxl_communication.htm
"""

# pylint: disable=too-few-public-methods

import sys
import glob
import time
import serial
from senseact.devices.dxl.utils import Instructions, MalformedStatus
import numpy as np
from sys import platform

def make_connection(baudrate, timeout, port_str='None'):
    """Establishes a serial connection with a dxl device.

    Args:
        baudrate: an integer representing a baudrate to connect at
        timeout: a float representing connection timeout parameter in seconds

    Returns:
        An instance of Serial (i.e., an open serial port)
    """
    if port_str == 'None':
        if platform == 'darwin':
            port_str = glob.glob('/dev/tty.usb*')[0]
        elif platform == 'linux' or platform == 'linux2':
            port_str = glob.glob('/dev/ttyACM*')[0]
        else:
            print("Unrecognized platform: ", platform)
            sys.exit(1)

    return serial.Serial(port=port_str, baudrate=baudrate, timeout=timeout)


def read_a_block(port, idn, read_block, read_wait_time):
    """Reads a block of sensor values from dxl device.

    Args:
        port: An instance of Serial
        idn: An integer representing the DXL ID number
        read_block: An instance of Contiguous Registers (defined in dxl_reg) containing the block of registers to read
        read_wait_time: A float representing time (in seconds) to wait before reading the buffer

    Returns:
        A dictionary containing register names and their corresponding values
    """
    vals = read_a_block_vals(port, idn, read_block, read_wait_time)
    return {reg.name: val for reg, val in zip(read_block, vals)}


def read_a_block_vals(port, idn, read_block, read_wait_time):
    """ Reads a block of sensor values from dxl device.

    Args:
        port: An instance of Serial
        idn: An integer representing the DXL ID number
        read_block: An instance of Contiguous Registers (defined in dxl_reg) containing the block of registers to read
        read_wait_time: A float representing time (in seconds) to wait before reading the buffer

    Returns:
        A list containing sensor values of each register in the read_block
    """
    packet = packet_read(idn, read_block.offset, read_block.width)
    reply = loop_until_written(port, idn, packet, read_wait_time)
    return read_block.vals_from_data(reply.data)


def as_instruction_packet(dxl_id, instruction, *params):
    """Constructs instruction packet for sending a command.

    Args:
        dxl_id: An integer representing the DXL ID number
        instruction: Hex code representing instruction types for DXL command packet (e.g., Read, Write, Ping, etc.)
        *params: Depending on the instruction, start address of data to be read, length of data, data to write, etc.

    Returns:
        A bytearray representing the instruction packet
    """
    packet = bytearray((0xff, 0xff, dxl_id, len(params) + 2, instruction))
    packet.extend(params)
    packet.append(255 - (sum(packet) + 2) % 256)
    return packet


def packet_read(dxl_id, reg0, num_regs):
    """Create an instruction packet to read data from the DXL control table. 
    Args:
        dxl_id: An integer representing the DXL ID number
        reg0: An integer representing the register index in the control table
        num_regs: An integer representing the number of registers to read from the control table starting at reg0

    Returns:
        A bytearray representing the read data instruction packet
    """
    return as_instruction_packet(dxl_id, Instructions.ReadData, reg0, num_regs)


def read_status_reply(handle, expected_id):
    """Read a status reply packet from serial device.

    Args:
        handle: An instance of Serial
        expected_id: An integer representing target DXL ID number

    Returns:

    """
    raw4 = handle.read(4)
    n_read = len(raw4)
    if n_read < 4:
        raise MalformedStatus('wrong header bytes', raw4)
    header_0, header_1, motor_id, payload_len = raw4
    if motor_id != expected_id:
        raise MalformedStatus('wrong motor')
    if header_0 != 0xff or header_1 != 0xff:
        raise MalformedStatus('wrong header bytes', raw4)
    if payload_len == 0:
        raise MalformedStatus('invalid payload length (0)')
    raw_payload = handle.read(payload_len)
    n_read = len(raw_payload)
    if n_read != payload_len:
        raise MalformedStatus('short payload', raw_payload)
    chksum = 255 - ((sum(raw_payload[:-1]) + motor_id + payload_len) % 256)
    if chksum != raw_payload[-1]:
        raise MalformedStatus('chksum fail')
    status = Status(raw_payload[0], raw_payload[1:-1])
    status.dxl_id = motor_id
    return status


def loop_until_written(port, dxl_id, packet, read_wait_time=0.0001):
    """Loop until instruction packet is written in the DXL control table

    Args:
        port: An instance of Serial
        idn: An integer representing the DXL ID number
        packet: A bytearray representing the instruction packet
        read_wait_time: A float representing time (in seconds) to wait before reading the buffer
    """
    while True:
        port.write(packet)
        time.sleep(read_wait_time)
        try:
            reply = read_status_reply(port, dxl_id)
            return reply
        except MalformedStatus as e:
            print('MalformedStatus: ', e)
            time.sleep(0.05)
            port.reset_input_buffer()
            port.reset_output_buffer()
            time.sleep(0.05)


def packet_write(dxl_id, reg0, *register_values):
    """Create an instruction packet to write data in the DXL control table.
    Args:
        dxl_id: An integer representing the DXL ID number
        reg0: An integer representing the register index in the control table
        *register_values: A list containing integer values to be written to the DXL control table

    Returns:
        A bytearray representing the write data instruction packet
    """
    return as_instruction_packet(
        dxl_id, Instructions.WriteData, reg0, *register_values)


def packet_write_buffer(dxl_id, block, buf):
    """Return a write packet from a payload buffer.

    Args:
        dxl_id: An integer representing the DXL ID number
        block: A block of contiguous registers
        buf: A list containing integer values to be written to the DXL control table

    Returns:
        A bytearray representing the write data instruction packet
    """
    reg0 = block.offset
    packet = bytearray(
        (0xff, 0xff, dxl_id, len(buf) + 3, Instructions.WriteData, reg0))
    packet.extend(buf)
    packet.append(255 - (sum(packet) + 2) % 256)
    return packet


def packets_sync_write(block, data_rows):
    """Returns list of packets that will perform a SYNC_WRITE.

    Args:
        reg0: An integer specifying first register to write (on every dynamixel)
        data_rows: A numpy array (M, N + 1) of uint type.
            Contains parameter values. Each row is the parameters for one
            dynamixel. The first number in each row is the dynamixel ID.
            The remaining numbers in each row are the N parameters to write to
            the memory from reg0 to reg0 + N-1 (inclusive).
            # id0 id0-reg0 id0-reg1 id0-reg{N-1}
            # id1 id1-reg0 id1-reg1 id1-reg{N-1}

    Returns:
        List of packets (numpy uint8 vectors) to write to serial bus.

    """
    #pylint: disable=invalid-name
    reg0 = block.offset
    n_dxls, n_data_per_dxl = data_rows.shape
    L = data_rows.size + 4
    if 4 + L > 140:  # packet is too long, divide and conquer
        assert n_dxls >= 2
        rowsA = data_rows[:n_dxls / 2]
        rowsB = data_rows[n_dxls / 2:]
        rvalA = packets_sync_write(reg0, rowsA)
        rvalB = packets_sync_write(reg0, rowsB)
        return rvalA + rvalB

    # command will fit into one packet

    data = np.zeros(4 + L, dtype=np.uint8)
    # FF FF FE len instr reg0 N
    # id0 id0-reg0 id0-reg1 id0-reg{N-1}
    # id1 id1-reg0 id1-reg1 id1-reg{N-1}
    # ...
    # chksum
    data[0:7] = [0xFF, 0xFF, 0xFE, L,
                 Instructions.SyncWrite,
                 reg0, n_data_per_dxl - 1]
    data[7:-1].reshape(data_rows.shape)[:] = data_rows
    data[-1] = np.bitwise_not(data[2:-1].sum(dtype=np.uint8))
    return [data]

def clear_port(port):
    """Clears device connection port."""
    return

def close(port):
    """Closes device connection port."""
    return


class ErrorBitsClass(object):
    """ The ErrorBits class represents syntactic support. After an instruction packet is sent to the DXl, it
    responds with a status packet. The error bits in the status packet can indicate voltage error, angle limit error,
    overload error, etc.
     
     The usage is the following:
        ErrorBits['overheat'] -> int with one bit set
        ErrorBits['overheat', 'overload'] -> int with two bits set

    """
    _bits = {
        'input_voltage': 1 << 0,
        'angle_limit': 1 << 1,
        'overheat': 1 << 2,
        'range': 1 << 3,
        'chksum': 1 << 4,
        'overload': 1 << 5,
        'instruction': 1 << 6}

    def items(self):
        """ Container view of ErrorBits """
        return self._bits.items()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._bits[item]
        if isinstance(item, (list, tuple)):
            rval = 0
            for item_i in item:
                rval |= self._bits[item_i]
            return rval


ERROR_BITS = ErrorBitsClass()


class Status(object):

    """Class representing status packets coming back from Dynamixel."""

    def __init__(self, error_byte, data):
        """Inits Status class objects with corresponding data.
        
        Args:
            // TODO
        """
        self.error_byte = error_byte
        self.data = data

    def __str__(self):
        alarms = ', '.join(
            [name
             for name, bit in sorted(ERROR_BITS.items())
             if self.error_byte & bit])

        if alarms:
            return 'Status{%s, data=%s}' % (alarms, self.data)
        else:
            return 'Status{"ok", data=%s}' % (self.data)

    # pylint: disable=missing-docstring

    @property
    def input_voltage(self):        
        return self.error_byte & ERROR_BITS['input_voltage']

    @property
    def angle_limit(self):
        return self.error_byte & ERROR_BITS['angle_limit']

    @property
    def overheat(self):
        return self.error_byte & ERROR_BITS['overheat']

    @property
    def range(self):
        return self.error_byte & ERROR_BITS['range']

    @property
    def chksum(self):
        return self.error_byte & ERROR_BITS['chksum']

    @property
    def overload(self):
        return self.error_byte & ERROR_BITS['overload']

    @property
    def instruction(self):
        return self.error_byte & ERROR_BITS['instruction']
