"""
Requires DynamixelSDK repo written by Robotis- https://github.com/ROBOTIS-GIT/DynamixelSDK.
In particular, we use the commit version - "d6420db120daf9d777a2cb83e9f2ba27687504e7"
This script is modeled after the dxl_commv1 script for ease of substitution and usage. It's not elegant or optimal.
But it's needed for backward compatibility. It doesn't contain every single function available within the dxl_commv1.
But it does contain the key function calls that are required within dxl_communicator.py.
"""

import sys
import glob
import ctypes
import senseact.lib.DynamixelSDK.python.dynamixel_functions_py.dynamixel_functions as dynamixel

from sys import platform

PROTOCOL_VERSION = 1
COMM_SUCCESS     = 0


def make_connection(baudrate, timeout, port_str='None'):
    """ Establishes serial connection with a dxl device.

    If port_str is 'None', the function searches for a serial port address and
    connects to the first device found.

    Args:
        baudrate: an integer representing a baudrate to connect at
        timeout: a float representing connection timeout parameter in seconds
        port_str: A string containing the serial port address (e.g., /dev/ttyACM0 or /dev/ttyUSB0 on linux)

    Returns:
        An instance of dynamixel.portHandler defined in C
    """

    if port_str == 'None':
        if platform == 'darwin':
            port_str = glob.glob('/dev/tty.usb*')[0]
        elif platform == 'linux' or platform == 'linux2':
            port_str = glob.glob('/dev/ttyACM*')[0]
        else:
            print("Unrecognized platform: ", platform)
            sys.exit(1)

    # Initialize PortHandler Structs
    # Set the port path
    # Get methods and members of PortHandlerLinux
    port_num = dynamixel.portHandler(port_str.encode('utf-8'))

    # Initialize PacketHandler Structs
    dynamixel.packetHandler()

    # Open port
    if dynamixel.openPort(port_num):
        print("Succeeded to open the port!")
    else:
        raise IOError("Failed to open the port!")

    # Set port baudrate
    if dynamixel.setBaudRate(port_num, baudrate):
        print("Baudrate set to: {}".format(baudrate))
    else:
        raise IOError("Failed to change the baudrate!")

    # Set port timeout
    timeout = int(timeout*1000)     # Convert to milli-seconds
    if dynamixel.setPacketTimeoutMSec(port_num, timeout):
        print("Timeout set to: {}!".format(timeout))
    else:
        raise IOError("Failed to change the timeout!")

    return port_num

def read_a_block(port, idn, read_block, read_wait_time):
    """ Reads a block of sensor values from dxl device.

    Args:
        port: Dynamixel portHandler object
        idn: An integer representing the DXL ID number
        read_block: An instance of Contiguous Registers (defined in dxl_reg) containing the block of registers to read
        read_wait_time: A float representing time (in seconds) to wait before reading the buffer

    Returns:
        A dictionary containing register names and their corresponding values
    """
    vals = read_a_block_vals(port, idn, read_block, read_wait_time)
    return {reg.name: val for reg, val in zip(read_block, vals)}

def read_a_block_vals(port, idn, read_block, read_wait_time=0.00001):
    """ Reads a block of sensor values from dxl device.

    Args:
        port: Dynamixel portHandler object
        idn: An integer representing the DXL ID number
        read_block: An instance of Contiguous Registers (defined in dxl_reg) containing the block of registers to read
        read_wait_time: A float representing time (in seconds) to wait before reading the buffer

    Returns:
        A list containing sensor values of each register in the read_block
    """
    dynamixel.readTxRx(port, PROTOCOL_VERSION, idn, read_block.offset, read_block.width)

    dxl_comm_result = dynamixel.getLastTxRxResult(port, PROTOCOL_VERSION)
    dxl_error = dynamixel.getLastRxPacketError(port, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        print(dynamixel.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        print(dynamixel.getRxPacketError(PROTOCOL_VERSION, dxl_error))

    data_pos = 0
    vals = []
    for reg in read_block._regs:
        data = dynamixel.getDataRead(port, PROTOCOL_VERSION, reg.width, data_pos)
        data_pos += reg.width
        vals.append(data)
    return read_block.vals_from_data(vals)

def read2bytes(port, idn, address):
    """ Read 2 bytes from the control table of a DXL with id = idn, starting at the specified address

    Args:
        port: Dynamixel portHandler object
        idn: An integer representing the DXL ID number
        address: An integer representing the register address in the DXL control table

    Returns:
        An int or float value read from the specified address of a DXL with id = idn
    """
    return dynamixel.read2ByteTxRx(port, PROTOCOL_VERSION, idn, address)

def read1byte(port, idn, address):
    """ Read 1 byte from the control table of a DXL with id = idn, starting at the specified address

    Args:
        port: Dynamixel portHandler object
        idn: An integer representing the DXL ID number
        address: An integer representing the register address in the DXL control table

    Returns:
        An int or float value read from the specified address of a DXL with id = idn
    """
    return dynamixel.read1ByteTxRx(port, PROTOCOL_VERSION, idn, address)

def packet_write_buffer(idn, block, buf):
    """ Returns a write packet from a payload buffer.

    NOTE: The namesake function in dxl_commv1 serves a specific purpose. However, this is just a filler here.
          We use this function to fit with the old code. Helps with backward compatibility.

    Args:
        idn: An integer representing the DXL ID number
        block: A block of contiguous registers
        buf: A list containing values to be written to the control table

    Returns:
        A tuple - (address of register, list of values to be written, width of the register in bytes)
    """
    reg0 = block.offset
    width = block.width
    return (reg0, buf, width)

def packet_read(idn, reg0, width):
    """ Create an instruction packet to read data from the DXL control table.

    NOTE: The namesake function in dxl_commv1 serves a specfic purpose. However, this is just a filler here.
          We use this function to fit with the old code. Helps with backward compatibility.

    Args:
        idn: An integer representing the DXL ID number
        reg0: An integer representing the register index in the control table
        num_regs: An integer representing the number of registers to read from the control table starting at reg0

    Returns:
        A tuple - (ID of DXL device, address of register, width of the register in bytes)
    """
    return (idn, reg0, width)

def write1byte(port, idn, address, data):
    """ Write 1 byte to the DXL control table

        Args:
            port: Dynamixel portHandler object
            idn: An integer representing the DXL ID number
            address: An integer representing the register address in the DXL control table
            data: An integer. Data to be written to the register
    """
    dynamixel.write1ByteTxRx(port, PROTOCOL_VERSION, idn, address, data)

def write2bytes(port, idn, address, data):
    """ Write 2 bytes to the DXL control table

        Args:
            port: Dynamixel portHandler object
            idn: An integer representing the DXL ID number
            address: An integer representing the register address in the DXL control table
            data: An integer. Data to be written to the register
    """
    dynamixel.write2ByteTxRx(port, PROTOCOL_VERSION, idn, address, data)

def loop_until_written(port, dxl_id, packet, read_wait_time=0.00001):
    """ Loop until instruction packet is written in the DXL control table

    Args:
        port: Dynamixel portHandler object
        idn: An integer representing the DXL ID number
        packet: A tuple - (address of register, list of values to be written, width of the register in bytes)
        read_wait_time: A float representing time (in seconds) to wait before reading the buffer
    """
    reg0, buf, width = packet
    if width == 1:
        write1byte(port, dxl_id, reg0, buf[0])
    elif width == 2:
        write2bytes(port, dxl_id, reg0, buf[0])

    dxl_comm_result = dynamixel.getLastTxRxResult(port, PROTOCOL_VERSION)
    dxl_error = dynamixel.getLastRxPacketError(port, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        print(dynamixel.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        print(dynamixel.getRxPacketError(PROTOCOL_VERSION, dxl_error))

def sync_write(port, block, data):
    """ Write to multiple DXLs in synchronized fashion

    This instruction is used to control multiple Dynamixels simultaneously with a single Instruction Packet
    transmission. When this instruction is used, several instructions can be transmitted at once, so that the
    communication time is reduced when multiple Dynamixels are connected in a single channel. However, the SYNC WRITE
    instruction can only be used to a single address with an identical length of data over connected Dynamixels.
    ID should be transmitted as Broadcasting ID.

        Args:
            port: Dynamixel portHandler object
            block: An instance of Contiguous Registers (defined in dxl_reg) containing the register to write to
            data: A zip of 2 lists - dxl_ids and values.
    """

    address = block.offset
    length = block.width

    group_num = dynamixel.groupSyncWrite(port, PROTOCOL_VERSION, address, length)

    for ind, (dxl_id, value) in enumerate(data):
        dxl_addparam_result = ctypes.c_ubyte(dynamixel.groupSyncWriteAddParam(group_num, dxl_id, value, length)).value
        if dxl_addparam_result != 1:
            print(dxl_addparam_result)
            print("[ID:%03d] groupSyncWrite addparam failed" % (dxl_id))

    # Syncwrite goal position
    dynamixel.groupSyncWriteTxPacket(group_num)
    dxl_comm_result = dynamixel.getLastTxRxResult(port, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        print(dynamixel.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))

    # Clear syncwrite parameter storage
    dynamixel.groupSyncWriteClearParam(group_num)

def init_bulk_read(port):
    """ Initialize groupBulkRead Structs

        Args:
            port: Dynamixel portHandler object

        Returns:
            An instance of dynamixel.groupBulkRead defined in C
    """
    group_num = dynamixel.groupBulkRead(port, PROTOCOL_VERSION)
    return group_num

def bulk_read(port, blocks, dxl_ids, group_num=None):
    """ Read from multiple DXL MX-64s sending one bulk read packet

    This instruction is used for reading values of multiple MX series DXLs simultaneously by sending a single
    Instruction Packet. The packet length is shortened compared to sending multiple READ commands, and the idle time
    between the status packets being returned is also shortened to save communication time. However, this cannot be
    used to read a single module. If an identical ID is designated multiple times, only the first designated
    parameter will be processed.

        Args:
            port: Dynamixel portHandler object
            blocks: A list containing blocks of contiguous registers
            dxl_ids: A list containing DXL id numbers
            group_num: An instance of dynamixel.groupBulkRead defined in C

        Returns:
            A dictionary containing the motor id, the register names and their corresponding values.
            For e.g., if present position and goal position are read from 2 motors, the output would be:
            {'1': {'present_pos': 2.34, 'goal_pos': 3.21}, '2': {'present_pos': 1.23, 'goal_pos': 2.55}}
    """
    # Initialize Group bulk read instance
    if group_num == None:
        group_num = init_bulk_read(port)

    if not isinstance(blocks, list):
        blocks = blocks[blocks]

    if not isinstance(dxl_ids, list):
        dxl_ids = [dxl_ids]

    assert len(blocks)==len(dxl_ids)

    # Add parameter storage for Dynamixel#1 present position value
    for i, (id, block) in enumerate(zip(dxl_ids, blocks)):
        address = block.offset
        length = block.width

        dxl_addparam_result = ctypes.c_ubyte(
            dynamixel.groupBulkReadAddParam(group_num, id, address, length)).value
        if dxl_addparam_result != 1:
            print("[ID:%03d] groupBulkRead addparam failed" % (id))

    # Bulkread specified address
    dynamixel.groupBulkReadTxRxPacket(group_num)
    dxl_comm_result = dynamixel.getLastTxRxResult(port, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        print(dynamixel.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))

    # Read the values and convert them
    vals_dict = {}
    for i, (id, block) in enumerate(zip(dxl_ids, blocks)):
        address = block.offset
        length = block.width
        # Check if groupbulkread data of Dynamixel#1 is available
        dxl_getdata_result = ctypes.c_ubyte(
            dynamixel.groupBulkReadIsAvailable(group_num, id, address, length)).value
        if dxl_getdata_result != 1:
            print("[ID:%03d] groupBulkRead getdata failed" % (id))

        raw_val = dynamixel.groupBulkReadGetData(group_num, id, address, length)
        data = block.vals_from_data([raw_val])
        vals_dict[i] = data
    return vals_dict

def clear_port(port):
    """ Clears device port. """
    dynamixel.clearPort(port)

def close(port):
    """ Closes device port. """
    dynamixel.closePort(port)
