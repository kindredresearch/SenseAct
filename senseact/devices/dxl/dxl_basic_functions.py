# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
A bunch of functional tests useful to test and debug DXL MX64 series servos
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from senseact.devices.dxl import dxl_mx64

"""
Choose device driver, id(s) and baudrate for running the tests
"""
# Use this flag to choose between ctypes based DynamixelSDK or a custom pyserial DXL driver.
use_ctypes_driver = True
baudrate = 1000000
idn = 9

if use_ctypes_driver:
    from senseact.devices.dxl import dxl_driver_v1 as dxl_commv1
else:
    from senseact.devices.dxl import dxl_commv1



port = dxl_commv1.make_connection(baudrate=baudrate, timeout=5)


def read_time():
    """ Read the entire control table of the DXL MX-64AT device 'N' times and plot the mean & percentile time taken. """
    read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)
    times = []
    for i in range(1000):
        t1 = time.time()
        read(read_block)
        times.append(time.time() - t1)

    print(np.mean(times))
    print(np.percentile(times, 99))
    plt.figure()
    plt.plot(times)
    plt.show()


def read(read_block):
    """ Read a given read block and print the relevant registers.

    Args:
        read_block: An instance of Contiguous Registers (defined in dxl_reg) containing the register to read from

    Returns:
        A dict containing the register names in the read block and corresponding values
    """
    vals_dict = dxl_commv1.read_a_block(port, idn, read_block, read_wait_time=0.0001)
    return vals_dict


def write_angle_limit(angle_low, angle_high):
    """ Sets the cw_angle_limit (clockwise) and ccw_angle_limit (counter clockwise)

    The angles limits are set by writing to the corresponding registers. This chooses
    wheel or joint mode for the dxl servo. Only wheel mode is available in torque control mode.

    Args:
        angle_low: A float. Angle in radians
        angle_high: A float. Angle in radians
    """
    block = dxl_mx64.MX64.subblock('angle_limit_ccw', 'angle_limit_cw')
    packet = dxl_commv1.packet_write_buffer(idn, block, block.data_from_vals([angle_low, angle_high]))
    dxl_commv1.loop_until_written(port, idn, packet)


def write_pos(angle):
    """ Writes to goal position register in the dxl control table. Works only in position control mode.

    Args:
        angle: A float. Angle in radians
    """
    block = dxl_mx64.MX64.subblock('goal_pos', 'goal_pos', ret_dxl_type=use_ctypes_driver)
    packet = dxl_commv1.packet_write_buffer(idn, block, block.data_from_vals([angle]))
    dxl_commv1.loop_until_written(port, idn, packet)


def write_torque_mode_enable(enable, id = 0):
    """ Enables/Disables torque control mode.

    0 - Disabled - Position control mode
    1 - Enabled - Torque control mode

    Args:
        enable: 0 or 1
        id: An integer representing the DXL ID number
    """
    if id == 0:
        id = idn
    block = dxl_mx64.MX64.subblock('torque_control_mode_enable', 'torque_control_mode_enable',
                                   ret_dxl_type=use_ctypes_driver)
    packet = dxl_commv1.packet_write_buffer(id, block, block.data_from_vals([enable]))
    dxl_commv1.loop_until_written(port, id, packet)

def write_torque_enable(enable, id = 0):
    """ Enables/Disables torque

    Args:
        enable: 0 or 1
        id: An integer representing the DXL ID number
    """
    if id == 0:
        id = idn
    block = dxl_mx64.MX64.subblock('torque_enable', 'torque_enable', ret_dxl_type=use_ctypes_driver)
    packet = dxl_commv1.packet_write_buffer(id, block, block.data_from_vals([enable]))
    dxl_commv1.loop_until_written(port, id, packet)

def write_torque(current):
    """ Write a value to goal torque register.

    Torque control is done by controlling the current in DXL. This command is valid only for select DXL models.

    Args:
        current: A float between -1024 and 1024
    """
    block = dxl_mx64.MX64.subblock('goal_torque', 'goal_torque', ret_dxl_type=use_ctypes_driver)
    packet = dxl_commv1.packet_write_buffer(idn, block, block.data_from_vals([current]))
    dxl_commv1.loop_until_written(port, idn, packet)

def write_to_register(reg_name, val):
    """ General function to write a value to any given register

    Args:
        reg_name: A string containing the name of the register to write to
        val: An int or a float depending on the register
    """
    block = dxl_mx64.MX64.subblock(reg_name, reg_name, ret_dxl_type=use_ctypes_driver)
    packet = dxl_commv1.packet_write_buffer(idn, block, block.data_from_vals([val]))
    dxl_commv1.loop_until_written(port, idn, packet)

def set_wheel_mode():
    """ Sets the DXL to wheel mode (i.e., infinite turns)

    Wheel mode can be initialized by setting angle_limit_cw and
    angle_limit_ccw registers to zero in the control table
    """
    write_to_register('angle_limit_cw', -pi)
    write_to_register('angle_limit_ccw', -pi)

def random_torque():
    """ Read the entire control table and randomly sampled torque commands to the DXL.

    This is done 'N' times and timed. Relevant data is plotted.
    """
    write_torque_mode_enable(1)
    read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)
    times = []
    vals_dict = {'present_pos': 2 * pi/3.0, 'current': 0}
    actions = []
    currents = []
    for i in range(1000):
        t1 = time.time()
        if vals_dict['present_pos'] < pi/3.0:
            # write_torque_mode_enable(0)
            # write_pos(100)
            action = 1000
            write_torque(action)
            time.sleep(0.001)
            # write_torque_mode_enable(1)
        elif vals_dict['present_pos'] > pi:
            # write_torque_mode_enable(0)
            # write_pos(120)
            action = -1000
            write_torque(action)
            time.sleep(0.001)
            # write_torque_mode_enable(1)
        else:
            action = int(np.random.uniform(-1, 1)*1000)
        #action = 0 #1000
        write_torque(action)
        # time.sleep(0.001)
        # print("action: ", action)
        # print("pos: ", vals_dict['present_pos'])
        # print("current: ", vals_dict['current'])
        vals_dict = read(read_block)
        actions.append(action)
        currents.append(vals_dict['current'])
        times.append(time.time() - t1)
    write_torque(0)
    print(np.mean(times))
    print(currents[:10])
    plt.xcorr(currents, actions)
    #print(np.corrcoef(actions[:-1], currents[1:])[0, 1])
    plt.figure()
    plt.plot(np.cumsum(times), actions, label='actions')
    plt.plot(np.cumsum(times), currents, label='currents')
    plt.legend()
    plt.figure()
    plt.plot((times))
    plt.show()

def sync_write_test(dxl_ids):
    """ Performs synchronized write operations to a target register in multiple DXLs.

    NOTE: Valid only on select DXL models.

    Args:
        dxl_ids: A list of ints containing DXL id numbers
    """
    if use_ctypes_driver:
        speeds = [300] * len(dxl_ids)
        goals = [2000] * len(dxl_ids)
        speed_block = dxl_mx64.MX64.subblock('moving_speed', 'moving_speed', ret_dxl_type=use_ctypes_driver)

        goal_block = dxl_mx64.MX64.subblock('goal_pos', 'goal_pos', ret_dxl_type=use_ctypes_driver)

        read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)

        data = zip(dxl_ids, goals)
        dxl_commv1.sync_write(port, goal_block, data)

        for i in range(100):
            for id in dxl_ids:
                vals_dict = dxl_commv1.read_a_block(port, id, read_block, read_wait_time=0.0001)
                print(id, vals_dict['goal_pos'], vals_dict['present_pos'] )


def bulk_read_test(dxl_ids):
    """ Test bulk read operations.

    Valid only on select DXL models. While not synchronized, it saves time by writing just one
    read command for multiple motors and reads the contiguous block of registers.

    Args:
        dxl_ids: A list of ints containing DXL id numbers
    """
    if use_ctypes_driver:
        goal_block = dxl_mx64.MX64.subblock('goal_pos', 'goal_pos', ret_dxl_type=use_ctypes_driver)

        pos_block = dxl_mx64.MX64.subblock('present_pos', 'present_pos', ret_dxl_type=use_ctypes_driver)

        vals_dict = dxl_commv1.bulk_read(port, [goal_block, pos_block], dxl_ids)
        print(vals_dict)


if __name__ == '__main__':
    # Enable torque and move the actuator for 0.5s
    write_torque_mode_enable(1); write_torque(-200); time.sleep(0.5); write_torque(0)
