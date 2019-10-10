# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import pi

import senseact.devices.dxl.dxl_commv1 as no_ctypes_driver
import senseact.devices.dxl.dxl_driver_v1 as ctypes_driver
from senseact.devices.dxl import dxl_mx64


def get_driver(use_ctypes_driver=True):
    if use_ctypes_driver:
        return ctypes_driver
    return no_ctypes_driver


def make_connection(driver, baudrate=1000000, timeout=5, port_str=None):
    """
    Opens a connection to the dynamixel's USB interface
    :param driver: Driver used. Returned by get_driver()
    :param baudrate: Baud rate configured on the motors of interest.
    :param timeout: Connection timeout
    :param port_str: None or "None" - will search for /dev/ttyUSB* or /dev/ttyACM* and connect to the first
    :return:
    """
    return driver.make_connection(baudrate=baudrate, timeout=timeout, port_str=port_str)


def close(driver, port):
    """
    Closes the port

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    """
    driver.close(port)


def read(driver, port, idn, read_block):
    """
    Read a given read block and print the relevant registers.

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param read_block: An instance of Contiguous Registers (defined in dxl_reg) containing the register to read from
    :return: A dict containing the register names in the read block and corresponding values
    """
    vals_dict = driver.read_a_block(port, idn, read_block, read_wait_time=0.0001)
    return vals_dict


def read_vals(driver, port, idn):
    """
    Reads the entire status block of a given motor.
    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :return: A dictionary containing register names and corresponding values
    """
    read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=driver.is_ctypes_driver)
    return read(driver, port, idn, read_block)


def read_register(driver, port, idn, reg_name):
    """
    Read a specific register from a specific motor
    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param reg_name: Name of the register.
    :return:
    """
    read_block = dxl_mx64.MX64.subblock(reg_name, reg_name, ret_dxl_type=driver.is_ctypes_driver)
    return read(driver, port, idn, read_block)


def write_to_register(driver, port, idn, reg_name, val):
    """ Writes a value to a given register of the DXL device

    General function to write a value to any given register

    :param driver: Driver used. Returned by get_driver()
    :param port: Dynamixel portHandler object
    :param idn: An integer representing the DXL ID number
    :param reg_name: A string containing the name of the register to write to
    :param val: An int or a float depending on the register
    """
    block = dxl_mx64.MX64.subblock(reg_name, reg_name, ret_dxl_type=driver.is_ctypes_driver)
    packet = driver.packet_write_buffer(idn, block, block.data_from_vals([val]))
    driver.loop_until_written(port, idn, packet)


def write_angle_limit(driver, port, idn, angle_low, angle_high):
    """
    Sets the cw_angle_limit (clockwise) and ccw_angle_limit (counter clockwise)

    The angles limits are set by writing to the corresponding registers. This chooses
    wheel or joint mode for the dxl servo. Only wheel mode is available in torque control mode.

    angle_limit_cw is negative and angle_limit_ccw is positive. In default joint mode the limits
    are [-pi, pi]

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param angle_low: A float. Angle in radians
    :param angle_high: A float. Angle in radians

    """
    write_to_register(driver, port, idn, "angle_limit_cw", angle_low)
    write_to_register(driver, port, idn, "angle_limit_ccw", angle_high)


def write_pos(driver, port, idn, angle):
    """ Writes to goal position register in the dxl control table. Works only in position control mode.

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param angle: A float. Angle in radians
    """
    write_to_register(driver, port, idn, "goal_pos", angle)


def write_torque_mode_enable(driver, port, idn, enable):
    """
    Enables/Disables torque control mode.

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param enable:  0 - Disabled - Position control mode
                    1 - Enabled - Torque control mode
    """
    write_to_register(driver, port, idn, "torque_control_mode_enable", enable)


def write_torque_enable(driver, port, idn, enable):
    """
    Enables/Disables torque
    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param enable: 0 or 1
    """
    write_to_register(driver, port, idn, 'torque_enable', enable)


def write_torque(driver, port, idn, current):
    """
    Write a value to goal torque register.

    Torque control is done by controlling the current in DXL. This command is valid only for select DXL models.

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param current: A float between -1024 and 1024
    """
    write_to_register(driver, port, idn, 'goal_torque', current)


def write_wheel_mode(driver, port, idn):
    """ Sets the DXL to wheel mode (i.e., infinite turns)

    Wheel mode can be initialized by setting angle_limit_cw and
    angle_limit_ccw registers to zero in the control table

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    """
    write_to_register(driver, port, idn, 'angle_limit_cw', -pi)
    write_to_register(driver, port, idn, 'angle_limit_ccw', -pi)


def write_id(driver, port, current_id, new_id):
    """
    Changes a motor's id number.
    :param driver: Driver used. Returned by get_driver()
    :param port: Port returned by make_connection
    :param current_id: Motor's current id
    :param new_id: Motor's new id
    :return: Nothing
    """
    write_to_register(driver, port, current_id, 'bus_id', new_id)


def write_baud(driver, port, idn, new_baud):
    """
    Set the baud on a motor. After this operation you should close the current port and open
    a new one at the new baud rate
    :param driver: Driver used. Returned by get_driver()
    :param port: Port returned by make_connection
    :param idn: Motor id
    :param new_baud: New baud value
    :return: Nothing
    """
    write_to_register(driver, port, idn, 'baud', new_baud)


def write_rtd(driver, port, idn, rtd=0):
    """
    Sets the Return Delay Time
    (http://support.robotis.com/en/product/actuator/dynamixel/mx_series/mx-64at_ar.htm#Actuator_Address_05).
    Recommend value of 0.

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor id
    :param rtd: Return Delay Time value
    :return:
    """
    write_to_register(driver, port, idn, 'rtd', rtd)
