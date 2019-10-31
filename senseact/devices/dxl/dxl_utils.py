# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from math import pi

import senseact.devices.dxl.dxl_commv1 as no_ctypes_driver
import senseact.devices.dxl.dxl_driver_v1 as ctypes_driver
from senseact.devices.dxl import dxl_mx64

"""
First create a driver:

driver = get_driver()

Then open a port:

port = make_connection(driver)

Then you can use the driver and port to access the different utility functions.

Operation in the different modes:

Joint mode: 

Joint mode allows you to specify a position for the motor to move to. The Dynamixel will use its internal 
PID controller to move to the target position. To use joint mode:

set_joint_mode(driver, port, 1)
write_pos(driver, port, 1, 2.0)

Boundaries can be set directly in the write_joint_mode call or in write_angle_limit.
Speed can be set directly in the write_joint_mode call or in write_speed.
The motor can be stopped by calling stop, but you will need to call write_joint_mode again to return to joint_mode.


Speed mode:

Speed mode allows you to make the motor turn continuously as a target speed. The motor will use its PID controller
to achieve thjis.

Use speed mode as follows.

set_speed_mode()
write_speed
stop

Torque mode:

Torque mode allows you to control the motor by providing a target load.

To use torque mode:

set_torque_mode
write_torque

stop

"""


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


def set_joint_mode(driver, port, idn, angle_low=-pi, angle_high=pi, speed=None):
    """
    Sets the motor into joint mode. Joint mode has fixed rotation limits.
    It appears that if goal_pos and present_pos are not aligned prior to this call that
    they will be aligned internally after this call. So it won't suddenly jump.
    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param angle_low: min is -pi (fully CW)
    :param angle_high: max is pi (fully CCW)
    """
    write_torque_mode_enable(driver, port, idn, 0)
    write_angle_limit(driver, port, idn, angle_low, angle_high)

    if speed is not None:
        write_speed(driver, port, idn, speed)


def set_speed_mode(driver, port, idn):
    """
    Sets operation to speed mode
    :param driver: Driver used. Returned by get_driver()
    :param port: Port returned by make_connection
    :param idn: Motor id
    """
    set_wheel_mode(driver, port, idn)
    write_torque_mode_enable(driver, port, idn, 0)


def set_torque_mode(driver, port, idn):
    """
    Puts the motor into torque mode
    :param driver: Driver used. Returned by get_driver()
    :param port: Port returned by make_connection
    :param idn: Motor id
    :return:
    """
    set_wheel_mode(driver, port, idn)
    write_torque_mode_enable(driver, port, idn, 1)
    write_torque_enable(driver, port, idn, 1)


def set_wheel_mode(driver, port, idn):
    """ Sets the DXL to wheel mode (i.e., infinite turns)

    Wheel mode can be initialized by setting angle_limit_cw and
    angle_limit_ccw registers to zero in the control table

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    """
    write_to_register(driver, port, idn, 'angle_limit_cw', -pi)
    write_to_register(driver, port, idn, 'angle_limit_ccw', -pi)


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


def write_speed(driver, port, idn, speed):
    """

    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor's id (integer)
    :param speed:  Degrees/sec. [-684, 684]
    :return:
    """
    write_to_register(driver, port, idn, 'moving_speed', speed)


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


def stop(driver, port, idn):
    """
    Stops the motor by setting it to wheel mode and giving it a velocity of zero.
    If a velocity of zero is given in joint mode then the motor goes full speed.
    This has the side effect that you will need to call write_joint_mode again.
    :param driver: Driver used. Returned by get_driver()
    :param port: Port return by make_connection
    :param idn: Motor id
    :return:
    """
    set_wheel_mode(driver, port, idn)
    write_speed(driver, port, idn, 0)
    write_torque(driver, port, idn, 0)
