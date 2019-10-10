#!/usr/bin/env python

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from senseact.devices.dxl import dxl_mx64
from senseact.devices.dxl.dxl_utils import *


def read_time(port, idn):
    """ Read the entire control table of the DXL MX-64AT device 'N' times and plot the mean & percentile time taken. """
    read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)
    times = []
    for i in range(1000):
        t1 = time.time()
        read(port, idn, read_block)
        times.append(time.time() - t1)

    print(np.mean(times))
    print(np.percentile(times, 99))
    plt.figure()
    plt.plot(times)
    plt.show()


def random_torque(port, idn):
    """ Read the entire control table and randomly sampled torque commands to the DXL.

    This is done 'N' times and timed. Relevant data is plotted.
    """
    write_torque_mode_enable(port, idn, 1)
    read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)
    times = []
    vals_dict = {'present_pos': 2 * pi / 3.0, 'current': 0}
    actions = []
    currents = []
    for i in range(1000):
        t1 = time.time()
        if vals_dict['present_pos'] < pi / 3.0:
            # write_torque_mode_enable(0)
            # write_pos(100)
            action = 1000
            write_torque(port, idn, action)
            time.sleep(0.001)
            # write_torque_mode_enable(1)
        elif vals_dict['present_pos'] > pi:
            # write_torque_mode_enable(0)
            # write_pos(120)
            action = -1000
            write_torque(port, idn, action)
            time.sleep(0.001)
            # write_torque_mode_enable(1)
        else:
            action = int(np.random.uniform(-1, 1) * 1000)
        # action = 0 #1000
        write_torque(port, idn, action)
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
    # print(np.corrcoef(actions[:-1], currents[1:])[0, 1])
    plt.figure()
    plt.plot(np.cumsum(times), actions, label='actions')
    plt.plot(np.cumsum(times), currents, label='currents')
    plt.legend()
    plt.figure()
    plt.plot((times))
    plt.show()


def sync_write_test(port, dxl_ids):
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
                print(id, vals_dict['goal_pos'], vals_dict['present_pos'])


def bulk_read_test(port, dxl_ids):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default=1, type=int)
    parser.add_argument("--id", default=1, type=int)
    args = parser.parse_args()
    # Enable torque and move the actuator for 0.5s
    port = make_connection()
    write_torque_mode_enable(1);
    write_torque(-200);
    time.sleep(0.5);
    write_torque(0)
