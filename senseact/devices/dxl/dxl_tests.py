#!/usr/bin/env python

import argparse
import time

from math import pi
import matplotlib.pyplot as plt
import numpy as np

import senseact.devices.dxl.dxl_utils as dxl
import senseact.devices.dxl.dxl_mx64 as dxl_mx64


def read_time(driver, port, idn):
    """ Read the entire control table of the DXL MX-64AT device 'N' times and plot the mean & percentile time taken. """
    times = []
    for i in range(1000):
        t1 = time.time()
        dxl.read_vals(driver, port, idn)
        times.append(time.time() - t1)

    print(np.mean(times))
    print(np.percentile(times, 99))
    plt.figure()
    plt.plot(times)
    plt.show()


def random_torque(driver, port, idn):
    """ Read the entire control table and randomly sampled torque commands to the DXL.

    This is done 'N' times and timed. Relevant data is plotted.
    """
    dxl.write_torque_mode_enable(driver, port, idn, 1)

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
            dxl.write_torque(driver, port, idn, action)
            time.sleep(0.001)
            # write_torque_mode_enable(1)
        elif vals_dict['present_pos'] > pi:
            # write_torque_mode_enable(0)
            # write_pos(120)
            action = -1000
            dxl.write_torque(driver, port, idn, action)
            time.sleep(0.001)
            # write_torque_mode_enable(1)
        else:
            action = int(np.random.uniform(-1, 1) * 1000)
        # action = 0 #1000
        dxl.write_torque(driver, port, idn, action)
        # time.sleep(0.001)
        # print("action: ", action)
        # print("pos: ", vals_dict['present_pos'])
        # print("current: ", vals_dict['current'])
        vals_dict = dxl.read_vals(driver, port, idn)
        actions.append(action)
        currents.append(vals_dict['current'])
        times.append(time.time() - t1)
    dxl.write_torque(driver, port, idn, 0)
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


def sync_write_test(driver, port, dxl_ids):
    """ Performs synchronized write operations to a target register in multiple DXLs.

    NOTE: Valid only on select DXL models.

    Args:
        dxl_ids: A list of ints containing DXL id numbers
    """
    if driver.is_ctypes_driver:
        count = len(dxl_ids)
        goals = [np.random.randint(0, 2000)] * count

        joint_mode_block = dxl_mx64.MX64.subblock('torque_control_mode_enable', 'torque_control_mode_enable',
                                                  ret_dxl_type=True)

        driver.sync_write(port, joint_mode_block, zip(dxl_ids, [0] * count))

        low_angle_limit_block = dxl_mx64.MX64.subblock('angle_limit_cw', 'angle_limit_cw', ret_dxl_type=True)
        driver.sync_write(port, low_angle_limit_block, zip(dxl_ids, [0] * count))

        high_angle_limit_block = dxl_mx64.MX64.subblock('angle_limit_ccw', 'angle_limit_ccw', ret_dxl_type=True)
        driver.sync_write(port, high_angle_limit_block, zip(dxl_ids, [2048] * count))

        speed_block = dxl_mx64.MX64.subblock('moving_speed', 'moving_speed', ret_dxl_type=True)
        driver.sync_write(port, speed_block, zip(dxl_ids, [1000] * count))

        goal_block = dxl_mx64.MX64.subblock('goal_pos', 'goal_pos', ret_dxl_type=True)

        read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=True)

        data = zip(dxl_ids, goals)
        driver.sync_write(port, goal_block, data)

        for i in range(100):
            for id in dxl_ids:
                vals_dict = driver.read_a_block(port, id, read_block, read_wait_time=0.0001)
                print(id, vals_dict['goal_pos'], vals_dict['present_pos'])


def bulk_read_test(driver, port, dxl_ids):
    """ Test bulk read operations.

    Valid only on select DXL models. While not synchronized, it saves time by writing just one
    read command for multiple motors and reads the contiguous block of registers.

    Args:
        dxl_ids: A list of ints containing DXL id numbers. Should only contain 2 ids.
    """
    if driver.is_ctypes_driver:
        assert (len(dxl_ids) == 2)
        goal_block = dxl_mx64.MX64.subblock('goal_pos', 'goal_pos', ret_dxl_type=True)

        pos_block = dxl_mx64.MX64.subblock('present_pos', 'present_pos', ret_dxl_type=True)

        vals_dict = driver.bulk_read(port, [goal_block, pos_block], dxl_ids)
        print(vals_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--id", default=1, type=int)
    parser.add_argument("--baud", default=1000000, type=int)
    parser.add_argument("--noctypes", default=False, help="Set to use the non ctypes driver",
                        action="store_true")
    args = parser.parse_args()
    # Enable torque and move the actuator for 0.5s
    idn = args.id
    driver = dxl.get_driver(not args.noctypes)
    port = dxl.make_connection(driver, baudrate=args.baud)
    dxl.write_torque_mode_enable(driver, port, idn, 1)
    dxl.write_torque(driver, port, idn, -200)
    time.sleep(5)
    dxl.write_torque(driver, port, idn, 0)
