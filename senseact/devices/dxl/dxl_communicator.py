# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import signal

from math import pi
from threading import Lock
from senseact.sharedbuffer import SharedBuffer
from senseact.communicator import Communicator
from senseact.devices.dxl import dxl_mx64


class DXLCommunicator(Communicator):
    """ Communicator class for DXL MX-64 devices.

    This class implements Dynamixel MX-64 series specific packet communicator (uses DXL Protocol 1.0).
    Only goal torque actuator commands are currently allowed.
    """

    def __init__(self,
                 idn,
                 baudrate,
                 timeout_connection=0.5,
                 sensor_dt=0.006,
                 device_path='None',
                 use_ctypes_driver=True,
                 ):
        """ Inits DXLCommunicator class with device and task-specific parameters.

        Establishes serial connection with a dxl device. If device_path is 'None', the function searches
        for a serial port address and connects to the first device found.

        Args:
            idn: An integer representing the DXL ID number
            baudrate: An integer representing a baudrate to connect at
            timeout_connection: A float representing connection timeout parameter in seconds
            sensor_dt: A float representing the cycle time for polling sensory information and
                writing torque commands.
            device_path: A string containing the serial port address (e.g., /dev/ttyACM0 or /dev/ttyUSB0 on linux)
            use_ctypes_driver: A bool. Use this flag to choose between ctypes based DynamixelSDK or
                a custom pyserial DXL driver. By default, the ctypes driver is used.
        """
        self.idn = idn
        self.baudrate = baudrate
        self.timeout = timeout_connection
        self.sensor_dt = sensor_dt
        self.device_path = device_path
        self.dt_tol = 0.0001
        self.read_wait_time = 0.0001

        self.use_ctypes_driver = use_ctypes_driver
        self.read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)

        self.write_block = dxl_mx64.MX64.subblock('goal_torque', 'goal_torque', ret_dxl_type=use_ctypes_driver)

        sensor_args = {
            'buffer_len': SharedBuffer.DEFAULT_BUFFER_LEN,
            'array_len': self.read_block.len(),
            'array_type': 'd',
            'np_array_type': 'd'
        }

        actuator_args = {
            'buffer_len': SharedBuffer.DEFAULT_BUFFER_LEN,
            'array_len': 1,
            'array_type': 'd',
            'np_array_type': 'd'
        }

        super(DXLCommunicator, self).__init__(use_sensor=True,
                                              use_actuator=True,
                                              sensor_args=sensor_args,
                                              actuator_args=actuator_args)

        self.just_read = 0
        self.just_read_lock = Lock()
        self.port_lock = Lock()
        self.read_start_time = self.read_end_time = time.time()
        self.read_time = 0
        self.last_actuation_updated = time.time()
        self.max_actuation_time = 1
        self.torque = 0

    def run(self):
        """ Override base class run method to setup dxl driver.

        The shared object file should be loaded within the process that
        sends commands and receives information from the DXL. Hence the dxl_driver
        is initialized here. This is an issue only when using the ctypes driver.
        Regular imports work for the pyserial driver.
        """
        if self.use_ctypes_driver:
            from senseact.devices.dxl import dxl_driver_v1 as dxl_commv1
        else:
            from senseact.devices.dxl import dxl_commv1
        self.dxl_driver = dxl_commv1

        # Make connection
        self.port = dxl_commv1.make_connection(self.baudrate, self.timeout, self.device_path)

        # Prepare for torque control
        self.write_torque_mode_enable(dxl_commv1, self.port, self.idn, 1, use_ctypes_driver=self.use_ctypes_driver)

        vals = self.dxl_driver.read_a_block(self.port, self.idn, self.read_block, self.read_wait_time)

        # Set Return Delay time = 0
        if not vals["rtd"] == 0:
            self.write_return_delay_time(dxl_commv1, self.port, self.idn, 0, use_ctypes_driver=self.use_ctypes_driver)

        # Set the dynamixel to wheel mode (infinite rotations and mx-64AT encoder positions Є [0, 4095])
        if not (vals["angle_limit_cw"] == -pi and vals["angle_limit_ccw"] == -pi):
            self.set_wheel_mode()

        # Overwrite torque command written in the preceding expt in case of unsafe exit
        self.write_torque(0)

        # catching SIGINT (Ctrl+C) so that we can close dxl safely
        signal.signal(signal.SIGINT, self._close)

        super(DXLCommunicator, self).run()

    def _sensor_handler(self):
        """ Receives and stores sensory packets from DXL.

        Waits for packets to arrive from DXL through `dxl_driver.read_a_block_vals` call,
        communicator sensor cycle time, and stores the packet in `sensor_buffer`.
        """

        self._time_and_wait()
        self.port_lock.acquire()
        vals = self.dxl_driver.read_a_block_vals(self.port, self.idn, self.read_block, self.read_wait_time)
        self.port_lock.release()
        self.just_read_lock.acquire()
        self.just_read = 1
        self.just_read_lock.release()
        self.sensor_buffer.write(vals)

    def _actuator_handler(self):
        """ Sends actuation commands to DXL.

        Only torque control allowed now. We send zero torque
        (i.e., stopping movements) if there is a delay in actuation.
        """
        if time.time() - self.last_actuation_updated > self.max_actuation_time:
            self.torque = 0
        self.just_read_lock.acquire()
        if self.just_read == 1:
            self.just_read = 0
            self.just_read_lock.release()
        else:
            self.just_read_lock.release()
            time.sleep(0.0001)
            return
        if self.actuator_buffer.updated():
            self.last_actuation_updated = time.time()
            recent_actuation, _, _ = self.actuator_buffer.read_update()
            self.torque = recent_actuation[0]
        self.write_torque(self.torque)

    def write_torque(self, torque):
        """ Writes goal torque commands to the DXL control table.

        Args:
            torque: An float between [-1024 and 1024]
        """
        packet = self.dxl_driver.packet_write_buffer(self.idn,
                                                     self.write_block,
                                                     self.write_block.data_from_vals([torque]))
        self.port_lock.acquire()
        self.dxl_driver.loop_until_written(self.port, self.idn, packet)
        self.port_lock.release()

    def _time_and_wait(self):
        """ Maintains communicator cycle time.

        If sending a command takes 'x'ms, wait (sensor_dt - x)ms to maintain cycle time.
        """
        self.read_end_time = time.time()
        self.read_time = self.read_end_time - self.read_start_time
        if self.read_time > self.sensor_dt:
            print("Warning: Iteration time exceeded sensor_dt {}ms by {}ms".format(
                self.sensor_dt * 1000,
                (self.read_time - self.sensor_dt) * 1000)
            )
        else:
            time.sleep(max(0, self.sensor_dt - self.read_time - self.dt_tol))
        self.read_start_time = time.time()

    def _close(self, *args, **kwargs):
        """ Safely closes the DXL device. Disables torque before shutting down the script.

        NOTE: This method currently gets called 3 times due to interactions with gym.core.env
              the following two lines can be used for debugging if this needs to get fixed

        Args:
            *args: Additional args
            **kwargs: Additional keyword args
        """
        print("Closing Gripper communicator")

        # Wait for the sensor and actuator threads to close before setting the torque to 0
        # This is to ensure there isn't contention for sending messages on the port
        super(DXLCommunicator, self)._close()
        self.write_torque_mode_enable(self.dxl_driver, self.port, self.idn, 0,
                                      use_ctypes_driver=self.use_ctypes_driver)
        self.write_torque(0)
        self.dxl_driver.clear_port(self.port)
        self.dxl_driver.close(self.port)

    @staticmethod
    def write_torque_mode_enable(dxl_driver, port, idn, enable, use_ctypes_driver=True):
        """ Enables/Disables torque control mode in DXL MX-64 devices.

        0 - Disabled - Position control mode
        1 - Enabled - Torque control mode

        Args:
            dxl_driver: A variable that points to the DXL driver
            port: Dynamixel portHandler object
            idn: An integer representing the DXL ID number
            enable: 0 or 1
            use_ctypes_driver: A bool to choose between Ctypes and pyserial driver
        """
        block = dxl_mx64.MX64.subblock('torque_control_mode_enable', 'torque_control_mode_enable',
                                       ret_dxl_type=use_ctypes_driver)
        packet = dxl_driver.packet_write_buffer(idn, block, block.data_from_vals([enable]))
        dxl_driver.loop_until_written(port, idn, packet)

    @staticmethod
    def write_return_delay_time(dxl_driver, port, idn, rtd, use_ctypes_driver=True):
        """ Writes the return delay time to the DXL control table

        Return Delay Time is the delay time per data value that takes from the transmission of
        Instruction Packet until the return of Status Packet.

        Args:
            dxl_driver: A variable that points to the DXL driver
            port: Dynamixel portHandler object
            idn: An integer representing the DXL ID number
            rtd: A float representing delay time between packets in milliseconds.
            use_ctypes_driver: A bool to choose between Ctypes and pyserial driver
        """
        block = dxl_mx64.MX64.subblock('rtd', 'rtd', ret_dxl_type=use_ctypes_driver)
        packet = dxl_driver.packet_write_buffer(idn, block, block.data_from_vals([rtd]))
        dxl_driver.loop_until_written(port, idn, packet)

    @staticmethod
    def write_to_register(dxl_driver, port, idn, reg_name, val, use_ctypes_driver=True):
        """ Writes a value to a given register of the DXL device

        General function to write a value to any given register

        Args:
            dxl_driver:A variable that points to the DXL driver
            port: Dynamixel portHandler object
            idn: An integer representing the DXL ID number
            reg_name: A string containing the name of the register to write to
            val: An int or a float depending on the register
            use_ctypes_driver: A bool to choose between Ctypes and pyserial driver
        """
        block = dxl_mx64.MX64.subblock(reg_name, reg_name, ret_dxl_type=use_ctypes_driver)
        packet = dxl_driver.packet_write_buffer(idn, block, block.data_from_vals([val]))
        dxl_driver.loop_until_written(port, idn, packet)

    def set_wheel_mode(self):
        """ Sets the DXL to wheel mode (i.e., infinite turns)

        This is done by setting angle_limit_cw and
        angle_limit_ccw registers to zero in the control table
        """
        self.write_to_register(self.dxl_driver, self.port, self.idn, 'angle_limit_cw', -pi,
                               use_ctypes_driver=self.use_ctypes_driver)
        self.write_to_register(self.dxl_driver, self.port, self.idn, 'angle_limit_ccw', -pi,
                               use_ctypes_driver=self.use_ctypes_driver)
