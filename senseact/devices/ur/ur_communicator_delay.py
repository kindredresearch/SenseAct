# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import socket
import time
import numpy as np
from senseact.communicator import Communicator
from senseact.devices.ur import ur_utils
from senseact.sharedbuffer import SharedBuffer
from threading import Lock


class URCommunicator(Communicator):
    """Communicator class for UR5 robot with artificial packet delays.

    This class implements UR-robot specific paket communicator.
    The packets it receives from UR-robot is of type
    `ur_utils.REALTIME_COMM_PACKET`. The actuator commands it allows
    sending currently are:
    - servoj
    - speedj
    - movej
    - movel
    - stopj
    - unlock protective stop.
    The code allows for introducing artificial sensory packet delays
    to study the effect of delays on learning algorithms.
   """

    def __init__(self, host,
                 delay=0,
                 actuation_sync_period=True,
                 disable_nagle_algorithm=True,
                 speedj_timeout=0.5,
                 buffer_len=None
                 ):
        """Inits URCommunicator class with device- and task-specific parameters.

        Args:
            host: a string specifying UR5 Controller IP address
            actuation_sync_period: a boolean specifying whther to sync sending
                actuations to UR5 with receiving sensory packets from UR5
            delay: a float specifying artificial sensory packet delay in s
            disable_nagle_algorithm: a boolean specifying a parameter in
                socket class
            speedj_timeout: a float specifying time-out on a speedj command in s
            buffer_len: an integer length of sensor and actuation buffers
        """
        self._host = host
        self._delay = delay
        self._disable_nagle_algorithm = disable_nagle_algorithm
        self._actuation_sync_period = actuation_sync_period
        self._speedj_timeout = speedj_timeout

        # The number of sensor reads since the last actuator write
        self._num_reads = 0
        self._num_read_lock = Lock()

        self._buffer_len = buffer_len
        if buffer_len is None:
            self._buffer_len = SharedBuffer.DEFAULT_BUFFER_LEN

        sensor_args = {'buffer_len': self._buffer_len,
                       'array_len': ur_utils.REALTIME_COMM_PACKET_SIZE,
                       'array_type': 'b',
                       'np_array_type': ur_utils.REALTIME_COMM_PACKET,
                       }

        actuator_args = {'buffer_len': self._buffer_len,
                         'array_len': 1 + max([x['size'] for x in ur_utils.COMMANDS.values()]),
                         'array_type': 'd',
                         'np_array_type': 'd',
                         }

        super(URCommunicator, self).__init__(use_sensor=True,
                                             use_actuator=True,
                                             sensor_args=sensor_args,
                                             actuator_args=actuator_args)

        # make connections
        self._sock = URCommunicator.make_connection(
            self._host,
            port=ur_utils.REALTIME_COMM_CLIENT_INTERFACE_PORT,
            disable_nagle_algorithm=self._disable_nagle_algorithm
        )
        self._sock.settimeout(0.2)

        self._dashboard_sock = URCommunicator.make_connection(
            host=self._host,
            port=ur_utils.DASHBOARD_SERVER_PORT,
            disable_nagle_algorithm=self._disable_nagle_algorithm
        )
        time.sleep(0.5)

        self._start_time = time.time()
        self._previous_start_time = time.time()
        self._recv_time = self._start_time
        self._prev_recv_time = self._start_time

        self._last_command = None
        self._commands_queue = []
        self._sent_times = []


        # This flag is specific to the `actuator_handle`, used to stop it from sending
        # actuation command in the event of a lost connection, which is detected and
        # re-established from `_sensor_handler`. This is the only way the sensor and
        # the actuator threads communicate.
        self._stop = False

    def _sensor_handler(self):
        """Receives and stores sensory packets from UR5.

        Waits for packets to arrive from UR-robot through `socket.recv` call,
        checks for any delay in transmission, and stores the packet in `sensor_buffer`.
        This method also handles re-establishing of a lost connection.

        Raises: IOError, ValueError - data convertion errors.

        """

        try:
            data = self._sock.recv(ur_utils.REALTIME_COMM_PACKET_SIZE)
            self._recv_time = time.time()  # time after receiving packet

            # check and parse received packet
            self.pre_check(data)
            parsed = np.frombuffer(data, dtype=ur_utils.REALTIME_COMM_PACKET)

            self.sensor_buffer.write(parsed)

            self._prev_recv_time = self._recv_time
            self._num_read_lock.acquire()
            self._num_reads += 1
            self._num_read_lock.release()
        except (IOError, ValueError) as e:
            if isinstance(e, ValueError):
                # May happen with weak wireless connection
                print('', e, "Could not convert data.")
            else:
                print('', e, ': Lost socket to UR, going into reconnect loop')
            self._stop = True
            time.sleep(ur_utils.ACTUATOR_DT * 2)  # to let _actuator_handler thread return
            self._sock.close()
            self._dashboard_sock.close()
            self._sock = self.make_connection(self._host,
                                              ur_utils.REALTIME_COMM_CLIENT_INTERFACE_PORT,
                                              disable_nagle_algorithm=self._disable_nagle_algorithm
                                              )
            self._dashboard_sock = self.make_connection(self._host,
                                                        ur_utils.DASHBOARD_SERVER_PORT,
                                                        disable_nagle_algorithm=self._disable_nagle_algorithm
                                                        )
            self._stop = False
        except Exception as e:
            import sys
            print("Unexpected error:", e, sys.exc_info()[0])
            raise

    def _actuator_handler(self):
        """Sends actuation commands to UR5.

        Waits for `actuator_buffer` being updated. If the connection
        is lost, does not do anything. Otherwise, proceeds with
        sending the command corresponding to the content of
        `actuator_buffer` with an added delay. The delay is randomly
        exponentially distributed with scale set by self._delay attribute."""

        # sleep until ready to send an actuation command
        # the ready condition depends on the `actuator_sync_period`
        while self._actuator_running:
            if self._actuation_sync_period > 0:
                # Synchronized to sensor readings, so check to see if the correct
                # number of readings have been detected to break out of the loop
                self._num_read_lock.acquire()
                if self._num_reads >= self._actuation_sync_period:
                    self._num_reads = 0
                    self._num_read_lock.release()
                    break
                self._num_read_lock.release()
            else:
                if self.actuator_buffer.updated():
                    break
            time.sleep(0.0001)

        # Do not perform an actuation if the socket connection is broken
        # or the actuator should no longer be running
        if self._stop or not self._actuator_running:
            return
        # obtain actuation command if it is ready
        # or a read was just performed and an action needs to be sent
        sent_count = 0
        for entry in self._commands_queue:
            cmd_str, timestamp, delay = entry
            now = time.time()
            if now - timestamp > delay:
                self._last_command = cmd_str
                self._sent_times.append(now)
                if not 'speedj' in str(self._last_command):
                    self._sock.send(cmd_str)
                sent_count += 1
                time.sleep(0.0001)
            else:
                break
        self._commands_queue = self._commands_queue[sent_count:]
        if not self._last_command is None and 'speedj' in str(self._last_command):
            self._sock.send(self._last_command)
        if self.actuator_buffer.updated() or self._actuation_sync_period > 0 or self._delay > 0.0:
            # keep track of the updated flag as it will be set to 0 below
            updated = self.actuator_buffer.updated()
            self._num_read_lock.acquire()
            self._num_reads = 0
            self._num_read_lock.release()
            recent_actuation, time_stamp, _ = self.actuator_buffer.read_update()
            recent_actuation = recent_actuation[0]
            if recent_actuation[0] == ur_utils.COMMANDS['SERVOJ']['id']:
                servoj_values = ur_utils.COMMANDS['SERVOJ']
                cmd = ur_utils.ServoJ(
                    q=recent_actuation[1:1 + servoj_values['size'] - 3],
                    t=servoj_values['default']['t']
                    if recent_actuation[-3] == ur_utils.USE_DEFAULT else recent_actuation[-3],
                    lookahead_time=servoj_values['default']['lookahead_time']
                    if recent_actuation[-2] == ur_utils.USE_DEFAULT else recent_actuation[-2],
                    gain=servoj_values['default']['gain']
                    if recent_actuation[-1] == ur_utils.USE_DEFAULT else recent_actuation[-1]
                )
            elif recent_actuation[0] == ur_utils.COMMANDS['SPEEDJ']['id']:
                if time.time() - time_stamp[-1] > self._speedj_timeout:
                    return
                speedj_values = ur_utils.COMMANDS['SPEEDJ']
                cmd = ur_utils.SpeedJ(
                    qd=recent_actuation[1:1 + speedj_values['size'] - 2],
                    a=speedj_values['default']['a']
                    if recent_actuation[-2] == ur_utils.USE_DEFAULT else recent_actuation[-2],
                    t_min=speedj_values['default']['t_min']
                    if recent_actuation[-1] == ur_utils.USE_DEFAULT else recent_actuation[-1],
                )
            elif not updated:
                # The commands below this point should only be executed
                # if they are new actuations
                return
            elif recent_actuation[0] == ur_utils.COMMANDS['MOVEL']['id']:
                movel_values = ur_utils.COMMANDS['MOVEL']
                cmd = ur_utils.MoveL(
                    pose=recent_actuation[1:1 + movel_values['size'] - 4],
                    a=movel_values['default']['a']
                    if recent_actuation[-4] == ur_utils.USE_DEFAULT else recent_actuation[-4],
                    v=movel_values['default']['v']
                    if recent_actuation[-3] == ur_utils.USE_DEFAULT else recent_actuation[-3],
                    t=movel_values['default']['t']
                    if recent_actuation[-2] == ur_utils.USE_DEFAULT else recent_actuation[-2],
                    r=movel_values['default']['r']
                    if recent_actuation[-1] == ur_utils.USE_DEFAULT else recent_actuation[-1],
                )
            elif recent_actuation[0] == ur_utils.COMMANDS['MOVEJ']['id']:
                movej_values = ur_utils.COMMANDS['MOVEJ']
                cmd = ur_utils.MoveJ(
                    q=recent_actuation[1:1 + movej_values['size'] - 4],
                    a=movej_values['default']['a']
                    if recent_actuation[-4] == ur_utils.USE_DEFAULT else recent_actuation[-4],
                    v=movej_values['default']['v']
                    if recent_actuation[-3] == ur_utils.USE_DEFAULT else recent_actuation[-3],
                    t=movej_values['default']['t']
                    if recent_actuation[-2] == ur_utils.USE_DEFAULT else recent_actuation[-2],
                    r=movej_values['default']['r']
                    if recent_actuation[-1] == ur_utils.USE_DEFAULT else recent_actuation[-1],
                )
            elif recent_actuation[0] == ur_utils.COMMANDS['STOPJ']['id']:
                cmd = ur_utils.StopJ(
                    a=recent_actuation[1]
                )
            elif recent_actuation[0] == ur_utils.COMMANDS['UNLOCK_PSTOP']['id']:
                print("Unlocking p-stop")
                self._dashboard_sock.send('unlock protective stop\n'.encode('ascii'))
                return
            elif recent_actuation[0] == ur_utils.COMMANDS['NOTHING']['id']:
                return
            else:
                raise NotImplementedError
            cmd_str = '{}\n'.format(cmd)
            self._commands_queue.append([cmd_str.encode('ascii'),
                                        time.time(),
                                        np.minimum(int(np.random.exponential(scale=self._delay) / 0.008) * 0.008, 0.32)])

    def pre_check(self, data):
        """Checks time and completeness of packet reception.

        Args:
            data: a numpy array with sensory information received from UR5
        """
        if self._recv_time > self._prev_recv_time + 1.1 / 125:
            print(
                '{}: Hiccup of {:.2f}ms overhead between UR packets)'.format(
                    self._recv_time - self._start_time,
                    (self._recv_time - self._prev_recv_time - 0.008) * 1000,
                ))
        if len(data) != ur_utils.REALTIME_COMM_PACKET.itemsize:
            print('Warning: incomplete packet from UR')
            return

    @staticmethod
    def make_connection(host, port, disable_nagle_algorithm):
        """Establishes a TCP/IP socket connection with a UR5 controller.

        Args:
            host: a string specifying UR5 Controller IP address
            port: a string specifying UR5 Controller port
            disable_nagle_algorithm: a boolean specifying whether to
                disable nagle algorithm

        Returns:
             None or TCP socket connected to UR5 device.
        """
        sock = None
        for res in socket.getaddrinfo(host,
                                      port,
                                      socket.AF_UNSPEC,
                                      socket.SOCK_STREAM):
            afam, socktype, proto, canonname, sock_addr = res
            del canonname
            try:
                sock = socket.socket(afam, socktype, proto)
                if disable_nagle_algorithm:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
            except OSError as msg:
                print(msg)
                sock = None
                continue
            try:
                sock.connect(sock_addr)
            except OSError as msg:
                print(msg)
                sock.close()
                sock = None
                continue
            break
        return sock
