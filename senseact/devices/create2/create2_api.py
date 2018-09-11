# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import serial
import struct
import inspect
import time
import logging
import numpy as np

from multiprocessing import Lock

import senseact.devices.create2.create2_config as create2_config


class Create2SerialInterface(object):
    """The class represents Create2 serial communication interface.

    Allows for easier porting to C in the future if required.
    """
    def __init__(self, port, baudrate, timeout=2.0):
        """Inits Create2SerialInterface object with device-specific parameters.

        Creates and opens a serial connection to the Create2 device. Attempts to wake
        the robot by sending additional pulses on the BRC pin.

        Args:
            port:  a string specifying the port to connect, e.g. '/dev/ttyUSB0'
            baudrate: an integer specifying the target baudrate (should always
                be 115200 on start, unless previously changed by `baud` command)
            timeout: a float specifying the timeout for the read/write block
        """
        self._serial = serial.Serial(port=port,
                                     baudrate=baudrate,
                                     bytesize=serial.EIGHTBITS,
                                     parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE,
                                     timeout=timeout,
                                     writeTimeout=timeout,
                                     xonxoff=False,
                                     rtscts=False,
                                     dsrdtr=False)
        # Create2 seems to support full-duplex, but we still won't want two threads to do the
        # same operation at the same time (ie. 2 threads both read, or 2 threads both write)
        self._serial_read_lock = Lock()
        self._serial_write_lock = Lock()

        time.sleep(0.1)
        self.pulse()

        self._serial.flush()

    def pulse(self):
        """Pulses the BRC pin to wake or prevent sleep in PASSIVE mode.

        This is the documented way to keep Create2 alive, but it doesn't prevent sleep.
        """
        # Waking the robot from sleep; the sequence of 3 is required.
        logging.debug("Sending BRC pulse to the Create2.")
        self._serial.rts = True
        time.sleep(1.0)
        self._serial.rts = False
        time.sleep(1.0)
        self._serial.rts = True

    def write(self, opcode, data=None):
        """Sends actuation command to the Create2.

        Args:
            opcode: an integer representing a valid opcode
            data: a byte or an array containing the data for each byte of output
        """
        output_bytes = [opcode]
        if isinstance(data, list):
            output_bytes.extend(data)
        elif data is not None:
            output_bytes.append(data)

        logging.debug("Sending opcode {} with data: {}".format(opcode, data))
        with self._serial_write_lock:
            self._serial.write(struct.pack('B' * len(output_bytes), * output_bytes))

    def read(self, num_bytes):
        """Reads sensory data from the Create2.

        Args:
            num_bytes: an integer number of bytes to wait for and read

        Returns:
            A raw byte array containig data received.

        Raises:
            IOError: if nothing is read
        """
        logging.debug("Reading {} bytes from port.".format(num_bytes))
        with self._serial_read_lock:
            data = self._serial.read(num_bytes)

        if len(data) != num_bytes:
            raise IOError("Error reading from serial port. Received: {}".format(data))

        logging.debug("Received: {}".format(data))
        return data

    def read_all(self):
        """Reads from the Create2 until there's nothing in the input buffer.

        Returns:
           A raw byte array containing data received.
        """
        logging.debug("Reading all available data from port.")

        data = b''
        with self._serial_read_lock:
            size = self._serial.in_waiting
            while size:
                data += self._serial.read(size)
                size = self._serial.in_waiting
        logging.debug("Received: {}".format(data))
        return data

    def flush_input(self):
        """Flushes any unread input."""
        with self._serial_read_lock:
            self._serial.reset_input_buffer()

    def flush_output(self):
        """Flushes any unsent output."""
        with self._serial_write_lock:
            self._serial.reset_output_buffer()

    def close(self):
        """Closes the serial connection."""
        self._serial.close()


class Create2(object):
    """Main Create2 driver interface."""

    def __init__(self):
        """Creates the api without actually connecting to the device."""
        self._mode = 0

        # create2 needs this amount of time to react to mode change command
        self._internal_time = 0.15

        # store the correct streaming packet ids
        self._stream_packets = []
        self._streaming = False
        self._serial = None

    def connect(self, port, baudrate):
        """Makes the actual connection to Create2.

        Args:
            port: a string specifying the port to use, for example /dev/ttyUSB0
            baudrate: an integer specifying baudrate to connect (Create2 startup in 115200)
        """
        self._serial = Create2SerialInterface(port, baudrate)

    def disconnect(self):
        """Closes the serial connection.

        User should make sure the Create2 is in OFF/PASSIVE mode to avoid
        battery discharging (Create2 can't charge on other mode).
        """
        self._serial.close()
        self._serial = None

    # ========== Open Interface commands ==========

    def start(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['start'])
        time.sleep(self._internal_time)
        self._serial.flush_input()
        self._mode = 1

    def reset(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['reset'])
        self._serial.flush_input()
        # sleep to give the hardware time to react, log the returned welcome message
        time.sleep(5.0)
        logging.info(self._serial.read_all())

    def stop(self, *args):
        self._serial.flush_output()
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['stop'])
        time.sleep(self._internal_time)
        self._serial.flush_input()
        # for some reason the stop command sometimes switch to PASSIVE instead
        # of OFF, so include that mode in the verification
        # self._verify_mode([0, 1])
        self._mode = [0, 1]

    def baud(self, baudrate, *args):
        baud_dict = {
            300: 0,
            600: 1,
            1200: 2,
            2400: 3,
            4800: 4,
            9600: 5,
            14400: 6,
            19200: 7,
            28800: 8,
            38400: 9,
            57600: 10,
            115200: 11
        }

        if baudrate not in baud_dict:
            raise ValueError("Invalid buad rate: {}".format(baudrate))

        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['baud'], baud_dict[baudrate])
        time.sleep(self._internal_time)

    def safe(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['safe'])
        time.sleep(self._internal_time)
        self._serial.flush_input()
        self._mode = 2

    def full(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['full'])
        time.sleep(self._internal_time)
        self._serial.flush_input()
        self._mode = 3

    def clean(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['clean'])

    def max(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['max'])

    def spot(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['spot'])

    def seek_dock(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['seek_dock'])

    def power(self, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['power'])

    def schedule(self, *args):
        raise NotImplementedError

    def set_day_time(self, *args):
        raise NotImplementedError

    def drive(self, velocity, radius, *args):
        velocity = max(-500, min(500, velocity))
        if radius not in [32767, 32768, -1, 1]:
            radius = max(-2000, min(2000, radius))

        # two's complement and convert to hex
        data = [velocity >> 8 & 0x00ff, velocity & 0x00ff, radius >> 8 & 0x00ff, radius & 0x00ff]
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['drive'], data)

    def drive_direct(self, left, right, *args):
        left = max(-500, min(500, left))
        right = max(-500, min(500, right))

        # two's complement and convert to hex
        data = [right >> 8 & 0x00ff, right & 0x00ff, left >> 8 & 0x00ff, left & 0x00ff]
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['drive_direct'], data)

    def drive_pwm(self, *args):
        raise NotImplementedError

    def motors(self, value, *args):
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['motors'], value)

    def motors_pwm(self, *args):
        raise NotImplementedError

    def led(self, *args):
        raise NotImplementedError

    def scheduling_led(self, *args):
        raise NotImplementedError

    def digit_led_raw(self, *args):
        raise NotImplementedError

    def buttons(self, *args):
        raise NotImplementedError

    def digit_led_ascii(self, *args):
        raise NotImplementedError

    def song(self, *args):
        raise NotImplementedError

    def play(self, *args):
        raise NotImplementedError

    def sensors(self, packet_id, *args):
        if packet_id in create2_config.PACKET_INFO:
            self._serial.write(create2_config.OPCODE_NAME_TO_CODE['sensors'], packet_id)
        else:
            raise ValueError("Invalid packet id, failed to send")

    def query_list(self, *args):
        raise NotImplementedError

    def stream(self, packet_ids, *args):
        # check if all packet ids are valid
        if any([packet_id not in create2_config.PACKET_INFO for packet_id in packet_ids]):
            raise Exception("Packet ids contain invalid value.")

        self._stream_packets = packet_ids
        self._streaming = True

        actual_packet = [len(packet_ids)]
        actual_packet.extend([int(packet_id) for packet_id in packet_ids])
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['stream'], actual_packet)

    def pause_resume_stream(self, pause_resume_flag, *args):
        if not self._stream_packets:
            return
        if pause_resume_flag == 0:
            self._streaming = False
        else:
            self._streaming = True
        self._serial.write(create2_config.OPCODE_NAME_TO_CODE['pause_resume_stream'], pause_resume_flag)
        time.sleep(self._internal_time)
        self._serial.flush_input()

    # ========== Helper methods ==========

    @staticmethod
    def get_op_params(opcode):
        """Returns the parameters definition of a Create2 actuator command.

        Args:
            opcode: an integer representing the opcode of the command

        Returns:
            An OrderedDict containing the parameters.
        """
        opcode_info = create2_config.OPCODE_INFO[opcode]
        opcode_params = opcode_info['params'] if 'params' in opcode_info else {}

        # verify the config matches the signature
        param_signature = inspect.signature(getattr(Create2, opcode_info['name'])).parameters
        # add 2 for 'self' and '*args'
        if len(param_signature) != len(opcode_params) + 2:
            raise Exception("Opcode {} configuration and signature mismatch.".format(opcode))

        for p in opcode_params.keys():
            if p not in param_signature:
                raise Exception("Opcode {} configuration and signature mismatch.".format(opcode))

        return opcode_params

    def exec_op(self, opcode, *args):
        """Calls the corresponding command given opcode and arguments.

        Args:
            opcode: an integer representing the opcode of the command
            *args: additional arguments (unprocessed non-raw form) if any
        """
        if not self._serial:
            return

        op = getattr(self, create2_config.OPCODE_INFO[opcode]['name'])
        op(*args)

    @staticmethod
    def get_packet_dtype(packet_id):
        """Gets the numpy datatype of the packet.

        Args:
            packet_id: an integer representing the packet id of the target packet

        Returns:
            A numpy dtype structure.
        """
        if packet_id not in create2_config.PACKET_INFO:
            raise ValueError("Invalid packet id: {}".format(packet_id))

        packet_info = create2_config.PACKET_INFO[packet_id]
        packet_dtype = []
        if "subpackets" in packet_info:
            for subpacket_id in packet_info['subpackets']:
                packet_dtype.extend(Create2.get_packet_dtype(subpacket_id).descr)
        elif packet_info['fmt'] != '-':
            packet_dtype.append((packet_info['name'], '>' + packet_info['fmt']))
        else:
            packet_dtype.append((packet_info['name'], '>B'))

        return np.dtype(packet_dtype)

    def get_packet(self, packet_id):
        """Gets a single sensor packet information.

        The Create2 guide suggests not calling this faster than 15 ms.

        Args:
            packet_id: an integer representing a valid packet id

        Returns:
            A structured numpy array containing the data received
        """
        if not self._serial:
            return None

        packet_dtype = self.get_packet_dtype(packet_id)

        self.sensors(packet_id)
        packet_byte_data = self._serial.read(packet_dtype.itemsize)

        return np.frombuffer(packet_byte_data, dtype=packet_dtype)

    def get_stream_packets(self):
        """Gets the next available stream packets if streaming is enabled.

        Create2 updates the stream every 15 ms, so if the read is less
        frequent than that it could be reading from the buffer and returning
        states that are no longer valid.

        Returns:
            A structured numpy array containing the data received

        Raises:
            IOError:  if after 1 sec still unable to align the packet / pass checksum
        """
        if not self._serial or not self._streaming:
            return None

        packet_dtypes = []
        for packet_id in self._stream_packets:
            packet_dtypes.extend(self.get_packet_dtype(packet_id).descr)
        packets_dtype = np.dtype(packet_dtypes)

        packets_size = packets_dtype.itemsize

        # return structure is [19][N-bytes][Packet ID 1][Packet 1 data...][Packet ID 2][Packet 2 data...][Checksum]
        # we need to add the 1 byte for each packet id, [19], [N-bytes] and [Checksum]
        packets_size += len(self._stream_packets) + 3

        start_time = time.time()

        # loop and read the next fully aligned packets
        packets_byte_data = b''
        while len(packets_byte_data) != packets_size:
            packets_byte_data += self._serial.read(packets_size - len(packets_byte_data))

            if packets_byte_data[0] == 19 and \
                    packets_byte_data[1] == packets_size - 3 and \
                    sum(packets_byte_data) & 0xFF == 0:
                break
            else:
                logging.debug("Misaligned packet: {}".format(packets_byte_data))
                # find the next possible aligned point if any
                try:
                    next_start = packets_byte_data[1:].index(19)
                except ValueError:
                    next_start = packets_size

                packets_byte_data = packets_byte_data[next_start+1:]

            # do not want to loop forever unable to align packets
            if time.time() - start_time > 1.0:
                raise IOError("Serial connection taking too long to get the next aligned packet.")

        # strip away the extra info from the raw packet
        stripped_byte_data = b''
        curr_byte = 2
        for i, packet_id in enumerate(self._stream_packets):
            next_byte = curr_byte + 1 + packets_dtype[i].itemsize
            stripped_byte_data += packets_byte_data[curr_byte+1:next_byte]
            curr_byte = next_byte

        return np.frombuffer(stripped_byte_data, dtype=packets_dtype)

    def verify_mode(self):
        """Helper method to check the Create2's current OI mode against expected.

        Raises:
            Exception when the current mode is not the target.
        """
        curr_mode = self.get_packet(create2_config.PACKET_NAME_TO_ID['oi mode'])[0]['oi mode']
        if (isinstance(self._mode, list) and curr_mode not in self._mode) or \
                (isinstance(self._mode, int) and curr_mode != self._mode):
            raise Exception("Create2 not at target mode: expected {} got {}".format(self._mode, curr_mode))

    def wake(self):
        """Attempts to ping the Create2 to wake it up.

        Create2 goes to sleep after 5 min in PASSIVE mode.  If docked, the sleep timer is
        60 sec instead.  Pulsing the BRC pin supposed to keep it awake but not in real
        practice.  However, it does seem to useful for waking it.
        """
        self._serial.pulse()

