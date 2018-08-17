import time
import logging
import os
import signal

from threading import Thread
from multiprocessing import Process

from senseact.sharedbuffer import SharedBuffer


class Communicator(Process):
    """An abstract base class for device-specific packet communicators.

    Communicators create interface between a physical device and
    a Reinforcement Learning system. A communciator facilitates reading
    sensory data from the device and sending actuation data to the device
    if applicable. A communicator runs two python threads,
    one of them "listens" to device and transfer sensory data as soon
    as those are available, the other waits on actuation commands from RL
    system and transfers those to the device as soon as those are available.
    The threads are run asynchronous in order to avoid blocking and unnecessary
    delays. Typically, one would create one instance of Communicator
    per device, which will handle the packet communication with that device,
    for example, a Dynamixel actuator or a camera. A Communicator object
    can be instantiated by inheriting this class and implementing _sensor_handler
    for listening to a device and implementing _actor_handler to transfer
    actuation commands. The two python threads run these two methods continually.
    There can be communicators having only one of the methods implemented and
    thus running only one of the threads for a device with simplex communication.

    Attrbiutes:
        use_sensor: a boolean specifying whether given communicator
            transmits sensory data.
        use_actuator: a boolean specifying whether given communicator
            transmits actuation data
        sensor_buffer: a SharedBuffer object containing the latest sensory
            data to be received from a device and stored in the buffer by
            _sensor_handler
        actuator_buffer: a SharedBuffer object containing the latest
            actuation data based on agent's action, which is to be received
            from the buffer by _actuator_handler
    """

    def __init__(self, sensor_args, actuator_args, use_sensor=True, use_actuator=True):
        """Inits communicator class with device-specific sensor and actuation arguments.

        Args:
            sensor_args: a dictionary containing information about device-specific
                sensory data in the format:
                {
                'buffer_len': length of the sensor buffer, where each element is an array
                'array_len': size of the array required to store a single sensory packet
                'array_type': ctypes shared array data type, e.g. 'd'
                'np_array_type': numpy array data type, e.g. 'float64', a shared array
                                type is interpreted to
                }
            actuator_args: a dictionary containing information about device-specific
                actuation commands in the format specified above.
            use_sensor: a boolean, indicating whether communicator will transfer
                sensory data
            use_actuator: a boolean, indicating whether communicator will transfer
                actuation data (e.g. video cameras may not have actuations)
        """
        super(Communicator, self).__init__()
        self.use_sensor = use_sensor
        self.use_actuator = use_actuator
        self._parent_pid = os.getpid()
        self._sensor_thread = None
        self._actuator_thread = None
        self._sensor_running = False
        self._actuator_running = False
        if self.use_sensor:
            self.sensor_buffer = SharedBuffer(**sensor_args)

        if self.use_actuator:
            self.actuator_buffer = SharedBuffer(**actuator_args)

    def run(self):
        """Starts sensor and actuator related threads/processes if they exist."""
        # catching SIGTERM from terminate() call so that we can close thread
        # on this spawn-process
        signal.signal(signal.SIGTERM, self._close)

        if self.use_sensor:
            self._sensor_start()

        if self.use_actuator:
            self._actuator_start()

        while self._sensor_running or self._actuator_running:
            # if parent pid is no longer the same (1 on Linux if re-parented to init), then
            # main process has been closed
            if os.getppid() != self._parent_pid:
                logging.info("Main environment process has been shutdown, closing communicator.")
                self._close()
                return

            if (self._sensor_thread is not None and not self._sensor_thread.is_alive()) or \
               (self._actuator_thread is not None and not self._actuator_thread.is_alive()):
                logging.error("Sensor/Actuator thread has exited, closing communicator.")
                self._close()
                return

            time.sleep(1)

    def _sensor_start(self):
        """Starts sensor thread."""

        self._sensor_thread = Thread(target=self._sensor_run)
        self._sensor_thread.start()

    def _actuator_start(self):
        """Starts actuator thread."""
        self._actuator_thread = Thread(target=self._actuator_run)
        self._actuator_thread.start()

    def _sensor_run(self):
        """Loop for handling sensors."""
        self._sensor_running = True
        while self._sensor_running:
            self._sensor_handler()

    def _actuator_run(self):
        """Loop for handling actuators."""
        self._actuator_running = True
        while self._actuator_running:
            self._actuator_handler()

    def _sensor_handler(self):
        """Handles sensor packet communication and necessary processing.

        Re-establishes connection when it is lost.
        """
        raise NotImplementedError

    def _actuator_handler(self):
        """Handles actuator packet communication and necessary processing.

        Re-establishes connection when it is lost.
        """
        raise NotImplementedError

    def _close(self, *args, **kwargs):
        """Closes child threads and processes and performs necessary clean up.

        Close method can only be called on the spawn-process after `run` is called.
        *args, **kwargs - arguments required by signal.SIGTERM
        """
        self._sensor_running = False
        self._actuator_running = False
        if self._sensor_thread is not None:
            self._sensor_thread.join()
        if self._actuator_thread is not None:
            self._actuator_thread.join()

