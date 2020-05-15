# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import numpy as np

from multiprocessing import Array, Value, Lock
from multiprocessing.sharedctypes import RawArray


class SharedBuffer(object):
    """Represents a circular shared buffer data structure.

    This class encapsulates the data structure for a circular buffer of
    arrays stored in a shared address space to be accessed from multiple
    forked processes created by multiprocessing.Process.

    Attributes:
        array_len: An integer size of each buffer element (usually numpy array)
        array_type: A ctypes data type of buffer elements, e.g. 'd'
    """
    DEFAULT_BUFFER_LEN = 10

    def __init__(self, array_len, array_type, np_array_type, buffer_len=DEFAULT_BUFFER_LEN, array_lock=True):
        """Inits the SharedBuffer object with size and data type.

        Args:
            buffer_len: An integer size of the buffer
            array_len: An integer size of each buffer element (usually numpy array)
            array_type: A ctypes data type of buffer elements, e.g. 'd'
            np_array_type: A numpy data type of buffer elements, e.g. 'float64'
            array_lock: A bool specifying whether the buffer will be used with Lock

        """
        self.array_len = array_len
        self.np_array_type = np_array_type
        self._buffer_len = buffer_len
        self._array_type = array_type

        # Data is stored in a circular buffer of shared arrays
        self._data_buffer = []
        if array_lock:
            for _ in range(self._buffer_len):
                self._data_buffer.append(np.frombuffer(Array(self._array_type, self.array_len).get_obj(),
                                                       dtype=self.np_array_type))
            # We also store time stamps corresponding to each array record
            self._timestamp_buffer = Array('d', self._buffer_len)
            # We also store the index corresponding to each array record
            self._index_buffer = Array('l', self._buffer_len)
        else:
            # use RawArray without internal lock if needed
            for _ in range(self._buffer_len):
                self._data_buffer.append(np.frombuffer(RawArray(self._array_type, self.array_len),
                                                       dtype=self.np_array_type))
            self._timestamp_buffer = RawArray('d', self._buffer_len)
            self._index_buffer = RawArray('l', self._buffer_len)
        # Value of `index_buffer` is always set to `self._counter`, which is then increased
        self._counter = 0
        # buffer_p is a pointer which always points to the next available slot in `data_buffer`
        # where the newest data array can be stored
        self._buffer_p = Value('i', 0)
        # This variable is set to 1 when a new array is stored and
        # set to 0 when a new array is read
        self._data_updated = Value('i', 0)

        # Lock to ensure that changing the `data_updated` as well as the data itself is atomic
        self._access_lock = Lock()

    def read_update(self, window_len=1):
        """Reads the last window_len buffer elements and sets updated flag.

        Args:
            window_len: An integer number of elements to read from the buffer.
                Must not be greater than buffer_len.

        Returns:
            A list of last `window_len` buffer elements and corresponding timestamps.
            Also sets `data_updated` flag to 0 to notify that there is no fresh data unread.
        """
        self._access_lock.acquire()
        self._data_updated.value = 0
        ret = self._read(window_len)
        self._access_lock.release()
        return ret

    def read(self, window_len=1):
        """Reads the last window_len buffer elements and does not set updated flag.

        Args:
            window_len: An integer number of elements to read from the buffer.
                Must not be greater than buffer_len.

        Returns:
            A list of last `window_len` buffer elements and corresponding timestamps.
        """
        self._access_lock.acquire()
        ret = self._read(window_len)
        self._access_lock.release()
        return ret

    def _read(self, window_len=1):
        """Returns a list of last 'window_len' data arrays and corresponding timestamps.

        Should not be called by itself since the content of the different windows
        will not be synced without lock.

        Args:
            window_len: An integer number of elements to read from the buffer.
                Must not be greater than buffer_len.

        Returns:
            A tuple (data_window, timestamp_window, index_window) of lists of last
            `window_len` buffer elements and corresponding timestamps and indices.
        """
        data_window = []
        timestamp_window = []
        index_window = []
        buffer_pointer = self._buffer_p.value
        for i in range(-window_len, 0):
            index = (self._buffer_len + buffer_pointer + i) % self._buffer_len
            data_window.append(self._data_buffer[index].copy())
            timestamp_window.append(self._timestamp_buffer[index])
            index_window.append(self._index_buffer[index])
        return data_window, timestamp_window, index_window

    def write(self, data, timestamp=None):
        """Writes a new element to the circular buffer.

        Writes a data array to the circular buffer, writes the timestamp,
        moves the buffer pointer to the next available location, and
        sets the `data_updated` flag to one to notify that there is
        fresh unread data.

        Args:
            data: A numpy array of size `self._array_len` and type `self._array_type`
            timestamp: a timestamp to store with the data
        """
        self._access_lock.acquire()
        buffer_pointer = self._buffer_p.value
        np.copyto(self._data_buffer[buffer_pointer], data)
        if timestamp is None:
            timestamp = time.time()
        self._timestamp_buffer[buffer_pointer] = timestamp
        self._index_buffer[buffer_pointer] = self._counter
        self._buffer_p.value = (buffer_pointer + 1) % self._buffer_len
        self._data_updated.value = 1
        self._counter += 1
        self._access_lock.release()

    def updated(self):
        """Returns a bool indicating arrival of new data.

        Returns:
            The value of `_data_updated` typically to indicate whether
            `data_buffer` has new data or not.
        """
        self._access_lock.acquire()
        updated = self._data_updated.value
        self._access_lock.release()
        return updated

    def set_data_update(self, value):
        """Sets the value of `_data_updated`.

        Typically, setting value to 0 means fresh data is read from
        `data_buffer` and setting value to 1 means fresh data is written
        to `data_buffer`.
        Args:
            value: either 0 or 1
        """
        self._access_lock.acquire()
        self._data_updated.value = value
        self._access_lock.release()


class Command:
    sizes = {}

    def __init__(self):
        pass

    @classmethod
    def from_array(cls, the_array):
        obj = cls()
        idx = 0
        for key in sorted(obj.__dict__.keys()):
            size = cls.sizes.get(key, 1)
            if size == 1:
                setattr(obj, key, the_array[idx])
            else:
                setattr(obj, key, the_array[idx:idx + size].tolist())

            idx += size

        return obj

    def to_array(self):
        out = []
        for key in sorted(self.__dict__.keys()):
            val = self.__dict__[key]
            if type(val) is list:
                out.extend(val)
            else:
                out.append(val)
        return out

    def __array__(self):
        return np.array(self.to_array())


class SharedBufferSerializer:
    class CommandEntry:
        def __init__(self, command_type, factory):
            self.command_type = command_type
            self.factory = factory

    def __init__(self, commands: list):
        """
        :param commands: Each list entry should contain (class type, factory method)
        """
        self.commands = list(commands)
        self.dictionary = {}
        for idx, command in enumerate(commands):
            if type(command) is tuple or type(command) is list:
                SharedBufferSerializer.CommandEntry(*command)
            elif type(command) is SharedBufferSerializer.CommandEntry:
                self.dictionary[command.command_type] = idx
            else:
                raise ValueError("Invalid Command argument")

    def marshall(self, command: Command) -> np.ndarray:
        """
        When an object is marshalled its index in the commands list is looked up by its type. This index
        is then written as the first entry in serialization array. The commands to_array method is then called
        and the result is added to the buffer entry.
        
        :param command: The command to be serialized
        """
        if type(command) not in self.dictionary:
            raise ValueError

        out = [self.dictionary[type(command)]]
        out.extend(command.to_array)
        return np.array(out)

    def demarshall(self, nparray: np.ndarray) -> object:
        """
        The first entry in the nparray indicates the class of object to be instantiated.
        This is an index into self.commands.
        The corresponding CommandEntry is fetched from the list and the factory method is called
        with the remainder of the buffer.
        :param nparray:
        :return:
        """
        cmd_idx = nparray[0]
        if cmd_idx < 0 or cmd_idx >= len(self.commands):
            raise ValueError

        clz = self.commands[cmd_idx]
        instance = clz.factory(nparray[1:])
        return instance
