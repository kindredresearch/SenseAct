# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import time
import numpy as np
from multiprocessing import Process, Value, Array
from multiprocessing.sharedctypes import RawArray
from senseact.sharedbuffer import SharedBuffer


class TestSharedBuffer(unittest.TestCase):
    def setUp(self):
        self.array_len = 7
        self.array_type = "d"
        self.np_array_type = "float64"
        self.buffer_len = SharedBuffer.DEFAULT_BUFFER_LEN

    def tearDown(self):
        return

    def testInitLockTrue(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        # Test array types are correct
        self.assertEqual(len(buffer._data_buffer), self.buffer_len)
        self.assertIsInstance(buffer._data_buffer[0], np.ndarray)
        self.assertIs(buffer._data_buffer[0].dtype, np.dtype(self.np_array_type))
        self.assertIsInstance(buffer._data_buffer[0].base,
                              type(Array(self.array_type, self.array_len).get_obj()))
        self.assertIsInstance(buffer._timestamp_buffer.get_obj(),
                              type(Array("d", self.buffer_len).get_obj()))
        self.assertIsInstance(buffer._index_buffer.get_obj(),
                              type(Array("l", self.buffer_len).get_obj()))

    def testInitLockFalse(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=False)
        # Test array types are correct
        self.assertEqual(len(buffer._data_buffer), self.buffer_len)
        self.assertIsInstance(buffer._data_buffer[0], np.ndarray)
        self.assertIs(buffer._data_buffer[0].dtype, np.dtype(self.np_array_type))
        self.assertIsInstance(buffer._data_buffer[0].base,
                              type(Array(self.array_type, self.array_len).get_obj()))
        self.assertIsInstance(buffer._timestamp_buffer,
                              type(RawArray("d", self.buffer_len)))
        self.assertIsInstance(buffer._index_buffer,
                              type(RawArray("l", self.buffer_len)))

    def testWriteBasic(self):
        buffer = SharedBuffer(array_len=2, array_type='d', np_array_type='d')
        buffer.write([1, 2])
        self.assertTrue(buffer.updated())

    def testWritePointerAdvance(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        true_counter = 0
        for i in range(self.buffer_len + 1):
            buffer.write(np.zeros(self.array_len))
            true_counter += 1
            if true_counter == self.buffer_len:
                true_counter = 0
            self.assertEqual(buffer._buffer_p.value, true_counter)

    def testWriteData(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        true_counter = 0
        for i in range(self.buffer_len + 1):
            data = np.random.random(self.array_len)
            buffer.write(data)
            self.assertTrue(np.array_equal(buffer._data_buffer[true_counter], data))
            true_counter += 1
            if true_counter == self.buffer_len:
                true_counter = 0

    def testWriteTimeStamp(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        true_counter = 0
        for i in range(self.buffer_len + 1):
            now = time.time()
            buffer.write(np.zeros(self.array_len))
            self.assertTrue(np.abs(buffer._timestamp_buffer[true_counter] - now) < 1e-3)
            true_counter += 1
            if true_counter == self.buffer_len:
                true_counter = 0
            time.sleep(np.random.random() * 0.1)

    def testWriteIndex(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        true_counter = 0
        for i in range(self.buffer_len + 1):
            buffer.write(np.zeros(self.array_len))
            self.assertEqual(buffer._index_buffer[true_counter], i)
            true_counter += 1
            if true_counter == self.buffer_len:
                true_counter = 0
            time.sleep(np.random.random() * 0.1)

    def testRead(self):
        buffer = SharedBuffer(array_len=2, array_type='d', np_array_type='d')
        buffer.write([1, 2])
        data, timestamp, index = buffer.read()
        np.testing.assert_array_equal(data, [[1, 2]])
        self.assertTrue(buffer.updated())

    def testReadUpdate(self):
        buffer = SharedBuffer(array_len=2, array_type='d', np_array_type='d')
        buffer.write([1, 2])
        data, timestamp, index = buffer.read_update()
        np.testing.assert_array_equal(data, [[1, 2]])
        self.assertFalse(buffer.updated())

    def testReadWindow(self):
        buffer = SharedBuffer(array_len=2, array_type='d', np_array_type='d', buffer_len=2)
        buffer.write([1, 2])
        buffer.write([2, 3])
        data, timestamp, index = buffer.read_update(2)
        np.testing.assert_array_equal(data, [[1, 2], [2, 3]])

    def testReadWindowLoopOver(self):
        buffer = SharedBuffer(array_len=2, array_type='d', np_array_type='d', buffer_len=3)
        buffer.write([1, 2])
        buffer.write([2, 3])
        buffer.write([3, 4])
        buffer.write([4, 5])
        data, timestamp, index = buffer.read_update(4)
        np.testing.assert_array_equal(data, [[4, 5], [2, 3], [3, 4], [4, 5]])

    def testReadLoop(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        # fake some data and push into buffer
        true_data = []
        true_times = []
        true_indices = []
        for i in range(self.buffer_len + 1):
            data = np.random.random(self.array_len)
            true_data.append(data)
            true_times.append(time.time())
            true_indices.append(i)
            buffer.write(data)

        # read windows of varied length from the buffer
        # and compare to the ground truth data,
        # time stamps and indices
        test_sizes = [1,
                      2,
                      self.buffer_len//2,
                      self.buffer_len-1,
                      self.buffer_len
                      ]
        for size in test_sizes:
            data_window, time_stamp, index = buffer._read(size)
            for i in range(1, size + 1):
                self.assertTrue(np.array_equal(data_window[-i], true_data[-i]))
                self.assertTrue(np.abs(time_stamp[-i] - true_times[-i]) < 1e-3)
                self.assertEqual(index[-i], true_indices[-i])

    def testUpdates(self):
        buffer = SharedBuffer(array_len=self.array_len,
                              array_type=self.array_type,
                              np_array_type=self.np_array_type,
                              array_lock=True)
        # Test updated flag functionality
        self.assertEqual(buffer.updated(), 0)
        buffer.write(np.zeros(self.array_len))
        self.assertEqual(buffer.updated(), 1)
        buffer.read(1)
        self.assertEqual(buffer.updated(), 1)
        buffer.read_update(1)
        self.assertEqual(buffer.updated(), 0)
        buffer.set_data_update(1)
        self.assertEqual(buffer.updated(), 1)
        buffer.set_data_update(0)
        self.assertEqual(buffer.updated(), 0)

    def testLock(self):
        def spawn_write_process(buffer):
            buffer.write([1, 2])

        def spawn_read_process(buffer, hold_lock):
            def mock_read(*args, **kwargs):
                while hold_lock.value == 1:
                    time.sleep(0.01)
            buffer._read = mock_read
            buffer.read_update()

        buffer = SharedBuffer(array_len=2, array_type='d', np_array_type='d', buffer_len=3)
        hold_lock = Value('i', 1)
        # spawn a read process that will just hold onto the read to simulate large read
        read_process = Process(target=spawn_read_process, args=(buffer, hold_lock))
        read_process.start()
        time.sleep(0.1)
        # spawn a write process that should get stuck unable to get lock to write
        write_process = Process(target=spawn_write_process, args=(buffer,))
        write_process.start()
        time.sleep(0.5)

        self.assertTrue(write_process.is_alive())
        hold_lock.value = 0
        time.sleep(0.5)
        self.assertFalse(write_process.is_alive())


if __name__ == '__main__':
    unittest.main(buffer=True)
