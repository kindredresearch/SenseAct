# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import unittest
import time
import psutil
import numpy as np
from multiprocessing import Value

from senseact.communicator import Communicator


class MockCommunicator(Communicator):
    """
    Basic barebone test communicator that can crash on demand.
    """
    def __init__(self):
        self._dt = 0.008
        # shared variable that all processes will see
        self.crash_flag = Value('i', 0)
        self.sense_flag = Value('i', 0)
        self.actor_flag = Value('i', 0)
        self.last_actuation = Value('d', 0)
        self.last_observation = Value('d', 0)
        sensor_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        actuator_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        super().__init__(use_sensor=True, use_actuator=True, sensor_args=sensor_args, actuator_args=actuator_args)

    def _sensor_handler(self):
        if self.crash_flag.value == 1:
            raise Exception("Random sensor exception encountering")
        self.sense_flag.value = 1
        obs = np.random.random() # received an observation from robot
        self.last_observation.value = obs
        self.sensor_buffer.write(obs)
        time.sleep(self._dt)

    def _actuator_handler(self):
        if self.crash_flag.value == 2:
            raise Exception("Random actuator exception encountering")
        self.actor_flag.value = 1
        if self.actuator_buffer.updated():
            actuation, _, _ = self.actuator_buffer.read_update()
            self.last_actuation.value = actuation[0]
        time.sleep(0.0001)


class TestCommunicator(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def testInit(self):
        sensor_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        actuator_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        comm = Communicator(sensor_args=sensor_args, actuator_args=actuator_args, use_sensor=True, use_actuator=True)
        self.assertEqual(comm.sensor_buffer.array_len, 1)
        self.assertEqual(comm.actuator_buffer.array_len, 1)

    def testRunNoImplementation(self):
        sensor_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        actuator_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        curr_process_util = psutil.Process()
        comm = Communicator(sensor_args=sensor_args, actuator_args=actuator_args, use_sensor=True, use_actuator=True)

        # the two child thread should die from the handler not being implemented
        comm.start()
        comm.join()
        self.assertEqual(len(curr_process_util.children(recursive=True)), 0)

    def testRunAndTerminate(self):
        curr_process_util = psutil.Process()
        comm = MockCommunicator()
        comm.start()
        time.sleep(0.5)
        child_processes = curr_process_util.children(recursive=True)
        self.assertEqual(len(child_processes), 1)
        self.assertEqual(child_processes[0].num_threads(), 3)
        comm.terminate()
        comm.join()
        self.assertEqual(len(curr_process_util.children(recursive=True)), 0)

    def testHandler(self):
        comm = MockCommunicator()
        comm.start()
        time.sleep(0.5)
        self.assertEqual(comm.sense_flag.value, 1)
        self.assertEqual(comm.actor_flag.value, 1)
        comm.terminate()
        comm.join()

    def testActuatorHandler(self):
        comm = MockCommunicator()
        comm.start()
        # send fake actuations in real time and make sure communicator
        # communicates them through with no more than 1ms delay
        time.sleep(0.5)
        self.assertEqual(comm.sense_flag.value, 1)
        self.assertEqual(comm.actor_flag.value, 1)
        steps = 10
        for i in range(steps):
            actuation = np.random.random()  # received action from the agent
            comm.actuator_buffer.write(actuation)
            time.sleep(0.001)
            self.assertEqual(comm.last_actuation.value, actuation)
        comm.terminate()
        comm.join()

    def testSensorHandler(self):
        comm = MockCommunicator()
        comm.start()
        # generate fake robot sensory data in real time and make sure
        # communicator communciates them through with no more than 1ms delay
        steps = 10
        count = 0
        while count < steps:
            if comm.sensor_buffer.updated():
                obs = comm.sensor_buffer.read_update(1)
                self.assertEqual(comm.last_observation.value, obs[0][0])
                self.assertTrue(np.abs(obs[1][0] - time.time()) < 1e-3)
                count += 1
            time.sleep(0.0001)

        comm.terminate()
        comm.join()

    def testHandlerException(self):
        curr_process_util = psutil.Process()
        comm = MockCommunicator()
        comm.start()
        time.sleep(0.5)
        comm.crash_flag.value = 1
        comm.join()
        self.assertEqual(len(curr_process_util.children(recursive=True)), 0)


if __name__ == '__main__':
    unittest.main(buffer=True)
