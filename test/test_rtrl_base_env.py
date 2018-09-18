# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
import numpy as np
import unittest
import time
import psutil

from multiprocessing import Process, Value, Array

from senseact import utils
from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.communicator import Communicator


class MockCommunicator(Communicator):
    """
    Basic barebone test communicator that can crash on demand.
    """
    def __init__(self):
        self._dt = 0.008

        # shared variable that all processes will see
        self.crash_flag = Value('i', 0)

        sensor_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        actuator_args = {'array_len': 1, 'array_type': 'd', 'np_array_type': 'd'}
        super().__init__(use_sensor=True, use_actuator=True, sensor_args=sensor_args, actuator_args=actuator_args)

    def _sensor_handler(self):
        if self.crash_flag.value == 1:
            raise Exception("Random sensor exception encountering")

        self.sensor_buffer.write(0)
        time.sleep(self._dt)

    def _actuator_handler(self):
        if self.crash_flag.value == 2:
            raise Exception("Random actuator exception encountering")

        if self.actuator_buffer.updated():
            actuation, _, _ = self.actuator_buffer.read_update()
        time.sleep(self._dt)


class MockEnv(RTRLBaseEnv):
    """
    Basic barebone test environment that can crash on demand.
    """
    def __init__(self, action_dim, observation_dim, **kwargs):
        # shared variable that all processes will see
        self.crash_flag = Value('i', 0)
        self.reset_call_flag = Value('i', 0)

        # Communicator Parameters
        communicator_setups = {'generic1': {'Communicator': MockCommunicator,
                                            'kwargs': {}},
                               'generic2': {'Communicator': MockCommunicator,
                                            'kwargs': {}}
                              }

        self._uniform_array_ = np.frombuffer(Array('d', 3).get_obj(), dtype=np.float64)

        super().__init__(communicator_setups=communicator_setups,
                         action_dim=action_dim,
                         observation_dim=observation_dim,
                         **kwargs)

    def _write_action(self, action):
        if self.crash_flag.value == 3:
            raise Exception("Write action crash triggered.")

        super(MockEnv, self)._write_action(action)

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        if self.crash_flag.value == 1:
            raise Exception("Compute sensation crash triggered.")

        return [3,2,1]

    def _compute_actuation_(self, action, timestamp, index):
        if self.crash_flag.value == 2:
            raise Exception("Compute actuation crash triggered.")

        self._actuation_packet_['generic1'] = action
        self._actuation_packet_['generic2'] = action
        values = self._rand_obj_.uniform(-1, +1, 3)
        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))
        np.copyto(self._uniform_array_, values)

    def _reset_(self):
        self.reset_call_flag.value = 1


class TestRTRLBaseEnv(unittest.TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def testInit(self):
        env = RTRLBaseEnv({}, 2, 3)

        self.assertFalse(env._running)
        self.assertEqual(env._action_buffer.array_len, 2)
        self.assertEqual(env._sensation_buffer.array_len, 5)

    def testInitWithCommunicator(self):
        env = RTRLBaseEnv({'generic': {'Communicator': MockCommunicator, 'kwargs': {}}}, 2, 3)

        self.assertFalse(env._running)
        self.assertEqual(len(env._all_comms), 1)
        self.assertEqual(env._action_buffer.array_len, 2)
        self.assertEqual(env._sensation_buffer.array_len, 5)

    def testStartSingalthread(self):
        env = RTRLBaseEnv({}, 2, 3, run_mode='singlethread')
        env.start()
        self.assertTrue(env._running)
        env.close()
        self.assertFalse(env._running)

    def testStartMultithread(self):
        env = RTRLBaseEnv({}, 2, 3, run_mode='multithread')
        env.start()
        self.assertTrue(env._running)
        time.sleep(0.5)
        self.assertTrue(env._polling_loop.is_alive())
        env.close()
        self.assertFalse(env._running)
        self.assertFalse(env._polling_loop.is_alive())

    def testStartMultiprocess(self):
        env = RTRLBaseEnv({}, 2, 3, run_mode='multiprocess')
        env.start()
        self.assertTrue(env._running)
        time.sleep(0.5)
        self.assertTrue(env._polling_loop.is_alive())
        env.close()
        self.assertFalse(env._running)
        self.assertFalse(env._polling_loop.is_alive())

    def testNotImplementedError(self):
        env = RTRLBaseEnv({}, 2, 3, run_mode='singlethread')
        env.start()
        with self.assertRaises(NotImplementedError):
            env.step(0)
        env.close()

    def testStartWithCommunicator(self):
        env = RTRLBaseEnv({'generic': {'Communicator': MockCommunicator, 'kwargs': {}}}, 2, 3, run_mode='singlethread')
        env.start()
        time.sleep(0.5)
        self.assertTrue(env._running)
        self.assertTrue(env._all_comms['generic'].is_alive())
        env.close()
        self.assertFalse(env._running)
        self.assertFalse(env._all_comms['generic'].is_alive())

    def testStepWithSinglethread(self):
        env = MockEnv(1, 1, run_mode='singlethread')
        env.start()
        time.sleep(0.5)
        obs, reward, done, info = env.step(0)
        self.assertEqual(obs, [3])
        self.assertEqual(reward, 2)
        self.assertEqual(done, 1)
        env.close()

    def testStepWithMultithread(self):
        env = MockEnv(1, 1, run_mode='multithread')
        env.start()
        time.sleep(0.5)
        obs, reward, done, info = env.step(0)
        self.assertEqual(obs, [3])
        self.assertEqual(reward, 2)
        self.assertEqual(done, 1)
        env.close()

    def testStepWithMultiprocess(self):
        env = MockEnv(1, 1, run_mode='multiprocess')
        env.start()
        time.sleep(0.5)
        obs, reward, done, info = env.step(0)
        self.assertEqual(obs, [3])
        self.assertEqual(reward, 2)
        self.assertEqual(done, 1)
        env.close()

    def testResetSinglethread(self):
        env = MockEnv(1, 1, run_mode='singlethread')
        env.start()
        time.sleep(0.5)
        obs = env.reset()
        self.assertEqual(obs, [3])
        self.assertEqual(env.reset_call_flag.value, 1)
        env.close()

    def testResetMultithreadBlocking(self):
        env = MockEnv(1, 1, run_mode='multithread')
        env.start()
        time.sleep(0.5)
        obs = env.reset()
        self.assertEqual(obs, [3])
        self.assertEqual(env.reset_call_flag.value, 1)
        env.close()

    def testResetMultithreadNonblocking(self):
        env = MockEnv(1, 1, run_mode='multithread')
        env.start()
        time.sleep(0.5)
        obs = env.reset(blocking=False)
        time.sleep(0.5)
        self.assertEqual(obs, [3])
        self.assertEqual(env.reset_call_flag.value, 1)
        env.close()

    def testResetMultiprocessBlocking(self):
        env = MockEnv(1, 1, run_mode='multiprocess')
        env.start()
        time.sleep(0.5)
        obs = env.reset()
        self.assertEqual(obs, [3])
        self.assertEqual(env.reset_call_flag.value, 1)
        env.close()

    def testResetMultiprocessNonblocking(self):
        env = MockEnv(1, 1, run_mode='multiprocess')
        env.start()
        time.sleep(0.5)
        obs = env.reset(blocking=False)
        time.sleep(0.5)
        self.assertEqual(obs, [3])
        self.assertEqual(env.reset_call_flag.value, 1)
        env.close()

    def testSinglethreadCommNotAlive(self):
        self._env = MockEnv(action_dim=1, observation_dim=1, run_mode='singlethread')
        self._env.start()
        self._env.step(0)

        # set the communicator flag to 1, wait a few time steps and check that step will crash
        self._env._all_comms['generic1'].crash_flag.value = 1
        time.sleep(1.5)

        with self.assertRaises(Exception):
            self._env.step(0)

        # give some time for process to completely close
        time.sleep(1.5)

        # check all communicators has been closed
        for comm in self._env._all_comms.values():
            self.assertFalse(comm.is_alive())

        self._env.close()

    def testMultithreadCommNotAlive(self):
        self._env = MockEnv(action_dim=1, observation_dim=1, run_mode='multithread')
        self._env.start()
        self._env.step(0)

        # set the communicator flag to 1, wait a few time steps and check that step will crash
        self._env._all_comms['generic1'].crash_flag.value = 1
        time.sleep(1.5)

        with self.assertRaises(Exception):
            self._env.step(0)

        # give some time for process to completely close
        time.sleep(1.5)

        # check all communicators has been closed
        for comm in self._env._all_comms.values():
            self.assertFalse(comm.is_alive())

        # check the polling thread has been closed
        self.assertFalse(self._env._polling_loop.is_alive())

        self._env.close()

    def testMultiprocessCommNotAlive(self):
        self._env = MockEnv(action_dim=1, observation_dim=1, run_mode='multiprocess')
        self._env.start()
        self._env.step(0)

        # set the communicator flag to 1, wait a few time steps and check that step will crash
        self._env._all_comms['generic1'].crash_flag.value = 1
        time.sleep(1.5)

        with self.assertRaises(Exception):
            self._env.step(0)

        # give some time for process to completely close
        time.sleep(1.5)

        # check all communicators has been closed
        for comm in self._env._all_comms.values():
            self.assertFalse(comm.is_alive())

        # check the polling thread has been closed
        self.assertFalse(self._env._polling_loop.is_alive())

        self._env.close()

    def testMultithreadPollingDead(self):
        self._env = MockEnv(action_dim=1, observation_dim=1, run_mode='multithread')
        self._env.start()
        self._env.step(0)

        self._env.crash_flag.value = 1
        time.sleep(1.5)

        with self.assertRaises(Exception):
            self._env.step(0)

        # give some time for process to completely close
        time.sleep(1.5)

        # check all communicators has been closed
        for comm in self._env._all_comms.values():
            self.assertFalse(comm.is_alive())

        # check the polling thread has been closed
        self.assertFalse(self._env._polling_loop.is_alive())

        self._env.close()

    def testMultiprocessPollingDead(self):
        self._env = MockEnv(action_dim=1, observation_dim=1, run_mode='multiprocess')
        self._env.start()
        self._env.step(0)

        self._env.crash_flag.value = 1
        time.sleep(1.5)

        with self.assertRaises(Exception):
            self._env.step(0)

        # give some time for process to completely close
        time.sleep(1.5)

        # check all communicators has been closed
        for comm in self._env._all_comms.values():
            self.assertFalse(comm.is_alive())

        # check the polling thread has been closed
        self.assertFalse(self._env._polling_loop.is_alive())

        self._env.close()

    def testSinglethreadMainProcessDead(self):
        def spawn_main_process():
            env = MockEnv(action_dim=1, observation_dim=1, run_mode='singlethread')
            env.start()
            while True:
                env.step(0)

        curr_process_util = psutil.Process()

        main_process = Process(target=spawn_main_process)
        main_process.start()

        # give some time to make sure everything running
        time.sleep(1.0)
        child_processes = curr_process_util.children(recursive=True)
        self.assertEqual(len(child_processes), 3)

        main_process.terminate()
        main_process.join()
        time.sleep(2.0)
        self.assertFalse(any([c.is_running() for c in child_processes]))

    def testMultithreadMainProcessDead(self):
        def spawn_main_process():
            env = MockEnv(action_dim=1, observation_dim=1, run_mode='multithread')
            env.start()
            while True:
                env.step(0)

        curr_process_util = psutil.Process()

        main_process = Process(target=spawn_main_process)
        main_process.start()

        # give some time to make sure everything running
        time.sleep(1.0)
        child_processes = curr_process_util.children(recursive=True)
        self.assertEqual(len(child_processes), 3)

        main_process.terminate()
        main_process.join()
        time.sleep(2.0)
        self.assertFalse(any([c.is_running() for c in child_processes]))

    def testMultiprocessMainProcessDead(self):
        def spawn_main_process():
            env = MockEnv(action_dim=1, observation_dim=1, run_mode='multiprocess')
            env.start()
            while True:
                env.step(0)

        curr_process_util = psutil.Process()

        main_process = Process(target=spawn_main_process)
        main_process.start()

        # give some time to make sure everything running
        time.sleep(1.0)
        child_processes = curr_process_util.children(recursive=True)
        self.assertEqual(len(child_processes), 4)

        main_process.terminate()
        main_process.join()
        time.sleep(2.0)
        self.assertFalse(any([c.is_running() for c in child_processes]))

    def testSinglethreadMainProcessException(self):
        def spawn_main_process():
            env = MockEnv(action_dim=1, observation_dim=1, run_mode='singlethread')
            env.start()
            env.crash_flag.value = 3
            env.step(0)

        main_process = Process(target=spawn_main_process)
        main_process.start()

        time.sleep(1.0)
        main_process.join()

        curr_process_util = psutil.Process()
        child_processes = curr_process_util.children(recursive=True)
        self.assertFalse(any([c.is_running() for c in child_processes]))

    def testMultithreadMainProcessException(self):
        def spawn_main_process():
            env = MockEnv(action_dim=1, observation_dim=1, run_mode='multithread')
            env.start()
            env.crash_flag.value = 3
            env.step(0)

        main_process = Process(target=spawn_main_process)
        main_process.start()

        time.sleep(1.0)
        main_process.join()

        curr_process_util = psutil.Process()
        child_processes = curr_process_util.children(recursive=True)
        self.assertFalse(any([c.is_running() for c in child_processes]))

    def testMultiprocessMainProcessException(self):
        def spawn_main_process():
            env = MockEnv(action_dim=1, observation_dim=1, run_mode='multiprocess')
            env.start()
            env.crash_flag.value = 3
            env.step(0)

        main_process = Process(target=spawn_main_process)
        main_process.start()

        time.sleep(1.0)
        main_process.join()

        curr_process_util = psutil.Process()
        child_processes = curr_process_util.children(recursive=True)
        self.assertFalse(any([c.is_running() for c in child_processes]))

    def testSharedRandomState(self):
        env = MockEnv(action_dim=1, observation_dim=1, run_mode='multiprocess')
        initial_rand_obj = copy.deepcopy(env._rand_obj_)
        initial_values = initial_rand_obj.uniform(-1, +1, 3)
        env.start()
        for _ in range(3):
            env.step(0)
        updated_rand_obj = np.random.RandomState()
        updated_rand_obj.set_state(utils.get_random_state_from_array(env._shared_rstate_array_))
        safe_final_values = updated_rand_obj.uniform(-1, +1, 3)
        unsafe_final_values = env._rand_obj_.uniform(-1, +1, 3)
        env.step(0)
        env.close()

        assert np.all(initial_values == unsafe_final_values)
        assert np.all(initial_values != safe_final_values)
        assert np.all(safe_final_values == env._uniform_array_)


if __name__ == '__main__':
    unittest.main(buffer=True)
