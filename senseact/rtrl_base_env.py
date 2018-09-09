import time
import logging
import os
import numpy as np
from threading import Thread
from multiprocessing import Process, Value, Array

from senseact import utils
from senseact.sharedbuffer import SharedBuffer


class RTRLBaseEnv(object):
    """An abstract class representing real time RL environment.

    The class provides a Reinforcement Learning discrete
    step environment interface for tasks on real robots with real time
    flow. The class maintains time steps of fixed real time duration
    (accounting also among other things for the time an algorithm took
    to compute an action) and facilitates timely exchange of actions
    and sensory data between a robot (possbily consisting of several
    asynchronous physical devices) and a learning algorithm.
    """

    def __init__(self,
                 communicator_setups,
                 action_dim,
                 observation_dim,
                 run_mode='multiprocess',
                 dt=.1,
                 dt_tol=1e-5,
                 sleep_time=0.0001,
                 busy_loop=True,
                 random_state=None,
                 **kwargs
                ):

        """Inits RTRLBaseEnv object with task specific parameters.

        Args:
            communicator_setups: A dictionary containing configuration
                parameters for each physical device communicator.
                The form of this dictionary is as follows:
                config = {'name':{'Communicator': CommunicatorClass,
                                  'kwargs': dict()
                                 }
                }
            action_dim: An integer dimensionality of the action space
            observation_dim: An integer dimensionality of the observation space
            run_mode: A string specifying the method of parallelism between
                the agent and the environment, one of 'singlethread',
                'multithread', or 'multiprocess'.
            dt: A float timestep duration to maintain when calling 'step'
                or 'sense_wait'
            dt_tol: a float small tolerance subtracted from dt to compensate for
                OS delays when exiting sleep() in 'step' or 'sense_wait'.
            random_state: A tuple containing random state returned by
                numpy.random.RandomState().get_state(). This is to ensure
                reproducibility by reusing the same random state externally from
                an experiment script.
            sleep_time: a float representing lower bound on sleep() function
                time resolution provided by OS. For linux based OSes the
                resolution is typically ~0.001s, for Windows based OSes its ~0.01s.
            busy_loop: a boolean specifying whether to use busy loops instead
                of time.sleep() to accurately maintain short real time intervals.
        """
        assert run_mode in ['singlethread', 'multithread', 'multiprocess']
        self._run_mode = run_mode

        # Used for gym compatible step function
        self._dt = dt
        self._dt_tol = dt_tol
        self._sleep_time = sleep_time
        self._busy_loop = busy_loop

        # create random object based on passed random_state tuple
        self._rand_obj_ = np.random.RandomState()
        if random_state is None:
            random_state = self._rand_obj_.get_state()
        else:
            self._rand_obj_.set_state(random_state)
        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(random_state)
        # Ideally, the random state tuple of `_rand_obj_` needs to be copied to this `_shared_rstate_array_`
        # after every use of `_rand_obj_` for generating random numbers.
        self._shared_rstate_array_ = np.frombuffer(Array('b', rand_state_array_size).get_obj(),
                                                   dtype=rand_state_array_type)
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))
        self._reset_flag = Value('i', 0)


        self._action_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=action_dim,
            array_type='d',
            np_array_type='d',
        )

        # Contains the observation vector, with the last element being the _reward_
        self._sensation_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=observation_dim + 2,
            array_type='d',
            np_array_type='d',
        )

        # A dictionary of dictionaries, one for each communicator that is required
        self._communicator_setups = communicator_setups

        # Dictionary of all instantiated communicator processes
        # Both sensor_comms and actuator_comms would have a reference to the same
        # communicator when that communicator contains both sensors and actuators
        self._sensor_comms = {}
        self._actuator_comms = {}

        # Contains a reference to every communicator.
        # Used for terminating all communicators once when the environment is closed

        self._all_comms = {}

        # Dictionary containing actuation packets for each communicator
        self._actuation_packet_ = {}

        # Dictionary containing the number of sensor packets to read at a time
        # from each communicator
        self._num_sensor_packets = {}

        # Construct the communicators without starting
        for name, setup in communicator_setups.items():
            # Initialize communicator with the given parameters
            comm = setup['Communicator'](**setup['kwargs'])

            if comm.use_actuator:
                self._actuation_packet_[name] = np.zeros(
                    shape=comm.actuator_buffer.array_len,
                    dtype=comm.actuator_buffer.np_array_type,
                )
                self._actuator_comms[name] = comm

            if comm.use_sensor:
                if 'num_sensor_packets' in setup.keys():
                    self._num_sensor_packets[name] = setup['num_sensor_packets']
                else:
                    self._num_sensor_packets[name] = 1
                self._sensor_comms[name] = comm

            self._all_comms[name] = comm

        self._running = False

    # ===== Main interfaces =====

    def start(self):
        """Starts all manager threads and communicator processes."""
        self._running = True

        # Start the communicator process
        for comm in self._all_comms.values():
            comm.start()

        time.sleep(0.5)  # let the communicator buffer have some packets

        self._new_obs_time = time.time()

        # Create a process/thread to read and write to all communicators
        if self._run_mode == 'multithread':
            # multithread case we don't need the check, but assigning here
            # to keep the polling loop the same
            self._parent_pid = os.getppid()
            self._polling_loop = Thread(target=self._run_loop_)
            self._polling_loop.start()
        elif self._run_mode == 'multiprocess':
            self._parent_pid = os.getpid()
            self._polling_loop = Process(target=self._run_loop_)
            self._polling_loop.start()

    def sense_wait(self):
        """Performs an environment step maintaining the duration `dt`.

        This method takes care of waiting enough to achieve `dt`
        cycle time of task step, in addition to calling `sense`.
        """
        # Only allow sensing if the environment is still running
        if not self._running:
            raise Exception("Attempted to sense on a non-running environment.")

        # Wait for one time-step
        time_used = time.time() - self._new_obs_time

        if time_used > (self._dt + self._dt_tol):
            logging.warning("Agent has over-run its allocated dt, it has been {:02} since the last observation, "
                            "{:02} more than allowed".format(time_used, time_used - self._dt))

        # sleep 1 ms less than needed to make sure OS wakes thread up on time
        if self._busy_loop:
            time.sleep(max(0, self._dt - time_used - 1e-3))
            # rest of the time step spend in a busy loop
            while time.time() - self._new_obs_time < self._dt:
                continue
        else:
            time.sleep(max(0, self._dt - time_used - self._dt_tol))

        self._new_obs_time = time.time()
        next_obs, reward, done = self.sense()

        return next_obs, reward, done

    def sense(self):
        """Provides environment information to the agent.

        Returns:
            A tuple (observation, reward, done)
        """
        try:
            if self._run_mode == 'singlethread':
                return self._sense_singlethread()
            else:
                return self._read_sensation()
        except Exception as e:
            self.close()
            raise e

    def act(self, action):
        """Writes the action to the action buffer."""
        try:
            if self._run_mode == 'singlethread':
                self._act_singlethread(action)
            else:
                self._write_action(action)
        except Exception as e:
            self.close()
            raise e

    def step(self, action):
        """Optional step function for OpenAI Gym compatibility.

        Returns: a tuple (observation, reward,  {} ('info', for gym compatibility))
        """
        # Set the desired action
        self.act(action)
        # Wait for one time-step
        next_obs, reward, done = self.sense_wait()
        return next_obs, reward, done, {}

    def reset(self, blocking=True):
        """Resets the environment based on the 'run_mode'.

        Returns:
            A numpy array with observation data.
        """
        if self._run_mode == 'singlethread':
            return self._reset_singlethread()
        else:
            return self._reset_flag_update(blocking=blocking)

    def close(self):
        """Closes all manager threads and communicator processes."""
        for name, comm in self._all_comms.items():
            comm.terminate()
            comm.join()

        self._running = False

        if self._run_mode == 'multithread':
            self._polling_loop.join()
        elif self._run_mode == 'multiprocess':
            self._polling_loop.terminate()
            self._polling_loop.join()

    # ===== Methods that should to be implemented by subclass =====

    def _reset_(self):
        """Performs the reset procedure for the task.

        To be implemented by the Environment class.
        """
        raise NotImplementedError

    def _check_done(self, env_done):
        """Checks whether the episode is over.

        This method looks at the done flag coming from the environment
        and can also be overridden to add additional checks for being done
        such as number of steps exceeding some threshold
        To be implemented by the Environment class.

        Args:
            env_done: A bool flag coming from the environment signalling
                if a done condition was met
        """
        return env_done

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        """Converts robot sensory data into observation data.

        This method processes sensory data, creates an observation vector,
        computes _reward_ and whether it is done, and returns all those
        to be written into shared `sensation` buffer.

        To be implemented by the Environment class.

        Returns:
            A numpy array containig data to be written to the `sensation` buffer
        """

        raise NotImplementedError

    def _compute_actuation_(self, action, timestamp, index):
        """Creates `actuation_packet`s.

        To be implemented by the Environment class.

        Args:
            action: A numpy array containing action from the agent
            timestamp: a float with associated timestamp
            index: An integer index of a current action
        """
        raise NotImplementedError

    # ===== Common helpers for both singlethread and multithread/multiprocess =====

    def _sensor_to_sensation_(self):
        """Checks for new packets from all connected sensor communicators.

        Calls `_compute_sensation_` for each updated `sensor_buffer`, which in turn
        updates the shared `_sensation_buffer`.
        """
        for name, comm in self._sensor_comms.items():
            if comm.sensor_buffer.updated():
                sensor_window, timestamp_window, index_window = comm.sensor_buffer.read_update(self._num_sensor_packets[name])
                s = self._compute_sensation_(name, sensor_window, timestamp_window, index_window)
                self._sensation_buffer.write(s)

    def _action_to_actuator_(self):
        """Converts action to robot actuation command.

        If there is a new action, reads from the `_action` shared buffer
        and calls `_compute_actuation_`, which should form `actuation_packet`s,
        and then call `send_packets`, which writes the `actuation_packet`s
        to corresponding `actuator_buffer`s.
        Instead of computing `actuation_packet`s and sending them to communicators
        one at a time, we enforce forming all the `actuation_packet`s first before
        sending any of them so that there is minimal delay between writes
        to different communicators.
        """
        if self._action_buffer.updated():
            action, timestamp, index = self._action_buffer.read_update()
            self._compute_actuation_(action[0], timestamp, index)
            self._write_actuation_()

    def _write_actuation_(self):
        """Sends `actuation_packet`s to all connected actuation communicators."""
        for name, comm in self._actuator_comms.items():
            comm.actuator_buffer.write(self._actuation_packet_[name])

    def _read_sensation(self):
        """Converts sensation to observation vector.

        This method reads from the `sensation` shared buffer
        and returns observation, _reward_, and done information. These can then
        be read directly by the agent.
        In multithread or multiprocess run mode, `sense` is defined to be
        this method.

        Returns:
            A tuple (observation, reward, done)
        """
        sensation, _, _ = self._sensation_buffer.read_update()
        done = self._check_done(sensation[0][-1])
        return sensation[0][:-2], sensation[0][-2], done

    def _write_action(self, action):
        """Writes action to the action buffer.

        Also checks if all of the required subprocesses are still running
        before writing the action.

        Args:
            action: A numpy array containing action

        Raises:
            Exception: if any of the communicator or internal helper
            processes crashed.
        """
        self._action_buffer.write(action)

        # Only allow action if the environment is still running; we are checking after
        # writing the action because we want any delays to be between act and sense.
        if not self._running:
            raise Exception("Attempted to act on a non-running environment.")

        # check if any communicator has stopped or the polling loop has died
        if any(not comm.is_alive() for comm in self._all_comms.values()) or \
           (hasattr(self, '_polling_loop') and not self._polling_loop.is_alive()):
            logging.error("One of the environment subprocess has died, closing all processes.")
            self.close()

            raise Exception("Environment has been shutdown due to subprocess error.")

    # ===== Singlethread specific methods =====

    def _reset_singlethread(self):

        self._reset_()
        self._new_obs_time = time.time()
        obs, _, _ = self.sense()

        return obs

    def _sense_singlethread(self):
        """Implements _sense for single thread mode.

        In single thread run mode, `sense` is defined to be this method,
        where reading from communicator buffer and providing information
        to the agent occur in the same thread.
        """
        self._sensor_to_sensation_()
        return self._read_sensation()

    def _act_singlethread(self, action):
        """Processes actions in single thread mode.

        Args:
            action: A numpy array containing action.
        """
        self._write_action(action)
        self._action_to_actuator_()

    # ===== Multithread/Multiprocess specific methods =====

    def _reset_flag_update(self, blocking=True):
        """Initiates a reset on the environment.

        OpenAI gym setup requires this call to be blocking.
        """

        # Signal to the `_run_loop_` that a reset is requested
        self._reset_flag.value = 1

        # Wait until reset procedure signals that it is finished
        while blocking and self._reset_flag.value:
            time.sleep(self._sleep_time)

        self._new_obs_time = time.time()

        # Retrieve the first observation of the new episode
        obs, _, _ = self.sense()

        return obs

    def _run_loop_(self):
        """Main manager method for multithread and multiprocess modes.

        In multithread or multiprocess run mode, this method manages the passing
        of information from sensor communicators to observation, _reward_, and done buffer
        as well as from the action buffer to the actuation communicators.
        In singlethread run mode, this method is not called and the passing of information
        is handled by `sense` and `act`.
        """
        while self._running:
            # XXX on windows the parent pid stay the same even after the parent process
            # has been killed, so this only works on Linux-based OS; possible alternative
            # would be to establish a socket between to allow checking if the connection
            # is alive.
            if os.getppid() != self._parent_pid:
                logging.info("Main environment process has been closed, shutting down polling loop.")
                return

            if self._reset_flag.value:
                # Perform reset procedure defined by the environment class.
                self._reset_()
                # Signal that the reset is complete.
                # The `reset` function in the main thread may block on this flag
                self._reset_flag.value = 0

            self._sensor_to_sensation_()
            self._action_to_actuator_()
            start = time.time()
            if self._busy_loop:
                while time.time() - start < self._sleep_time:
                    continue
            else:
                time.sleep(self._sleep_time)
