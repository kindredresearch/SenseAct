import gym
import time
import numpy as np

from senseact import utils
from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.devices.dxl import dxl_mx64
from senseact.devices.dxl.dxl_setup import setups
from senseact.devices.dxl import dxl_communicator as gcomm

from math import pi
from collections import deque
from multiprocessing import Array, Value


class DxlTracker1DEnv(RTRLBaseEnv, gym.core.Env):
    """ The Dynamixel Tracker 1D Environment (DxlTracker1DEnv)

    Here, the servo tries to reach a moving target position by controlling its joints via
    position/velocity/torque commands. The goal for this task is to rotate one Dynamixel joint to
    track a target trajectory (e.g., a fixed line, sine wave, etc), which is sub-sampled at the start
    of each episode. The servo can be reset to a fixed position. In this case, the start state is
    essentially its position in the previous episode.
    """
    def __init__(self,
                 setup='dxl_tracker_default',
                 idn=9,
                 baudrate=1000000,
                 obs_history=2,
                 dt=0.01,
                 gripper_dt=0.006,
                 rllab_box=False,
                 episode_length_step=None,
                 episode_length_time=4,
                 dof=1,
                 max_torque_mag=300,
                 control_type='torque',
                 target_type='position',
                 reset_type='zero',
                 reward_type='linear',
                 delay=0,
                 dxl_dev_path='None',
                 max_velocity=5,
                 history_dt=None,
                 use_ctypes_driver=True,
                 **kwargs
                 ):
        """ Inits DxlTracker1DEnv class with task and servo specific parameters.

        Args:
            setup: A dictionary containing DXL tracker task specifications,
                such as bounding box dimensions, joint angle ranges and max load.
            idn: An integer representing the DXL ID number
            baudrate: An integer representing a baudrate to connect at
            obs_history: An integer number of sensory packets concatenated
                into a single observation vector
            dt: A float specifying duration of an environment time step
                in seconds.
            gripper_dt: A float representing DXLCommunicator cycle time
            rllab_box: A bool specifying whether to wrap environment
                action and observation spaces into an RllabBox object
                (required for off-the-shelf rllab algorithms implementations).
            episode_length_time: A float duration of an episode defined
                in seconds
            episode_length_step: An integer duration of en episode
                defined in environment steps.
            dof: an integer number of degrees of freedom
            max_torque_mag: An integer representing max possible torque command
                to be sent to the DXL devive
            control_type:
            target_type: A string specifying in what space to provide
                target coordinates, either "position" for Cartesian space
                or "angle" for joints angles space.
            reset_type: A string specifying whether to reset the arm to a
                fixed position or to a random position.
            reward_type: A string specifying the reward function,
                (e.g.,  "linear" for - d_t)
            delay: A float specifying artificial observation delay in seconds
            dxl_dev_path: A string containing the serial port address
                (e.g., /dev/ttyACM0 or /dev/ttyUSB0 on linux)
            max_velocity: A float representing the max possible velocity command
                to be sent to the DXL device
            history_dt: A float that indicates that chooses last action as action chosen
                at t - history_dt
            use_ctypes_driver: A bool. Setting it to True chooses CType-based driver.
                We found the CType-based driver to provide substantially more timely
                and precise communication compared to the pyserial-based one.
            **kwargs: Keyword arguments
        """

        self.max_temperature = 60
        self.cool_down_temperature = 50
        self.obs_history = obs_history
        self.dt = dt
        self.gripper_dt = gripper_dt

        self.max_torque_mag = np.array([max_torque_mag])
        self.max_velocity = np.array([max_velocity])

        if rllab_box:
            from rllab.spaces import Box as RlBox  # use this for rllab TRPO
            Box = RlBox
        else:
            from gym.spaces import Box as GymBox  # use this for baselines algos
            Box = GymBox

        if control_type not in ['torque']:
            raise NotImplementedError('{} control not implemented'.format(control_type))
        self.control_type = control_type

        if target_type not in ['position']:
            raise NotImplementedError('{} target not implemented'.format(target_type))
        self.target_type = target_type

        if reset_type not in ['zero', 'random' ,'none']:
            raise NotImplementedError('{} reset not implemented'.format(reset_type))
        self.reset_type = reset_type

        if reward_type not in ['linear']:
            raise NotImplementedError('{} reward not implemented'.format(reward_type))
        self.reward_type = reward_type

        if control_type == 'torque':
            self.action_low = -self.max_torque_mag
            self.action_high = +self.max_torque_mag
        elif control_type == 'velocity':
            self.action_low = -self.max_velocity
            self.action_high = +self.max_velocity

        if setup not in setups:
            raise NotImplementedError('Config not found')
        self.angle_low = setups[setup]['angles_low'][0]
        self.angle_high = setups[setup]['angles_high'][0]

        # Load value for detecting a closed gripper during reset
        self.high_load = setups[setup]['high_load'][0]

        self._present_pos_ = np.zeros((obs_history, 1))

        self._observation_space = Box(
            low=np.array(
                # list(0*np.ones(self.obs_history))  # torque enable
                # + list(0*np.ones(self.obs_history))  # alarm led
                # + list(0*np.ones(self.obs_history))  # led
                list(-pi * np.ones(self.obs_history))  # present position
                + list(self.angle_low * np.ones(2))  # target position
                + list(-np.inf * np.ones(self.obs_history))  # present speed
                # + list(-np.inf*np.ones(self.obs_history))  # present load
                # + list(0 * np.ones(self.obs_history))  # temperature
                # + list(0*np.ones(self.obs_history))  # registered
                # + list(0*np.ones(self.obs_history))  # moving
                # + list(-np.inf * np.ones(self.obs_history))  # current
                # + list(-np.inf*np.ones(self.obs_history))  # voltage
                + list(self.action_low * np.ones(self.obs_history))  # last action
            ),
            high=np.array(
                # list(1 * np.ones(self.obs_history))  # torque enable
                # + list(128 * np.ones(self.obs_history))  # alarm led
                # + list(1 * np.ones(self.obs_history))  # led
                list(pi * np.ones(self.obs_history))  # present position
                + list(self.angle_high * np.ones(2))  # target position
                + list(+np.inf * np.ones(self.obs_history))  # present speed
                # + list(+np.inf * np.ones(self.obs_history))  # present load
                # + list(255 * np.ones(self.obs_history))  # temperature
                # + list(1 * np.ones(self.obs_history))  # registered
                # + list(1 * np.ones(self.obs_history))  # moving
                # + list(+np.inf * np.ones(self.obs_history))  # current
                # + list(+np.inf * np.ones(self.obs_history))  # voltage
                + list(self.action_high * np.ones(self.obs_history))  # last action
            )
        )
        self._action_space = Box(low=self.action_low, high=self.action_high)

        if rllab_box:
            from rllab.envs.env_spec import EnvSpec
            self._spec = EnvSpec(self.observation_space, self.action_space)

        self._comm_name = 'DxlTracker1D'
        self._dxl_dev_path = dxl_dev_path
        communicator_setups = {
            self._comm_name: {
                'Communicator': gcomm.DXLCommunicator,
                'num_sensor_packets': obs_history,
                'kwargs': {
                    'idn': idn,
                    'baudrate': baudrate,
                    'sensor_dt': gripper_dt,
                    'device_path': self._dxl_dev_path,
                    'use_ctypes_driver': use_ctypes_driver,
                }
            }
        }
        super(DxlTracker1DEnv, self).__init__(
            communicator_setups=communicator_setups,
            action_dim=1,
            observation_dim=self.observation_space.shape[0],
            dt=dt,
            **kwargs
        )

        read_block = dxl_mx64.MX64.subblock('version_0', 'goal_acceleration', ret_dxl_type=use_ctypes_driver)
        self.regnames = [reg.name for reg in read_block]
        self.reg_index = dict(zip(self.regnames, range(len(self.regnames))))

        self.episode_steps = Value('i', 0)
        if episode_length_step is not None:
            assert episode_length_time is None
            self.episode_length_step = episode_length_step
            self.episode_length_time = episode_length_step * dt
        elif episode_length_time is not None:
            assert episode_length_step is None
            self.episode_length_time = episode_length_time
            self.episode_length_step = int(episode_length_time / dt)
        else:
            # TODO: should we allow a continuous behaviour case here, with no episodes?
            print("episode_length_time or episode_length_step needs to be set")
            raise AssertionError

        # Task Parameters
        self.obs_history = obs_history
        self.dof = dof
        self.delay = delay
        self.pos_range = self.angle_high - self.angle_low

        if history_dt is not None:
            self.history_dt = history_dt
            self.dt_ratio = int(history_dt/self.gripper_dt)
        else:
            self.dt_ratio = int(self.dt/self.gripper_dt)
        self.comm_episode_length_step = self.episode_length_step * self.dt_ratio

        # Default initialization
        target_pos = np.random.uniform(low=self.angle_low, high=self.angle_high)
        self.pos_range = self.angle_high - self.angle_low
        self.reset_pos_center = self.angle_high - (self.pos_range / 2.)
        self.action_range = self.action_high - self.action_low

        self._reward_ = Value('d', 0.0)
        self._reset_pos_ = Value('d', self.reset_pos_center)
        self._present_pos_ = np.frombuffer(Array('f', self.obs_history).get_obj(), dtype='float32')
        self._target_pos_ = Value('d', target_pos)
        self._temperature_ = [0] * self.obs_history
        self._action_history = deque([0] * (self.obs_history + 1), self.obs_history + 1)

        # Generate trajectory
        self.generate_trajectory()

        # Tell the dxl to do nothing (overwritting previous command)
        self.nothing_packet = np.zeros(self._actuator_comms[self._comm_name].actuator_buffer.array_len)

        # PID control gains for reset
        self.kp = 161.1444  # Proportional gain
        self.ki = 0  # Integral gain
        self.kd = 0  # Derivative gain

    def generate_trajectory(self):
        """Generate a variable speed target trajectory for the episode."""
        direction = self._rand_obj_.choice([-1, 1])
        if direction == 1:
            start_pos = self._rand_obj_.uniform(low=self.angle_low, high=self.reset_pos_center)
            self.trajectory = np.linspace(start_pos, self.angle_high, self.comm_episode_length_step)
            # self.trajectory = np.linspace(start_pos, start_pos + self.pos_range/2., self.comm_episode_length_step)
        else:
            start_pos = self._rand_obj_.uniform(low=self.reset_pos_center, high=self.angle_high)
            self.trajectory = np.linspace(start_pos, self.angle_low, self.comm_episode_length_step)
            # self.trajectory = np.linspace(start_pos, start_pos - self.pos_range/2., self.comm_episode_length_step)

    def _reset_(self):
        """ Resets the environment episode.

        Moves the DXL to either fixed reference or random position and
        generates a new target within a bounding box.
        """
        print("Resetting")
        self.episode_steps.value = 0
        self.generate_trajectory()
        self._target_pos_.value = self.trajectory[0]
        present_pos = 0.0

        if self.reset_type != 'none':
            if self.reset_type == 'zero':
                self._reset_pos_.value = self.reset_pos_center
            elif self.reset_type == 'random':
                self._reset_pos_.value = self._rand_obj_.uniform(low=self.angle_low, high=self.angle_high)

            error_prior = 0
            integral = 0

            # Once in the correct regime, the `present_pos` values can be trusted
            start_time = time.time()
            while time.time() - start_time < 5:
                if self._sensor_comms[self._comm_name].sensor_buffer.updated():
                    sensor_window, timestamp_window, index_window = self._sensor_comms[
                        self._comm_name].sensor_buffer.read_update(1)
                    present_pos = sensor_window[0][self.reg_index['present_pos']]
                    current_temperature = sensor_window[0][self.reg_index['temperature']]

                    if current_temperature > self.cool_down_temperature:
                        print("Starting to overheat. sleep for a few seconds")
                        time.sleep(10)

                    error = self._reset_pos_.value - present_pos
                    if abs(error) > 0.017:  # ~1 deg
                        integral = integral + (error * self.gripper_dt)
                        derivative = (error - error_prior) / self.gripper_dt
                        action = self.kp * error + self.ki * integral + self.kd * derivative
                        error_prior = error
                    else:
                        break

                    self._actuator_comms[self._comm_name].actuator_buffer.write(action)
                    time.sleep(0.001)
        else:
            flag = 0
            while not flag:
                if self._sensor_comms[self._comm_name].sensor_buffer.updated():
                    sensor_window, timestamp_window, index_window = self._sensor_comms[
                        self._comm_name].sensor_buffer.read_update(1)
                    present_pos = sensor_window[0][self.reg_index['present_pos']]
                    self._reset_pos_.value = present_pos
                    flag = 1

        self._actuator_comms[self._comm_name].actuator_buffer.write(0)
        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))
        time.sleep(0.1)  # Give the shared buffer time to get updated and prevent false episode done conditions
        print("Reset done. Gripper pos: {}".format(present_pos))

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        """ Creates and saves an observation vector based on sensory data.

        For DXL reacher environments the observation vector is a concatenation of:
            - current joint angle positions;
            - current joint angle velocities;
            - target joint angle position;
            - previous action;
            - temperature (optional)
            - current (optional)

        Args:
            name: a string specifying the name of a communicator that
                received given sensory data.
            sensor_window: a list of latest sensory observations stored in
                communicator sensor buffer. the length of list is defined by
                obs_history parameter.
            timestamp_window: a list of latest timestamp values stored in
                communicator buffer.
            index_window: a list of latest sensor index values stored in
                communicator buffer.

        Returns:
            A numpy array containing concatenated [observation, reward, done]
            vector.
        """
        self._torque_enable_ = np.array(
            [sensor_window[i][self.reg_index['torque_enable']] for i in range(self.obs_history)])
        present_pos = np.array(
            [sensor_window[i][self.reg_index['present_pos']] for i in range(self.obs_history)])
        np.copyto(self._present_pos_, present_pos)
        self._present_speed_ = np.array(
            [sensor_window[i][self.reg_index['present_speed']] for i in range(self.obs_history)])
        self._current_ = np.array([sensor_window[i][self.reg_index['current']] for i in range(self.obs_history)])
        self._temperature_ = np.array(
            [sensor_window[i][self.reg_index['temperature']] for i in range(self.obs_history)])
        traj_ind = min(self.episode_steps.value*self.dt_ratio,
                       self.comm_episode_length_step-1)
        self._target_pos_.value = self.trajectory[traj_ind]
        target_obs = [
            self.trajectory[max(0, traj_ind-self.dt_ratio)],
            self.trajectory[traj_ind],
        ]

        self._reward_.value = self._compute_reward()
        done = [0]
        last_actions = list(self._action_history)
        last_actions_obs = np.array(last_actions[-self.obs_history:]).flatten()

        return np.concatenate(
            (
                self._present_pos_,
                [target_obs[0]],
                [target_obs[1]],
                self._present_speed_,
                # self._temperature_,
                # self._current_,
                self.scale_action(last_actions_obs),
                np.array([self._reward_.value]),
                done
            )
        )

    def _compute_actuation_(self, action, timestamp, index):
        """ Creates and sends actuation packets to the communicator.

        Computes actuation commands based on agent's action and
        control type and writes actuation packets to the
        communicators' actuation buffers. In case of  angle joints
        safety limits being violated overwrites agent's
        actions with actuations that return the DXL back within the box.

        Args:
            action: a numpoy array containing agent's action
            timestamp: a float containing action timestamp
            index: an integer containing action index
        """
        if self._temperature_[-1] < self.max_temperature:
            if self._present_pos_[-1] < self.angle_low:
                self._actuation_packet_[self._comm_name] = self.max_torque_mag // 2
            elif self._present_pos_[-1] > self.angle_high:
                self._actuation_packet_[self._comm_name] = -self.max_torque_mag // 2
            else:
                self._actuation_packet_[self._comm_name] = action
            self._action_history.append(action)
        else:
            self._actuator_comms[self._comm_name].actuator_buffer.write(self.nothing_packet)
            raise Exception('Operating temperature of the dynamixel device exceeded 50°C \n'
                            'Use the device once it cools down!')

    def _compute_reward(self):
        """ Computes reward at a given time step.

        Returns:
            A float reward.
        """
        reward = 0

        # if self._temperature_[-1] > self.cool_down_temperature:
        #     reward -= 2 * pi
        if self.reward_type == 'linear':
            goal_pos = self._target_pos_.value
            present_pos = self._present_pos_
            reward -= abs(goal_pos - present_pos[-1])
        reward *= self.dt / 0.04
        return np.array([reward])

    def _check_done(self, env_done):
        """ Checks whether the episode is over.

        Args:
            env_done:  a bool specifying whether the episode should be ended.

        Returns:
            A bool specifying whether the episode is over.
        """
        self.episode_steps.value += 1
        if self.episode_steps.value >= self.episode_length_step or env_done:
            self._actuator_comms[self._comm_name].actuator_buffer.write(self.nothing_packet)
            done = True
        else:
            done = False
        return np.array([done])

    def reset(self, blocking=True):
        """ Resets the arm, optionally blocks the environment until done. """
        ret = super(DxlTracker1DEnv, self).reset(blocking=blocking)
        self.episode_steps.value = 0
        return ret

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def scale_angle(self, x, oldMin=0., oldMax=1., newMin = 0., newMax = 1.):
        """ Scale angle to a different range

        Args:
            x: Angle in radians
            oldMin: A float representing original minimum value
            oldMax: A float representing original maximum value
            newMin: A float representing new minimum value
            newMax: A float representing new maximum value

        Returns:
            Scaled angle converted from the old range to new range
        """
        oldRange = oldMax - oldMin
        newRange = newMax - newMin
        x_new = ((x - oldMin) * newRange / oldRange) + newMin
        return x_new

    def scale_action(self, action):
        return (2*(action - self.action_low)/ self.action_range) - 1.

    def terminate(self):
        super(DxlTracker1DEnv, self).close()

    def render(self, **kwargs):
        return
