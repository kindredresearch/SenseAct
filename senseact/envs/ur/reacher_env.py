# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import time
import gym
import sys
from multiprocessing import Array, Value

from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.devices.ur import ur_utils
from senseact.devices.ur.ur_setups import setups
from senseact.sharedbuffer import SharedBuffer
from senseact import utils


class ReacherEnv(RTRLBaseEnv, gym.core.Env):
    """A class implementing UR5 Reacher2D and Reacher6D environments.

    A UR5 reacher task consists in reaching with an arm end-effector a given
    location in Cartesian space using low level UR5 joint control commands.
    The class implements a fixed duration episodic reacher task where the target is
    generated randomly at the beginning of each episode. The class implements
    both, position and velocity control. For each of those it implements direct,
    first and second derivative control. The class implements safety boundaries
    handling defined as a box in a Cartesian space. The class also implements reset
    function, which moves the arm into a predefined initial position at the end
    of each episode and generates a random target.
    """
    def __init__(self,
                 setup,
                 host=None,
                 dof=6,
                 control_type='position',
                 derivative_type='none',
                 target_type='position',
                 reset_type='random',
                 reward_type='linear',
                 deriv_action_max=10,
                 first_deriv_max=10,  # used only with second derivative control
                 vel_penalty=0,
                 obs_history=1,
                 actuation_sync_period=1,
                 episode_length_time=None,
                 episode_length_step=None,
                 rllab_box = False,
                 servoj_t=ur_utils.COMMANDS['SERVOJ']['default']['t'],
                 servoj_gain=ur_utils.COMMANDS['SERVOJ']['default']['gain'],
                 speedj_a=ur_utils.COMMANDS['SPEEDJ']['default']['a'],
                 speedj_t_min=ur_utils.COMMANDS['SPEEDJ']['default']['t_min'],
                 movej_t=2, # used for resetting
                 accel_max=None,
                 speed_max=None,
                 dt=0.008,
                 delay=0.0,  # to simulate extra delay in the system
                 **kwargs):
        """Inits ReacherEnv class with task and robot specific parameters.

        Args:
            setup: a dictionary containing UR5 reacher task specifications,
                such as safety box dimensions, joint angle ranges, boundary
                on the arm speed, UR5 Controller IP address etc
                (see senseact.devices.ur.ur_setups for examples).
            host: a string specifying UR5 IP address or None
            dof: an integer number of degrees of freedom, either 2 for
                Reacher2D or 6 for Reacher6D
            control_type: a string specifying UR5 control type, either
                position (using UR5 servoJ commands) or velocity
                (using UR5 speedJ commands)
            derivative_type: a string specifying what type of derivative
                control to use, either "none", "first" or "seconds"
            target_type: a string specifying in what space to provide
                target coordinates, either "position" for Cartesian space
                or "angle" for joints angles space.
            reset_type: a string specifying whether to reset the arm to a
                fixed position or to a random position.
            reward_type: a string specifying the reward function, either
                "linear" for - d_t, or "precision" for  - d_t + exp^( - d_t^2)
            deriv_action_max: a float specifying maximum value of an action
                for derivative control
            first_deriv_max: a float specifying maximum value of a first
                derivative of action if derivative_type =="second"
            vel_penalty: a float specifying the weight of a velocity
                penalty term in the reward function.
            obs_history: an integer number of sensory packets concatenated
                into a single observation vector
            actuation_sync_period: a bool specifying whether to synchronize
                sending actuation commands to UR5 with receiving sensory
                packets from UR5 (must be true for smooth UR5 operation).
            episode_length_time: a float duration of an episode defined
                in seconds
            episode_length_step: an integer duration of en episode
                defined in environment steps.
            rllab_box: a bool specifying whether to wrap environment
                action and observation spaces into an RllabBox object
                (required for off-the-shelf rllab algorithms implementations).
            servoj_t: a float specifying time parameter of a UR5
                servoj command.
            servoj_gain: a float specifying gain parameter of a UR5
                servoj command.
            speedj_a: a float specifying acceleration parameter of a UR5
                speedj command.
            speedj_t_min: a float specifying t_min parameter of a UR5
                speedj command.
            movej_t: a float specifying time parameter of a UR5
                speedj command.
            accel_max: a float specifying maximum allowed acceleration
                of UR5 arm joints. If None, a value from setup is used.
            speed_max: a float specifying maximum allowed speed of UR5 joints.
                If None, a value from setup is used.
            dt: a float specifying duration of an environment time step
                in seconds.
            delay: a float specifying artificial observation delay in seconds

        """


        # Check that the task parameters chosen are implemented in this class
        assert dof in [2, 6]
        assert control_type in ['position', 'velocity', 'acceleration']
        assert derivative_type in ['none', 'first', 'second']
        assert target_type in ['position', 'angle']
        assert reset_type in ['random', 'zero', 'none']
        assert actuation_sync_period >= 0

        if episode_length_step is not None:
            assert episode_length_time is None
            self._episode_length_step = episode_length_step
            self._episode_length_time = episode_length_step * dt
        elif episode_length_time is not None:
            assert episode_length_step is None
            self._episode_length_time = episode_length_time
            self._episode_length_step = int(episode_length_time / dt)
        else:
            #TODO: should we allow a continuous behaviour case here, with no episodes?
            print("episode_length_time or episode_length_step needs to be set")
            raise AssertionError

        # Task Parameters
        self._host = setups[setup]['host'] if host is None else host
        self._obs_history = obs_history
        self._dof = dof
        self._control_type = control_type
        self._derivative_type = derivative_type
        self._target_type = target_type
        self._reset_type = reset_type
        self._reward_type = reward_type
        self._vel_penalty = vel_penalty # weight of the velocity penalty
        self._deriv_action_max = deriv_action_max
        self._first_deriv_max = first_deriv_max
        self._delay = delay
        if accel_max==None:
            accel_max = setups[setup]['accel_max']
        if speed_max==None:
            speed_max = setups[setup]['speed_max']
        if self._dof == 6:
            self._joint_indices = [0, 1, 2, 3, 4, 5]
            self._end_effector_indices = [0, 1, 2]
        elif self._dof == 2:
            self._joint_indices = [1, 2]
            self._end_effector_indices = [1, 2]

        # Arm/Control/Safety Parameters
        self._end_effector_low = setups[setup]['end_effector_low']
        self._end_effector_high = setups[setup]['end_effector_high']
        self._angles_low = setups[setup]['angles_low'][self._joint_indices]
        self._angles_high = setups[setup]['angles_high'][self._joint_indices]
        self._speed_low = -np.ones(self._dof) * speed_max
        self._speed_high = np.ones(self._dof) * speed_max
        self._accel_low = -np.ones(self._dof) * accel_max
        self._accel_high = np.ones(self._dof) * accel_max

        self._box_bound_buffer = setups[setup]['box_bound_buffer']
        self._angle_bound_buffer = setups[setup]['angle_bound_buffer']
        self._q_ref = setups[setup]['q_ref']
        self._ik_params = setups[setup]['ik_params']

        # State Variables
        self._q_ = np.zeros((self._obs_history, self._dof))
        self._qd_ = np.zeros((self._obs_history, self._dof))

        self._episode_steps = 0

        self._pstop_time_ = None
        self._pstop_times_ = []
        self._safety_mode_ = ur_utils.SafetyModes.NONE
        self._max_pstop = 10
        self._max_pstop_window = 600
        self._clear_pstop_after = 2
        self._x_target_ = np.frombuffer(Array('f', 3).get_obj(), dtype='float32')
        self._x_ = np.frombuffer(Array('f', 3).get_obj(), dtype='float32')
        self._reward_ = Value('d', 0.0)

        if self._target_type == 'position':
            if self._dof == 2:
                self._target_ = np.zeros((2))
                self._target_low = self._end_effector_low[self._end_effector_indices]
                self._target_high = self._end_effector_high[self._end_effector_indices]
            elif self._dof == 6:
                self._target_ = np.zeros((3))
                self._target_low = self._end_effector_low
                self._target_high = self._end_effector_high
        elif self._target_type == 'angle':
            self._target_ = np.zeros((self._dof))
            self._target_low = self._angles_low
            self._target_high = self._angles_high

        # Set up action and observation space

        if self._derivative_type== 'second' or self._derivative_type== 'first':
            self._action_low = -np.ones(self._dof) * self._deriv_action_max
            self._action_high = np.ones(self._dof) * self._deriv_action_max
        else: # derivative_type=='none'
            if self._control_type == 'position':
                self._action_low = self._angles_low
                self._action_high = self._angles_high
            elif self._control_type == 'velocity':
                self._action_low = self._speed_low
                self._action_high = self._speed_high
            elif self._control_type == 'acceleration':
                self._action_low = self._accel_low
                self._action_high = self._accel_high

        # TODO: is there a nicer way to do this?
        if rllab_box:
            from rllab.spaces import Box as RlBox  # use this for rllab TRPO
            Box = RlBox
        else:
            from gym.spaces import Box as GymBox  # use this for baselines algos
            Box = GymBox

        self._observation_space = Box(
            low=np.array(
                list(self._angles_low * self._obs_history)  # q_actual
                + list(-np.ones(self._dof * self._obs_history))  # qd_actual
                + list(self._target_low)  # target
                + list(-self._action_low)  # previous action in cont space
            ),
            high=np.array(
                list(self._angles_high * self._obs_history)  # q_actual
                + list(np.ones(self._dof * self._obs_history))  # qd_actual
                + list(self._target_high)  # target
                + list(self._action_high)    # previous action in cont space
            )
        )


        self._action_space = Box(low=self._action_low, high=self._action_high)

        if rllab_box:
            from rllab.envs.env_spec import EnvSpec
            self._spec = EnvSpec(self.observation_space, self.action_space)

        # Only used with second derivative control
        self._first_deriv_ = np.zeros(len(self.action_space.low))

        # Communicator Parameters
        communicator_setups = {'UR5':
                                   {
                                    'num_sensor_packets': obs_history,

                                    'kwargs': {'host': self._host,
                                               'actuation_sync_period': actuation_sync_period,
                                               'buffer_len': obs_history + SharedBuffer.DEFAULT_BUFFER_LEN,
                                               }
                                    }
                               }
        if self._delay > 0:
            from senseact.devices.ur.ur_communicator_delay import URCommunicator
            communicator_setups['UR5']['kwargs']['delay'] = self._delay
        else:
            from senseact.devices.ur.ur_communicator import URCommunicator
        communicator_setups['UR5']['Communicator'] = URCommunicator

        super(ReacherEnv, self).__init__(communicator_setups=communicator_setups,
                                         action_dim=len(self.action_space.low),
                                         observation_dim=len(self.observation_space.low),
                                         dt=dt,
                                         **kwargs)

        # The last action
        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)

        # Defining packet structure for quickly building actions
        self._reset_packet = np.ones(self._actuator_comms['UR5'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        self._reset_packet[0] = ur_utils.COMMANDS['MOVEJ']['id']
        self._reset_packet[1:1 + 6] = self._q_ref
        self._reset_packet[-2] = movej_t

        self._servoj_packet = np.ones(self._actuator_comms['UR5'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        self._servoj_packet[0] = ur_utils.COMMANDS['SERVOJ']['id']
        self._servoj_packet[1:1 + 6] = self._q_ref
        self._servoj_packet[-3] = servoj_t
        self._servoj_packet[-1] = servoj_gain

        self._speedj_packet = np.ones(self._actuator_comms['UR5'].actuator_buffer.array_len) * ur_utils.USE_DEFAULT
        self._speedj_packet[0] = ur_utils.COMMANDS['SPEEDJ']['id']
        self._speedj_packet[1:1 + 6] = np.zeros((6,))
        self._speedj_packet[-2] = speedj_a
        self._speedj_packet[-1] = speedj_t_min

        # Tell the arm to do nothing (overwritting previous command)
        self._nothing_packet = np.zeros(self._actuator_comms['UR5'].actuator_buffer.array_len)

        self._pstop_unlock_packet = np.zeros(self._actuator_comms['UR5'].actuator_buffer.array_len)
        self._pstop_unlock_packet[0] = ur_utils.COMMANDS['UNLOCK_PSTOP']['id']


        # self.info['reward_dist'] = 0
        # self.info['reward_vel'] = 0

    def _reset_(self):
        """Resets the environment episode.

        Moves the arm to either fixed reference or random position and
        generates a new target within a safety box.
        """
        print("Resetting")

        self._q_target_, x_target = self._pick_random_angles_()
        np.copyto(self._x_target_, x_target)
        if self._target_type == 'position':
            self._target_ = self._x_target_[self._end_effector_indices]
        elif self._target_type == 'angle':
            self._target_ = self._q_target_
        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)
        self._cmd_prev_ = np.zeros(len(self._action_low))  # to be used with derivative control of velocity
        if self._reset_type != 'none':
            if self._reset_type == 'random':
                reset_angles, _ = self._pick_random_angles_()
            elif self._reset_type == 'zero':
                reset_angles = self._q_ref[self._joint_indices]
            self._reset_arm(reset_angles)

        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))

        print("Reset done")

    def _pick_random_angles_(self):
        """Generates a set of random angle positions for each joint."""
        movej_q = self._q_ref.copy()
        while True:
            reset_angles = self._rand_obj_.uniform(self._angles_low, self._angles_high)
            movej_q[self._joint_indices] = reset_angles
            inside_bound, inside_buffer_bound, mat, xyz = self._check_bound(movej_q)
            if inside_buffer_bound:
                break
        return reset_angles, xyz

    def _reset_arm(self, reset_angles):
        """Sends reset packet to communicator and sleeps until executed."""
        self._reset_packet[1:1 + 6][self._joint_indices] = reset_angles
        self._actuator_comms['UR5'].actuator_buffer.write(self._reset_packet)
        time.sleep(max(self._reset_packet[-2] * 1.5, 2.0))


    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        """Creates and saves an observation vector based on sensory data.

        For reacher environments the observation vector is a concatenation of:
            - current joint angle positions;
            - current joint angle velocities;
            - diference between current end-effector and target
              Cartesian coordinates;
            - previous action;

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
        index_end = len(sensor_window)
        index_start = index_end - self._obs_history
        self._q_ = np.array([sensor_window[i]['q_actual'][0] for i in range(index_start,index_end)])
        self._qt_ = np.array([sensor_window[i]['q_target'][0] for i in range(index_start,index_end)])
        self._qd_ = np.array([sensor_window[i]['qd_actual'][0] for i in range(index_start,index_end)])
        self._qdt_ = np.array([sensor_window[i]['qd_target'][0] for i in range(index_start,index_end)])
        self._qddt_ = np.array([sensor_window[i]['qdd_target'][0] for i in range(index_start,index_end)])

        self._current_ = np.array([sensor_window[i]['i_actual'][0] for i in range(index_start,index_end)])
        self._currentt_ = np.array([sensor_window[i]['i_target'][0] for i in range(index_start,index_end)])
        self._currentc_ = np.array([sensor_window[i]['i_control'][0] for i in range(index_start,index_end)])
        self._mt_ = np.array([sensor_window[i]['m_target'][0] for i in range(index_start,index_end)])
        self._voltage_ = np.array([sensor_window[i]['v_actual'][0] for i in range(index_start,index_end)])

        self._safety_mode_ = np.array([sensor_window[i]['safety_mode'][0] for i in range(index_start,index_end)])

        #TODO: should there be checks for safety modes greater than pstop here, and exit if found?

        # Compute end effector position
        x = ur_utils.forward(sensor_window[-1]['q_actual'][0], self._ik_params)[:3, 3]
        np.copyto(self._x_, x)

        if self._target_type == 'position':
            self._target_diff_ = self._x_[self._end_effector_indices] - self._target_
        elif self._target_type == 'angle':
            self._target_diff_ = self._q_[-1, self._joint_indices] - self._target_

        self._reward_.value = self._compute_reward_()
        # TODO: use the correct obs that matches the observation_space
        return np.concatenate((self._q_[:, self._joint_indices].flatten(),
                               self._qd_[:, self._joint_indices].flatten() / self._speed_high,
                               self._target_diff_,
                               self._action_ / self._action_high,
                               [self._reward_.value],
                               [0]))

    def _compute_actuation_(self, action, timestamp, index):
        """Creates and sends actuation packets to the communicator.

        Computes actuation commands based on agent's action and
        control type and writes actuation packets to the
        communicators' actuation buffers. In case of safety box or
        angle joints safety limits being violated overwrites agent's
        actions with actuations that return the arm back within the box.
        Clears p-stops if any.

        Args:
            action: a numpoy array containing agent's action
            timestamp: a float containing action timestamp
            index: an integer containing action index
        """
        if self._safety_mode_ == ur_utils.SafetyModes.NORMAL or \
            self._safety_mode_ == ur_utils.SafetyModes.NONE:
                self.pstop_time = None
        elif self._safety_mode_ == ur_utils.SafetyModes.REDUCED:
            self.pstop_time = None
        elif self._safety_mode_ == ur_utils.SafetyModes.PROTECTIVE_STOP:
            if self._pstop_time_ is None:
                print("Encountered p-stop")
                self._pstop_time_ = time.time()
                self._pstop_times_.append(self._pstop_time_)
                # Check to see if too many p-stops occurred within a short time window
                if len(self._pstop_times_) > self._max_pstop:
                    if self._pstop_time_ - self._pstop_times_[-self._max_pstop] < self._max_pstop_window:
                        print("Too many p-stops encountered, closing environment")
                        print('Greater than {0} p-stops encountered within {1} seconds'.format(
                                    self._max_pstop, self._max_pstop_window))
                        self.close()
                        sys.exit(1)
            elif time.time() > self._pstop_time_ + self._clear_pstop_after:  # XXX
                print("Unlocking p-stop")
                self._actuation_packet_['UR5'] = self._pstop_unlock_packet
                return
            self._actuation_packet_['UR5'] = self._nothing_packet
            return
        else:
            print('Fatal UR5 error: safety_mode={}'.format(self._safety_mode_))
            self.close()
        self._action_ = action
        action = np.clip(action, self._action_low, self._action_high)
        self.return_point = None   # a point within the box to which to return when out of box bounds
        self.escaped_the_box = False  # whether got outside of the box
        if self._derivative_type== 'none':
            self._cmd_ = action
        else:
            # decide direct_prev for each control type
            if self._control_type== 'position':
                direct_prev = self._q_[-1, self._joint_indices]
            elif self._control_type== 'velocity':
                direct_prev = self._cmd_prev_
            elif self._control_type== 'acceleration':
                direct_prev = self._cmd_prev_  # qddt should not be used
            # decide increment for each derivative type
            if self._derivative_type == 'first':
                increment = action
            elif self._derivative_type == 'second':
                increment = self._first_deriv_ = np.clip(self._first_deriv_ + action * self.dt,
                                                         -self._first_deriv_max, self._first_deriv_max)
            self._cmd_ = direct_prev + increment * self.dt
        if self._control_type == 'position':
            self._cmd_ = np.clip(self._cmd_, self._angles_low, self._angles_high)
            self._servoj_packet[1:1 + 6][self._joint_indices] = self._cmd_
            self._actuation_packet_['UR5'] = self._servoj_packet
        elif self._control_type == 'velocity':
            self._cmd_ = np.clip(self._cmd_, self._speed_low, self._speed_high)
            self._cmd_prev_ = self._cmd_
            self._speedj_packet[1:1 + 6][self._joint_indices] = self._cmd_
            self._actuation_packet_['UR5'] = self._speedj_packet
        elif self._control_type == 'acceleration':
            old_cmd = self._cmd_
            self._cmd_ = np.clip(self._cmd_, self._accel_low, self._accel_high)
            self._cmd_prev_ = self._cmd_.copy()
            self._cmd_, self._accel_val_ = self._accel_to_speedj(self._cmd_)
            self._cmd_ = np.clip(self._cmd_, self._speed_low, self._speed_high)
            self._speedj_packet[1:1 + 6][self._joint_indices] = self._cmd_
            self._speedj_packet[-2] = self._accel_val_
            self._actuation_packet_['UR5'] = self._speedj_packet
        if self._control_type in ["acceleration", "velocity"]:
            self._handle_bounds_speedj()
        elif self._control_type == "position":
            self._handle_bounds_servoj()

    def _handle_bounds_servoj(self):
        """Makes sure safety boundaries are respected for position control.

        Checks if safety boundaries are respected and computes position
        control actuations that return the arm back to safety
        in case they aren't.
        """
        inside_bound, inside_buffer_bound, mat, xyz = self._check_bound(self._qt_[-1])
        inside_angle_bound = np.all(self._angles_low <= self._qt_[-1, self._joint_indices]) and \
                             np.all(self._qt_[-1, self._joint_indices] <= self._angles_high)

        if inside_bound and inside_angle_bound:
            self.return_point = None
            self.escaped_the_box = False
            return

        if self.return_point is None:
            # we are outside the bounds and return point wasn't computed yet
            self.escaped_the_box = True
            if inside_bound and not inside_angle_bound:
                print("outside of angle bound")
                self.rel_indices = self._joint_indices
                self._cmd_ = self._q_[0][self._joint_indices]
                self._cmd_ = np.clip(self._cmd_, self._angles_low + self._angle_bound_buffer,
                                     self._angles_high - self._angle_bound_buffer)
                # a point within the box to which we will be returning
                self.return_point = self._cmd_.copy()
                # Speed at which arm approaches the boundary. The faster this speed,
                # the larger opposite acceleration we need to apply in order to slow down
                self.init_boundary_speed = np.max(np.abs(self._qd_.copy()))

            else:
                print("outside box bound")
                xyz = np.clip(xyz, self._end_effector_low + self._box_bound_buffer,
                              self._end_effector_high - self._box_bound_buffer)
                mat[:3, 3] = xyz
                ref_pos = self._q_ref.copy()
                ref_pos[self._joint_indices] = self._q_[-1, self._joint_indices]
                solutions = ur_utils.inverse_near(mat, wrist_desired=self._q_ref[-1], ref_pos=ref_pos,
                                                  params=self._ik_params)
                servoj_q = self._q_ref.copy()
                if len(solutions) == 0:
                    servoj_q[self._joint_indices] = self._q_[-1, self._joint_indices]
                else:
                    servoj_q[self._joint_indices] = solutions[0][self._joint_indices]
                self.return_point = servoj_q[self._joint_indices]
                self.init_boundary_speed = np.max(np.abs(self._qd_.copy()))
                # Speed at which arm approaches the boundary. The faster this speed,
                # the larger opposite acceleration we need to apply in order to slow down
                self.init_boundary_speed = np.max(np.abs(self._qd_.copy()))
                # if return point is already computed, keep going to it, no need
                # to recompute it at every time step
        self._cmd_ = self.return_point - self._q_[0][self._joint_indices]
        # Take the direction to return point and normalize it to have norm 0.1
        self._cmd_ /= np.linalg.norm(self._cmd_) / 0.1
        self._speedj_packet[1:1 + 6][self._joint_indices] = self._cmd_
        # This acceleration guarantees that we won't move beyond
        # the bounds by more than 0.05 radian on each joint. This
        # follows from kinematics equations.
        accel_to_apply = np.max(np.abs(self._qd_)) * self.init_boundary_speed / 0.05

        # self.boundary_packet[1:1 + 6][self.joint_indices] = self.return_point
        # self.actuator_comms['UR5'].actuator_buffer.write(self.reset_packet)
        # time.sleep(1.0)
        self._speedj_packet[-2] = np.clip(accel_to_apply, 2.0, 5.0)
        self._actuation_packet_['UR5'] = self._speedj_packet
        self._cmd_.fill(0.0)
        self._cmd_prev_.fill(0.0)
        self._first_deriv_.fill(0.0)

    def _handle_bounds_speedj(self):
        """Makes sure safety boundaries are respected for velocity control.

        Checks if safety boundaries are respected and computes velocity
        control actuations that return the arm back to safety
        in case they aren't.
        """
        inside_bound, inside_buffer_bound, mat, xyz = self._check_bound(self._qt_[-1])
        inside_angle_bound = np.all(self._angles_low <= self._qt_[-1, self._joint_indices]) and \
                             np.all(self._qt_[-1, self._joint_indices] <= self._angles_high)
        if inside_bound:
            self.return_point = None
        if inside_angle_bound:
            self.angle_return_point = False
        if not inside_bound:
            if self.return_point is None:
                # we are outside the bounds and return point wasn't computed yet
                self.escaped_the_box = True
                print("outside box bound")
                xyz = np.clip(xyz, self._end_effector_low + self._box_bound_buffer,
                              self._end_effector_high - self._box_bound_buffer)
                mat[:3, 3] = xyz
                ref_pos = self._q_ref.copy()
                ref_pos[self._joint_indices] = self._q_[-1, self._joint_indices]
                solutions = ur_utils.inverse_near(mat, wrist_desired=self._q_ref[-1], ref_pos=ref_pos,
                                                  params=self._ik_params)
                servoj_q = self._q_ref.copy()
                if len(solutions) == 0:
                    servoj_q[self._joint_indices] = self._q_[-1, self._joint_indices]
                else:
                    servoj_q[self._joint_indices] = solutions[0][self._joint_indices]
                self.return_point = servoj_q[self._joint_indices]
                self.init_boundary_speed = np.max(np.abs(self._qd_.copy()))
                # Speed at which arm approaches the boundary. The faster this speed,
                # the larger opposite acceleration we need to apply in order to slow down
                self.init_boundary_speed = np.max(np.abs(self._qd_.copy()))
                # if return point is already computed, keep going to it, no need
                # to recompute it at every time step
            self._cmd_ = self.return_point - self._q_[0][self._joint_indices]
            # Take the direction to return point and normalize it to have norm 0.1
            self._cmd_ /= np.linalg.norm(self._cmd_) / 0.1
            self._speedj_packet[1:1 + 6][self._joint_indices] = self._cmd_
            # This acceleration guarantees that we won't move beyond
            # the bounds by more than 0.05 radian on each joint. This
            # follows from kinematics equations.
            accel_to_apply = np.max(np.abs(self._qd_)) * self.init_boundary_speed / 0.05

            # self.boundary_packet[1:1 + 6][self.joint_indices] = self.return_point
            # self.actuator_comms['UR5'].actuator_buffer.write(self.reset_packet)
            # time.sleep(1.0)
            self._speedj_packet[-2] = np.clip(accel_to_apply, 2.0, 5.0)
            self._actuation_packet_['UR5'] = self._speedj_packet
            self._cmd_.fill(0.0)
            self._cmd_prev_.fill(0.0)
            self._first_deriv_.fill(0.0)

        elif not inside_angle_bound:
            # if return point is already computed, keep going to it, no need
            self.rel_indices = self._joint_indices
            cur_pos = self._q_[0][self._joint_indices]
            clipped_pos = np.clip(cur_pos, self._angles_low + self._angle_bound_buffer,
                                  self._angles_high - self._angle_bound_buffer)
            # a point within the box to which we will be returning
            affected_joints = np.where(clipped_pos != cur_pos)
            if not self.angle_return_point:
                print("outside of angle bound on joints %r" % (list(affected_joints[0])))
                self.angle_return_point = True
            self._cmd_[affected_joints] = np.sign(clipped_pos - cur_pos)[affected_joints]*np.max(np.abs(self._cmd_))
            self._speedj_packet[1:1 + 6][self._joint_indices] = self._cmd_
            self._actuation_packet_['UR5'] = self._speedj_packet

    def _accel_to_speedj(self, accel_vec):
        """Converts accelj command to speedj command.

        Args:
            accel_vec: a numpy array containing accelj command.

        Returns:
            A tuple (speed_vec, accel_val) where speed_vec is a speed
            vector for a speedj command, accel_val is an acceleration
            parameter of a speedj command.
        """
        if np.allclose(accel_vec, np.zeros(len(accel_vec))):
            return accel_vec, 0
        accel_val = np.max(np.abs(accel_vec))
        speed_vec = accel_vec * self._speed_high[0] / np.max(np.abs(accel_vec))
        return speed_vec, accel_val

    def _check_bound(self, q):
        """Checks whether given arm joints configuration is within box.

        Args:
            q: a numpy array of joints angle positions.

        Returns:
            A tuple (inside_bound, inside_buffer_bound, mat, xyz), where
            inside_bound is a bool specifying whether the arm is within
            safety bounds, inside_buffer_bound is a bool specifying
            whether the arm is within bounds and is at least at buffer
            distance to the closest bound, mat is a 4x4 position matrix
            returned by solving forward kinematics equations, xyz are
            the x,y,z coordinates of an end-effector in a Cartesian space.
        """
        mat = ur_utils.forward(q, self._ik_params)
        xyz = mat[:3, 3]
        inside_bound = np.all(self._end_effector_low <= xyz) and np.all(xyz <= self._end_effector_high)
        inside_buffer_bound = (np.all(self._end_effector_low + self._box_bound_buffer <= xyz) and \
                               np.all(xyz <= self._end_effector_high - self._box_bound_buffer))
        return inside_bound, inside_buffer_bound, mat, xyz

    def _compute_reward_(self):
        """Computes reward at a given time step.

        Returns:
            A float reward.
        """
        if self._target_type == "position":
            if self._reward_type == "linear":
                reward_dist = -np.linalg.norm(self._target_diff_, ord=2)
            elif self._reward_type == "precision":
                reward_dist = -np.linalg.norm(self._target_diff_, ord=2) +\
                              np.exp( -np.linalg.norm(self._target_diff_, ord=2)**2 / 0.01)
        elif self._target_type == "angle":
            if self._reward_type == "linear":
                reward_dist = -np.linalg.norm(self._target_diff_, ord=1)
            elif self._reward_type == "precision":
                reward_dist = -np.linalg.norm(self._target_diff_, ord=1) +\
                              np.exp(-np.linalg.norm(self._target_diff_, ord=1) ** 2 / 0.01)

        # TODO: doublecheck whether '0' or '-1' should be used as the index
        reward_vel = -self._vel_penalty * np.square(self._qd_[-1, self._joint_indices]).sum()

        #self.info['reward_dist'] = reward_dist
        #self.info['reward_vel'] = reward_vel

        return (reward_dist + reward_vel) * self._dt / 0.008

    def _check_done(self, env_done):
        """Checks whether the episode is over.


        Args:
            env_done:  a bool specifying whether the episode should be ended.

        Returns:
            A bool specifying whether the episode is over.
        """
        self._episode_steps += 1
        if (self._episode_steps >= self._episode_length_step) or env_done:
            self._actuator_comms['UR5'].actuator_buffer.write(self._nothing_packet)
            return True
        else:
            return False

    def reset(self, blocking=True):
        """Resets the arm, optionally blocks the environment until done."""
        ret = super(ReacherEnv, self).reset(blocking=blocking)
        self._episode_steps = 0
        return ret

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def terminate(self):
        """Gracefully terminates environment processes."""
        super(ReacherEnv, self).close()
