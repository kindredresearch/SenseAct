import time
import gym
import logging
import numpy as np
import senseact.devices.create2.create2_config as create2_config
from senseact import utils

from multiprocessing import Array, Value

from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.devices.create2.create2_communicator import Create2Communicator
from senseact.envs.create2.create2_observation import Create2ObservationFactory


class Create2DockerEnv(RTRLBaseEnv, gym.Env):
    """Create2 environment for training it to dock at the charging station.

    By default this environment observes the infrared character (binary state for detecting the dock),
    the wall sensors' strength, bump sensor, charging sources available flag, and last action.

    The reward is a sum of docking sensor reward, forward reward, and bumper penalty, each scaled to
    have equal weights of roughly 20.

    TODO:
        * move the gym env dependency to some base class
    """

    def __init__(self, episode_length_time, port='/dev/ttyUSB0', ir_window=20, ir_history=1,
                 obs_history=1, dt=0.015, auto_unwind=True, rllab_box=False, **kwargs):
        """Constructor of the environment.

        Args:
            episode_length_time:  A float duration of an episode defined in seconds
            port:                 the serial port to the Create2 (eg. '/dev/ttyUSB0')
            ir_window:            the number of IR history to include in calculation
            ir_history:           the number of IR packet history (not observation history)
            obs_history:          the number of observation history to keep
            dt:                   the cycle time in second
            auto_unwind:          boolean of whether we want to execute the auto cable-unwind code
            rllab_box:            whether we are using rllab algorithm or not
            **kwargs:             any other arguments to be passed to the base class
        """
        self._ir_window = ir_window
        self._ir_history = ir_history
        self._obs_history = obs_history
        self._episode_step_ = Value('i', 0)
        self._episode_length_time = episode_length_time
        self._episode_length_step = int(episode_length_time / dt)
        self._internal_timing = 0.015
        self._total_rotation = 0
        self._max_rotation = 720
        self._auto_unwind = auto_unwind
        self._min_battery = 1700
        self._max_battery = 2600

        # get the opcode for our main action (only 1 action)
        self._main_op = 'drive_direct'
        self._extra_ops = ['safe', 'seek_dock', 'drive']
        main_opcode = create2_config.OPCODE_NAME_TO_CODE[self._main_op]
        extra_opcodes = [create2_config.OPCODE_NAME_TO_CODE[op] for op in self._extra_ops]

        # store the previous action to be shared across processes
        self._prev_action_ = np.frombuffer(Array('i', 2).get_obj(), dtype='i')

        # create factory with common arguments for making an observation dimension
        observation_factory = Create2ObservationFactory(main_op=self._main_op,
                                                        dt=dt,
                                                        obs_history=self._obs_history,
                                                        ir_window=self._ir_window,
                                                        ir_history=self._ir_history,
                                                        internal_timing=self._internal_timing,
                                                        prev_action=self._prev_action_)

        # the definition of the observed state and the associated custom modification (if any)
        # before passing to the learning algorithm
        self._observation_def = [
            observation_factory.make_dim('light bump left signal'),
            observation_factory.make_dim('light bump front left signal'),
            observation_factory.make_dim('light bump center left signal'),
            observation_factory.make_dim('light bump center right signal'),
            observation_factory.make_dim('light bump front right signal'),
            observation_factory.make_dim('light bump right signal'),
            observation_factory.make_dim('infrared character omni'),
            observation_factory.make_dim('infrared character left'),
            observation_factory.make_dim('infrared character right'),
            observation_factory.make_dim('bumps and wheel drops'),
            observation_factory.make_dim('charging sources available'),
            observation_factory.make_dim('previous action')
        ]

        # extra packets we need for proper reset and charging
        self._extra_sensor_packets = ['angle', 'battery charge', 'oi mode', 'stasis', 'distance',
                                      'cliff left', 'cliff front left', 'cliff front right', 'cliff right']
        main_sensor_packet_ids = [d.packet_id for d in self._observation_def if d.packet_id is not None]
        extra_sensor_packet_ids = [create2_config.PACKET_NAME_TO_ID[nm] for nm in self._extra_sensor_packets]

        # TODO: move this out to some base class?
        if rllab_box:
            from rllab.spaces import Box as RlBox  # use this for rllab TRPO
            Box = RlBox
        else:
            from gym.spaces import Box as GymBox  # use this for baselines algos
            Box = GymBox

        # go thru the main opcode (just direct_drive in this case) and add the range of each param
        # XXX should the action space include the opcode? what about op that doesn't have parameters?
        self._action_space = Box(
            low=np.array([r[0] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()]),
            high=np.array([r[1] for r in create2_config.OPCODE_INFO[main_opcode]['params'].values()])
        )

        # loop thru the observation dimension and get the lows and highs
        self._observation_space = Box(
            low=np.concatenate([d.lows for d in self._observation_def]),
            high=np.concatenate([d.highs for d in self._observation_def])
        )

        self._comm_name = 'Create2'
        buffer_len = int(max([self._ir_window * self._ir_history,
                              dt / self._internal_timing]) + 1)
        communicator_setups = {self._comm_name: {'Communicator': Create2Communicator,
                                                 # have to read in this number of packets everytime to support
                                                 # all operations
                                                 'num_sensor_packets': buffer_len,
                                                 'kwargs': {'sensor_packet_ids': main_sensor_packet_ids +
                                                                                 extra_sensor_packet_ids,
                                                            'opcodes': [main_opcode] + extra_opcodes,
                                                            'port': port,
                                                            'buffer_len': 2 * buffer_len,
                                                           }
                                                }
                              }

        super(Create2DockerEnv, self).__init__(communicator_setups=communicator_setups,
                                               action_dim=len(self._action_space.low),
                                               observation_dim=len(self._observation_space.low),
                                               dt=dt,
                                               **kwargs)

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        """The required _computer_sensation_ interface.

        Args:
            name:               the name of communicator the sense is from
            sensor_window:      an array of size num_sensor_packets each containing 1 complete observation packets
            timestamp_window:   array of timestamp corresponds to the sensor_window
            index_window:       array of count corresponds to the sensor_window

        Returns:
            A numpy array with [:-2] the sensation, [-2] the reward, [-1] the done flag
        """
        # construct the actual sensation

        actual_sensation = []
        for d in self._observation_def:
            res = d.normalized_handler(sensor_window)
            actual_sensation.extend(res)

        # accumulate the rotation information
        self._total_rotation += sensor_window[-1][0]['angle']

        reward, done = self._calc_reward(sensor_window)

        return np.concatenate((actual_sensation, [reward], [done]))

    def _compute_actuation_(self, action, timestamp, index):
        """The required _compute_actuator_ interface.

        The side effect is to write the output to self._actuation_packet_[name] with [opcode, *action]

        Args:
            action:      an array of 2 numbers correspond to the speed of the left & right wheel
            timestamp:   the timestamp when the action was written to buffer
            index:       the action count

        TODO:
            * convert this interface to be more similar to _computer_sensation_
        """
        # TODO: communicator only support one operation at a time, modify communicator to support multiple
        #       in one update? Or modify _compute_actuation to write multiple commands into actuation_buffer?

        # add a safety check for any action with nan or inf
        if any([not np.isfinite(a) for a in action]):
            logging.warning("Invalid action received: {}".format(action))
            return

        # pass int only action
        action = action.astype('i')
        self._actuation_packet_[self._comm_name] = np.concatenate(
            ([create2_config.OPCODE_NAME_TO_CODE[self._main_op]], action))
        np.copyto(self._prev_action_, action)

    def _reset_(self):
        """The required _reset_ interface.

        This method does the handling of charging the Create2, repositioning, and set to the correct mode.
        """
        logging.info("Resetting...")
        self._episode_step_.value = 0
        np.copyto(self._prev_action_, np.array([0, 0]))
        for d in self._observation_def:
            d.reset()

        # wait for create2 to startup properly if just started (ie. wait to actually start receiving observation)
        while not self._sensor_comms[self._comm_name].sensor_buffer.updated():
            time.sleep(0.01)

        sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()

        # check if next episode should start with random reset or not
        random_reset = False
        if sensor_window[-1][0]['charging sources available'] > 0:
            random_reset = True
        else:
            # send the Create2 to dock to start
            logging.info("Sending Create2 to dock.")
            self._write_opcode('seek_dock')
            dock_wait = 20
            while sensor_window[-1][0]['charging sources available'] == 0 and dock_wait > 0:
                time.sleep(1.0)
                dock_wait -= 1

                sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()

        if sensor_window[-1][0]['battery charge'] <= self._min_battery:
            self._wait_until_charged()

            sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()

        # Always switch to SAFE mode to run an episode, so that Create2 will switch to PASSIVE on the
        # charger.  If the create2 is in any other mode on the charger, we will not be able to detect
        # the non-responsive sleep mode that happens at the 60 seconds mark.
        logging.info("Setting Create2 into safe mode.")
        self._write_opcode('safe')
        time.sleep(0.1)

        # after charging/docked, try to drive away from the dock if still on it
        if sensor_window[-1][0]['charging sources available'] > 0:
            logging.info("Undocking the Create2.")
            self._write_opcode('drive_direct', -250, -250)
            time.sleep(0.75)
            self._write_opcode('drive_direct', 0, 0)
            time.sleep(0.1)

        # drive fast toward a random places, then stop (using drive_direct since it's easier to calculate
        # the rotation angle)
        logging.info("Moving Create2 into position.")

        if random_reset:
            target_values = [self._rand_obj_.uniform(r[0], r[1]) for r in [[-250, -50], [-250, -50]]]
        else:
            target_values = [-100, -100]

        self._write_opcode('drive_direct', *target_values)
        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))
        time.sleep(2.5)
        self._write_opcode('drive', 0, 0)

        # find the rotation angle (right wheel distance - left wheel distance) / wheel base distance
        self._total_rotation += (target_values[1] * 1.5 - target_values[0] * 1.5) / 235.0 * 180.0 / 3.14
        self._wait_until_unwinded()

        # make sure in SAFE mode in case the random drive caused switch to PASSIVE, or
        # create2 stuck somewhere and require human reset (don't want an episode to start
        # until fixed, otherwise we get a whole bunch of one step episodes)
        sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()
        while sensor_window[-1][0]['oi mode'] != 2:
            logging.warning("Create2 not in SAFE mode, reattempting... (might require human intervention).")
            self._write_opcode('full')
            time.sleep(0.2)
            self._write_opcode('drive_direct', -50, -50)
            time.sleep(0.5)
            self._write_opcode('drive', 0, 0)
            time.sleep(0.1)
            self._write_opcode('safe')
            time.sleep(0.2)

            # try another unwind since unwind could have failed if stuck in PASSIVE mode
            self._wait_until_unwinded()
            sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()

        # don't want to state during reset pollute the first sensation
        time.sleep(2 * self._internal_timing)

        logging.info("Reset completed.")

    def _check_done(self, env_done):
        """The required _check_done_ interface.

        Args:
            env_done:   whether the environment is done from _compute_sensation_

        Returns:
            A boolean flag for done
        """
        self._episode_step_.value += 1
        return self._episode_step_.value >= self._episode_length_step or env_done

    def _calc_reward(self, sensor_window):
        """Helper to calculate reward.

        Args:
            sensor_window: the sensor_window from _compute_sensation_

        Returns:
            A tuple of (reward, done)
        """
        num_packets = int(self._dt / self._internal_timing)

        charging_sources_available = sensor_window[-1][0]['charging sources available']
        oi_mode = sensor_window[-1][0]['oi mode']

        # use the detection of the docking station IR & wheel drop as reward / penalty, and
        # include huge reward (0 or 100) when on docking station
        bw = 0
        for p in range(num_packets):
            bw |= sensor_window[-1 - p][0]['bumps and wheel drops']
        cl = 0
        for p in range(num_packets):
            cl += sensor_window[-1 - p][0]['cliff left']
            cl += sensor_window[-1 - p][0]['cliff front left']
            cl += sensor_window[-1 - p][0]['cliff front right']
            cl += sensor_window[-1 - p][0]['cliff right']

        ir_values = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        for p in range(self._ir_window * self._ir_history):
            ir_omni = sensor_window[-1 - p][0]['infrared character omni']
            ir_left = sensor_window[-1 - p][0]['infrared character left']
            ir_right = sensor_window[-1 - p][0]['infrared character right']
            ir_value = np.array([(ir_left >> 2 & 1), (ir_left & 1), (ir_left >> 3 & 1),  # left <-> green
                                 (ir_omni >> 2 & 1), (ir_omni & 1), (ir_omni >> 3 & 1),
                                 (ir_right >> 3 & 1), (ir_right & 1), (ir_right >> 3 & 1)])  # right <-> red
            ir_values += ir_value

        # scale the value base on which sensor should prioritize which IR led
        ir_values = ir_values / (self._ir_window * self._ir_history)
        ir_values = ir_values * np.array([1.0, 0.5, 0.05, 0.65, 0.15, 0.65, 0.05, 0.5, 1.0])

        distance_reward = 0
        for p in range(num_packets):
            # stasis = sensor_window[-1 - p][0]['stasis'] & 1
            distance = sensor_window[-1 - p][0]['distance']
            distance_reward += distance

        csa = 0
        csa_max = 0
        for p in range(num_packets):
            csa += (num_packets - p) / num_packets * (sensor_window[-1 - p][0]['charging sources available'] >> 1)
            csa_max += (num_packets - p) / num_packets
        csa = csa / csa_max

        # scaling each of the following reward to approximately the same scale (20)
        reward = 0
        reward += 4.0 * np.sum(ir_values)
        reward += 20 / (4.0 * int(self._dt / self._internal_timing)) * distance_reward
        reward -= 10.0 * ((bw & 1) + ((bw & 2) >> 1))   # penalty of bumping

        # make sure the reward below are bigger than possible reward above
        reward += csa * 150

        # makes reward dependent on cycle time
        reward *= self._dt

        # if in PASSIVE mode, not charging, not cliff triggered, then it might be in the automatic docking mode
        if oi_mode == 1 and charging_sources_available == 0 and cl == 0:
            self._write_opcode('safe')

        # If wheel dropped, it's done.
        done = 0
        if (bw >> 2) > 0:
            done = 1

        return reward, done

    def _write_opcode(self, opcode_name, *args):
        """Helper method to force write a command not part of the action dimension.

        Args:
            opcode_name:    the name of the opcode
            *args:          any arguments require for the operation
        """
        # write the command directly to actuator_buffer to avoid the limitation that the opcode
        # is not part of the action dimension
        self._actuator_comms[self._comm_name].actuator_buffer.write(
            np.concatenate(([create2_config.OPCODE_NAME_TO_CODE[opcode_name]], np.array(args).astype('i'))))

    def _wait_until_charged(self):
        """Waits until Create 2 is sufficiently charged."""
        sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()
        while sensor_window[-1][0]['battery charge'] < self._max_battery:
            # move it out of the dock to avoid the weird non-responsive sleep mode (the non-responsive sleep
            # mode can happen on any mode while on the dock, but only detectable when in PASSIVE mode)
            self._write_opcode('safe')
            time.sleep(0.1)
            self._write_opcode('drive_direct', -100, -100)
            time.sleep(0.5)
            self._write_opcode('drive_direct', 0, 0)
            time.sleep(0.1)
            self._write_opcode('seek_dock')

            logging.info("Create2 charging with current charge at {}.".format(sensor_window[-1]['battery charge']))
            time.sleep(50)

            sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read()

    def _wait_until_unwinded(self):
        """Waits until unwinded if the angle is high enough.

        Method could fail if the Create2 is stuck in PASSIVE mode.
        """
        if not self._auto_unwind:
            return

        if abs(self._total_rotation) >= self._max_rotation:
            logging.info("Unwinding the Create2 cables: {} degrees.".format(self._total_rotation))
            # circumference of wheel base is 738.27mm, find the seconds required at max speed
            total_distance = self._total_rotation * 738.27 / 360.0
            # fixed slow unwinding speed to minimize noise
            max_velocity = 150
            # increase unwind time by a factor to compensate not unwinding enough
            unwind_time = abs(total_distance / max_velocity)
            self._write_opcode('drive', max_velocity, -1 * np.sign(total_distance))

            unwind_start_time = time.time()
            while time.time() - unwind_start_time < unwind_time:
                if self._sensor_comms[self._comm_name].sensor_buffer.updated():
                    sensor_window, _, _ = self._sensor_comms[self._comm_name].sensor_buffer.read_update()
                    self._total_rotation += sensor_window[-1][0]['angle']

                    # if error in unwinding switched to PASSIVE mode
                    if sensor_window[-1][0]['oi mode'] != 2:
                        break
                time.sleep(0.4 * self._internal_timing)

            self._write_opcode('drive', 0, 0)

    # ======== rllab compatible gym codes =========

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def terminate(self):
        super(Create2DockerEnv, self).close()
