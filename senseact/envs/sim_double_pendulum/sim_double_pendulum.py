import numpy as np
import gym
import time

from senseact.rtrl_base_env import RTRLBaseEnv
from senseact.devices.sim_double_pendulum.gym_simulator import GymSimulator
from senseact.devices.sim_double_pendulum.cart_communicator import CartCommunicator
from senseact.devices.sim_double_pendulum.rod_communicator import RodCommunicator
from multiprocessing import Value, Process


class DoubleInvertedPendulumEnv(RTRLBaseEnv, gym.core.Env):
    """Double Inverted Pendulum Environment implemented in mujoco(DoubleInvertedPendulumEnv).

    This task simulates real world communications of the DoubleInvertedPendulumEnv-v2 by running
    the simulator (mujoco), the actuators, the sensors, in separate processes asynchronously.

    The sensors of the cartpole system are split into 2: the cart and the rod (2 links).
    The cart has a single actuator and 2 observations while the rod has a total of 9 observations.
    """

    def __init__(self,
                 agent_dt=0.01,
                 sensor_dt=[0.001, 0.001],
                 gym_dt=0.001,
                 is_render=False,
                 **kwargs
                ):
        """Inits DoubleInvertedPendulumEnv class with task and communication specific parameters.

        Args:
            agent_dt: Time step length associated with agent cycle.
            sensor_dt: List of time steps associated with each sensor cycle.
            gym_dt: Time step length associated with gym environment cycle.
                       This should match the dt increment set in openai gym.
            **kwargs: Keyword arguments
        """
        self.agent_dt = agent_dt
        self.gym_dt = gym_dt
        self.sensor_dt = sensor_dt
        self.is_render = is_render
        from gym.envs.registration import register
        register(
            id='SimDoublePendulum-v0',
            entry_point='senseact.envs.sim_double_pendulum:InvertedDoublePendulumEnvSenseAct',
            max_episode_steps=1000,
            reward_threshold=9100.0,
        )
        self.env = gym.make('SimDoublePendulum-v0') #Same as DoubleInvertedPendulumEnv-v2 but with dt=0.001s
        self.episode_steps = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._reward_ = Value('d', 0.0)
        self.observations = np.zeros(self.env.observation_space.shape[0])
        self.episode_length_step = int(2 / self.agent_dt)

        #start openai gym simulator in a separate process (simulates real word interactions)
        self._start_simulator_()

        #setup a communicator for each sensor/actuator
        # ------------------------------------------
        # CartCommunicator: reads x_cart position, velocity and sends x actuator commands
        # RodCommunicator: reads position and velocity of rod angles theta1 and theta2
        # Note that the buffer type is used to pass variables from gym environment to RL task.
        # On real robot, this buffer will be replaced by the device communication protocol.
        # ------------------------------------------
        communicator_setups = {
            'CartCommunicator': {
                'Communicator': CartCommunicator,
                'kwargs': {
                    'simulator': self.simulator,
                }
            },
            'RodCommunicator': {
                'Communicator': RodCommunicator,
                'kwargs': {
                    'simulator': self.simulator,
                }
            },
        }

        super(DoubleInvertedPendulumEnv, self).__init__(
            communicator_setups=communicator_setups,
            action_dim=1,
            observation_dim=self.observation_space.shape[0],
            dt=agent_dt,
            **kwargs
        )

    def _reset_(self):
        """Resets the environment episode."""
        #get last command sent to robot
        last_action, _, _ = self._actuator_comms['CartCommunicator'].actuator_buffer.read()
        #build new action vector [last_action[0][0], flag], where flag = 1 means reset mujoco environment
        action = np.array([last_action[0][0], 1])
        #write reset command to actuator_buffer
        self._actuator_comms['CartCommunicator'].actuator_buffer.write(action)
        time.sleep(.01)

    def _compute_sensation_(self, name, sensor_window, timestamp_window, index_window):
        """Creates and saves an observation vector based on sensory data.

        For the DoubleInvertedPendulumEnv environment the observation vector is a concatenation of:
            - cart position: x
            - link angle: [sin(theta1), sin(theta2)]
            - link angle: [cos(theta1), cos(theta2)]
            - velocities: [dx dtheta1 dtheta2]

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
        if name == 'CartCommunicator':
            self.observations[[0, 5]] = sensor_window[0]
            # update local environment with most recent cart observations
            self.env.env.data.set_joint_qpos('slider', self.observations[0])
            self.env.env.data.set_joint_qvel('slider', self.observations[5])
        elif name == 'RodCommunicator':
            self.observations[[1, 3, 6, 2, 4, 7]] = sensor_window[0]
            # update local environment with most recent rod observations
            self.env.env.data.set_joint_qpos('hinge', np.arcsin(self.observations[1]))
            self.env.env.data.set_joint_qvel('hinge', self.observations[6])
            self.env.env.data.set_joint_qpos('hinge2', np.arcsin(self.observations[2]))
            self.env.env.data.set_joint_qvel('hinge2', self.observations[7])
        self._reward_.value = self._compute_reward()

        #set state of local environment with most recent observations
        self.env.env.set_state(self.env.env.data.qpos, self.env.env.data.qvel)
        x, _, y = self.env.env.data.site_xpos[0]

        #check if episode is done
        done = bool(y <= 1)

        return np.concatenate(
            (
                self.observations,
                np.array([self._reward_.value]),
                np.array([done])
            )
        )

    def _compute_actuation_(self, action, timestamp, index):
        """Creates and sends actuation packets to the communicator.

        Computes actuation commands based on agent's action and
        control type and writes actuation packets to the
        communicators' actuation buffers.

        Args:
            action: a float containing the action value
            timestamp: a float containing action timestamp
            index: an integer containing action index
        """
        self._actuation_packet_['CartCommunicator'][0] = action
        self._actuation_packet_['CartCommunicator'][-1] = 0 #flag=0 means do not reset

    def _compute_reward(self):
        """Computes reward at a given timestamp.

        The reward is defined as in
        <rllab/rllab/envs/mujoco/inverted_double_pendulum_env.py>.
        The reward is computed using the latest observations updates.

        Returns:
            An array containing the scalar reward
        """
        x, _, y = self.env.env.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.env.env.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10
        reward = float(alive_bonus - dist_penalty - vel_penalty)
        reward *= (self.agent_dt / self.gym_dt)

        return np.array([reward])

    def _check_done(self, env_done):
        """Checks whether the episode is over.

        Args:
            env_done:  a bool specifying whether the episode should be ended.

        Returns:
            A bool specifying whether the episode is over.
        """
        #update episode length
        self.episode_steps += 1
        if self.episode_steps >= self.episode_length_step or env_done:
            # print('steps ', self.episode_steps, ' out of ', self.episode_length_step)

            done = True
        else:
            done = False
        return done

    def reset(self, blocking=True):
        """Resets environment and updates episode_steps.

        Returns:
            Array of observations
        """
        ret = super(DoubleInvertedPendulumEnv, self).reset(blocking=blocking)
        self.episode_steps = 0
        return ret

    def close(self):
        """Closes all manager threads and communicator processes.

        Overwrites rtrl_env close method.
        """
        for name, comm in self._all_comms.items():
            comm.terminate()
            comm.join()

        for process in self.simulator.sensor_pp:
            process.terminate()
            process.join()

        self.pp.terminate()
        self.pp.join()

        super(DoubleInvertedPendulumEnv, self).close()

    def _start_simulator_(self):
        """Starts gym simulator as in independent process that simulates the real world running in real-time."""
        #Define Simulator object
        self.simulator = GymSimulator(self.env,
                                      gym_dt=self.gym_dt,
                                      sensor_dt=self.sensor_dt,
                                      is_render=self.is_render,
                                     )

        #start simulator as a separate process
        self.pp = Process(target=self.simulator.run_simulator, args=())
        self.pp.start()
