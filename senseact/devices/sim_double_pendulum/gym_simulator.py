import time
import numpy as np
from senseact.sharedbuffer import SharedBuffer
from multiprocessing import Process

class GymSimulator():
    """A simulator that emulates real-world interactions.

    Runs a simulator (openai gym or mujoco) in a loop.1
    The simulator runs in parralel with the rl_agent,
    from which it reads the commanded action and sends
    its observations at the specified frequencies.
    """

    def __init__(self,
                 env,
                 gym_dt=0.001,
                 sensor_dt=[0.001, 0.001],
                 is_render=False,
                ):
        """Inits GymSimulator class with task and communication specific parameters.

        Args:
            env: mujoco or openaigym environment object.
            gym_dt: Time step length associated with gym environment cycle.
            sensor_dt: List of time steps assocaited with each sensor cycle.
            is_render: Flag that renders the simulation if set to True
        """
        self.initialize_buffer_variables()
        self.env = env
        self.gym_dt = gym_dt
        self.gym_update_time = time.time()
        self.sim_action_dim = len(self.env.action_space.low)
        self.observation_dim = len(self.env.observation_space.low)
        self.is_render = is_render

        #Starts process for each sensor (asynchronous updates to simulate real-world communications)
        robot_buffer_list = [self.sim_cart_obs_buffer, self.sim_rod_obs_buffer]
        index_list = [[0, 5], [1, 3, 6, 2, 4, 7]]
        self.sensor_pp = []
        for counter in range(len(robot_buffer_list)):
            self.sensor_pp.append(
                Process(target=self.sensor_loop,
                        args=(sensor_dt[counter],
                              robot_buffer_list[counter],
                              self.gym_obs_buffer,
                              index_list[counter])))
            self.sensor_pp[counter].start()
            counter += 1

    def update_from_gym(self, robot_buffer_list, gym_buffer, obs_index):
        """Reads most recent observations from gym buffer and updates the rl_buffer.

        Args:
            robot_buffer_list: buffer used to update the observations for a sensor
                at a given timestep sensor_dt.
            gym_buffer: buffer used to publish the observations at a given timestep
                gym_dt.
            obs_index: Observation index (or list of indices) associated with the
                particular sensor.
        """
        #read value from gym buffer
        value, _, _ = gym_buffer.read()
        value = np.array(value[0])
        #write value to placeholder_buffer used to update observations use by rl agent
        robot_buffer_list.write(value[obs_index])

    def sensor_loop(self, sensor_dt, robot_buffer_list, gym_buffer, obs_index):
        """Runs as separate process. Update rl_buffer for a given sensor at a desired frequency.

        Args:
            sensor_dt: timestep for this particular sensor
            robot_buffer_list: buffer used to update the observations for a sensor
                at a given timestep sensor_dt.
            gym_buffer: buffer used to publish the observations at a given timestep
                gym_dt.
            obs_index: Observation index (or list of indices) associated with the
                particular sensor.
        """
        time_last_update = time.time()
        while True:
            self.update_from_gym(robot_buffer_list, gym_buffer, obs_index)
            while time.time() - time_last_update < sensor_dt:
                time.sleep(sensor_dt*0.01)
            time_last_update = time.time()

    def run_simulator(self):
        """Runs simulator in a loop calling the env.step() function."""
        obs = self.env.reset()
        done = 0
        is_pause = 1
        flag = flag_previous = 0
        counter = 0
        while True:
            #maintain gym_dt time step
            while time.time() - self.gym_update_time < self.gym_dt:
                time.sleep(self.gym_dt*0.01)
            self.gym_update_time = time.time()
            #read latest desired action
            action_vec, _, _ = self.sim_action_buffer.read()
            action = action_vec[0][0]
            flag_previous = flag
            flag = action_vec[0][1]
            if flag_previous == 1 and flag == 0:
                is_pause=0
            if flag == 1 and not is_pause: #reset environment
                obs = self.env.reset()
                self.gym_obs_buffer.write(obs)
                is_pause = 1
            elif flag == 0 and not is_pause: #apply gym step function (simulate real robot)
                try:
                    obs, reward_dummy, done_dummy, info = self.env.step(action)
                    if self.is_render and counter%30==0:
                        self.env.render()
                    counter += 1
                except NameError:
                    print('Error: Action is undefined')
                #publish observations to gym buffer
                self.gym_obs_buffer.write(obs)

    def read_cart_sensor(self):
        """Gets sensor value from robot (blocking).

        Returns:
            List of observations associated with cart sensor.
        """
        while True:
            if self.sim_cart_obs_buffer.updated():
                val, _, _ = self.sim_cart_obs_buffer.read_update()
                break
            else:
                time.sleep(0.0001)
        return val

    def read_rod_sensor(self):
        """Gets sensor value from robot (blocking).

        Returns:
            List of observations associated with rod sensor.
        """
        while True:
            if self.sim_rod_obs_buffer.updated():
                val, _, _ = self.sim_rod_obs_buffer.read_update()
                break
            else:
                time.sleep(0.0001)
        return val

    def read_last_cart_sensor(self):
        """Get sensor value from robot (non-blocking)."""
        val, _, _ = self.sim_cart_obs_buffer.read()
        return val

    def read_last_rod_sensor(self):
        """Get sensor value from robot (non-blocking)."""
        val, _, _ = self.sim_rod_obs_buffer.read()
        return val

    def write_cart_action(self, action):
        """Sends commanded action to robot.

        Args:
            action: robot command to implement in env.step(action).
        """
        self.sim_action_buffer.write(action)

    def initialize_buffer_variables(self):
        """Initializes buffer types used to pass data between python processes."""
        self.sim_action_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=2,
            array_type='d',
            np_array_type='d',
        )
        self.sim_cart_obs_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=2,
            array_type='d',
            np_array_type='d',
        )
        self.gym_obs_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=11,
            array_type='d',
            np_array_type='d',
        )
        self.sim_rod_obs_buffer = SharedBuffer(
            buffer_len=SharedBuffer.DEFAULT_BUFFER_LEN,
            array_len=6,
            array_type='d',
            np_array_type='d',
        )
