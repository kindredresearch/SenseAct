# SenseAct: A computational framework for real-world robot learning tasks

This repository provides the implementation of several reinforcement learning (RL) tasks with three different real-world robots.
These tasks come with an interface similar to [OpenAI-Gym](https://github.com/openai/gym) so that learning algorithms can be plugged in easily and in a uniform manner across tasks.
All the tasks here are implemented based on a computational model of robot-agent communication proposed by Mahmood et al. (2018a), which we call *SenseAct*.
In this computational model, agent and environment-related computations are ordered and distributed among multiple concurrent processes in a specific way. By doing so, SenseAct enables the following:

- Timely communication between the learning agent and multiple robotic devices with reduced latency,
- Easy and systematic design of robotic tasks for reinforcement learning agents.
- Facilitate reproducible real-world reinforcement learning.

This repository provides the following real-world robotic tasks, which are proposed by Mahmood et al. (2018b) as benchmark tasks for reinforcement learning algorithms:

### For Universal-Robots (UR) robotic arms:
- [UR-Reacher](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/ur/reacher_env.py) (both 2 joint and 6 joint control)

### For Dynamixel (DXL) actuators:
- [DXL-Reacher](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/dxl/dxl_reacher_env.py)
- [DXL-Tracker](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/dxl/dxl_tracker_env.py)

### For iRobot Create 2 robots:
- [Create-Mover](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/create2/create2_mover_env.py)
- [Create-Docker](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/create2/create2_docker_env.py)

Mahmood et al. (2018b) provides extensive results comparing multiple reinforcement learning algorithms on these tasks. Their results can be reproduced by using this repository (see 'Running experiments' section).

# Installation

SenseAct uses Python3 (>=3.5).

On Linux and Mac OS X, run the following:
1. `git clone https://github.com/kindredresearch/SenseAct.git`
1. `cd SenseAct`
1. `pip install -e .` or `pip3 install -e .` depends on your setup

Additionally on Ubuntu, another package is needed:

1. `sudo apt-get install python3-tk`

The example experiments use the OpenAI Baselines implementation of Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)) for learning. To install baselines run:

```
sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install baselines==0.1.5
```

### Additional installation steps for Dynamixel-based tasks (Linux only)

Dynamixels can be controlled by drivers written using either ctypes by [Robotis](https://github.com/ROBOTIS-GIT/DynamixelSDK/releases/tag/3.5.4) or pyserial, which can be chosen by passing either `True` (ctypes) or `False` (pyserial) as an argument to the `use_ctypes_driver` parameter of a Dynamixel-based task (e.g., see `examples/dxl_reacher.py`). We found the ctypes-based driver to provide substantially more timely and precise communication compared to the pyserial-based one.

In order to use the CType-based driver, we need to install gcc and relevant packages for compiling the C libraries:

`sudo apt-get install gcc-5 build-essential gcc-multilib g++-multilib`

Then run the following to compile the C code:

`sudo bash setup_dxl.sh`

For additional setup and troubleshooting information regarding Dynamixels, please see [DXL Docs](senseact/devices/dxl/README.md).

# Running experiments

All environments in SenseAct inherit from OpenAI Gym base environment class, therefore they support Gym interface. In order to run for example an experiment with the UR-Reacher task, first one needs to create an environment object with parameters specific to the robot used in the experiment and start environment processes:

```python
from senseact.envs.ur.reacher_env import ReacherEnv

# Create UR-Reacher-2 environment
env = ReacherEnv(
        setup="UR5_default",
        host=None,
        dof=2,
        control_type="velocity",
        target_type="position",
        reset_type="zero",
        reward_type="precision",
        derivative_type="none",
        deriv_action_max=5,
        first_deriv_max=2,
        accel_max=1.4,
        speed_max=0.3,
        speedj_a=1.4,
        episode_length_time=4.0,
        episode_length_step=None,
        actuation_sync_period=1,
        dt=0.04,
        run_mode="multiprocess",
        rllab_box=False,
        movej_t=2.0,
        delay=0.0,
        random_state=None
    )
# Start environment processes
env.start()
```
After this we can use UR5 Reacher environment object as a regular Gym environment.
 For example, the  code below will run a random agent:

```python
import numpy as np
obs = env.reset()
while True:
    action = np.random.normal(size=env.action_space.shape)
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
```

More examples are provided in the `examples` directory. Our environment classes
are also [rllab](https://github.com/rll/rllab) compatible. For example, passing `rllab_box=True` as an argument to the
ReacherEnv makes it rllab compatible. Rllab uses object oriented abstractions for different
components required for their experiments. The environment should be constructed with the corresponding objects provided
by rllab.

# Understanding SenseAct tasks

A SenseAct task consists an environment class and communicator class implementations.
The communicator is written on a per-device basis.
When a device is reused for different tasks or even different robots, the device-specific communicator is supposed to be reused.
Therefore, the communicator is supposed to be generic and task agnostic.
Any information the environment class and the agent need to know about the robot should only go through a communicator.
On the other hand, the 'environment' class fleshes out the reinforcement learning task specification by defining the observation space, the action space and the reward function among other things, based only on the sensorimotor information available from communicators.
In SenseAct, communicators and an environment interact with each other in the following way:

![SenseAct](SenseAct.png)

The computations of the environment class are distributed among two processes: the experiment process and the task manager process as depicted above.

# Using SenseAct in Simulation

We provide an example of using a SenseAct task based on a simulated robot in `examples/sim_double_pendulum.py` so that the inner workings of SenseAct can be understood without requiring a real robot. The simulated robot is based on the [Double Inverted Pendulum environment](https://gym.openai.com/envs/InvertedDoublePendulum-v2/) from OpenAI Gym. The Gym environment is run asynchronously in a separate process than SenseAct processes, emulating how a real robot would work and be interfaced.

### Installation steps for simulation

The simulation requires OpenAI Baselines (see installation section [above](#installation)) and a popular physics simulator called MuJoCo. First install the prerequisites for MuJoCo:

```
sudo apt-get install patchelf libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```

Then install MuJoCo according to their [install guide](https://github.com/openai/mujoco-py#install-mujoco).

# Adding new tasks and robots

In order to add a new physical robot task to SenseAct, one needs to implement a 'communicator' class
facilitating communciation with the robot and an 'environment' class defining the task.
The roles of communicators and environments are described in 'Understanding SenseAct' section.

### Implementing a communicator
The communicator class inherits from base [`Communicator`](senseact/communicator.py) class. It should contain all the code required to read
sensory measurements from the robot and send actuation commands to the robot. For example,
reading packets from UR5 robot and sending it actuation commands are done via TCP/IP socket connection. Therefore a [`URCommunicator`](senseact/devices/ur/ur_communicator.py) creates a socket objects that connects to
a UR5 Controller:

```python
class URCommunicator(Communicator):
    def __init__(*args, **kwargs):
      ## some code
      self._sock = URCommunicator.make_connection(*args, **kwargs)
      ## more code
```

Communication with other physical robots may be done through other means, such as pymodbus connection for example, and the communicator class should contain the corresponding code.

The communicator class should implement at least the following methods defined in the base `Communicator` class:

```python
class Communicator(Process):
    def _sensor_handler(self):
        """Handles sensor packet communication and necessary processing.

        Re-establishes connection when it is lost.
        """
        raise NotImplementedError

    def _actuator_handler(self):
        """Handles actuator packet communication and necessary processing.

        Re-establishes connection when it is lost.
        """
        raise NotImplementedError

```

The `_sensor_handler` method is responsible for receiving sensory packets from
the robot, processing them into a numerical array and writing into a `sensor_buffer`, which is an object of [SharedBuffer](senseact/sharedbuffer.py).

For example  `_sensor_handler` method in the `URCommunicator` class receives packets from the
socket object and converts them into a numpy array:

```python
def _sensor_handler(self):
    ## some code
    data = self._sock.recv(ur_utils.REALTIME_COMM_PACKET_SIZE)
    parsed = np.frombuffer(data, dtype=ur_utils.REALTIME_COMM_PACKET)
    self.sensor_buffer.write(parsed)
    ## more code
```

The `_actuation_handler` method is responsible for reading actions from `actuator_buffer`, another `SharedBuffer` object, converting them into proper actuation commands and sending them to the robot.

For example, the following fragment of  `_actuation_handler` code in the URCommunicator class converts actions into `servoJ` UR5 commands and sends them over the socket connection:

```python
def _actuation_handler(self):
    recent_actuation, time_stamp, _ = self.actuator_buffer.read_update()
    recent_actuation = recent_actuation[0]
    if recent_actuation[0] == ur_utils.COMMANDS['SERVOJ']['id']:
        servoj_values = ur_utils.COMMANDS['SERVOJ']
        cmd = ur_utils.ServoJ(
          q=recent_actuation[1:1 + servoj_values['size'] - 3],
          t=servoj_values['default']['t']
            if recent_actuation[-3] == ur_utils.USE_DEFAULT else recent_actuation[-3],
          lookahead_time=servoj_values['default']['lookahead_time']
            if recent_actuation[-2] == ur_utils.USE_DEFAULT else recent_actuation[-2],
          gain=servoj_values['default']['gain']
            if recent_actuation[-1] == ur_utils.USE_DEFAULT else recent_actuation[-1]
        )
    cmd_str = '{}\n'.format(cmd)
    self._sock.send(cmd_str.encode('ascii'))
```

### Implementing an environment

An environment class inherits from base [`RTRLBaseEnv`](senseact/rtrl_base_env.py) class and from `gym.core.Env` class defined in [Gym](https://github.com/openai/gym/). It contains the code specifying all the aspects of a task a reinforcement learning agent is to perform on the robot.
The environment class should define the reward function, and action and observation spaces for the task and implement corresponding properties. For example, the [`ReacherEnv`](senseact/envs/ur/reacher_env.py) class defines continuous action and observation
spaces as:

```python
import gym
from senseact.rtrl_base_env import RTRLBaseEnv

class ReacherEnv(RTRLBaseEnv, gym.core.Env):
    def __init__(*args, **kwargs):
        ## some code
        from gym.spaces import Box
        self._observation_space = Box(
            low=np.array(
                list(self.angles_low * self.obs_history)  # q_actual
                + list(-np.ones(self.dof * self.obs_history))  # qd_actual
                + list(self.target_low)  # target
                + list(-self.action_low)  # previous action in cont space
            ),
            high=np.array(
                list(self.angles_high * self.obs_history)  # q_actual
                + list(np.ones(self.dof * self.obs_history))  # qd_actual
                + list(self.target_high)    # target
                + list(self.action_high)    # previous action in cont space
            )
        )
        self._action_space = Box(low=self.action_low, high=self.action_high)
        ## more code

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
```

An environment class should also construct a `communicator_setups` dictionary containing
arguments for corresponding communicator class and pass it to the constructor of a base `RTRLBaseEnv` class. The `ReacherEnv` class defines the following dictionary for `URCommunicator` class:

```python
class ReacherEnv(RTRLBaseEnv, gym.core.Env):
    def __init__(*args, **kwargs):
        ## some code
        communicator_setups = {'UR5':
                                   {
                                    'num_sensor_packets': obs_history,
                                    'kwargs': {'host': self.host,
                                               'actuation_sync_period': actuation_sync_period,
                                               'buffer_len': obs_history + SharedBuffer.DEFAULT_BUFFER_LEN,
                                               }
                                    }
                               }
        super(ReacherEnv, self).__init__(communicator_setups=communicator_setups,
                                         action_dim=len(self.action_space.low),
                                         observation_dim=len(self.observation_space.low),
                                         dt=dt,
                                         **kwargs)
```

In addition, an environment should implement at least the following methods defined in `RTRLBaseEnv` class:

```python
class RTRLBaseEnv(object):
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

    def _reset_(self):
      """Performs the reset procedure for the task.

      To be implemented by the Environment class.
      """
      raise NotImplementedError
```
The `_compute_sensation_` method converts (a history of) sensory data into an observation vector, computes reward, identifies whether the episode is over and returns these data as a flat numpy array.

The `_compute_actuation_` method converts an action produced by an RL agent into a numpy array representation of a corresponding actuation command and stores it into an `_actuation_packet_` dictionary, which has the format {communicator_string_name: actuation_numpy_array, ...}.

The `_reset_` method defines and executes the end of an episode reset function for a given task. For example, in UR5 Reacher reset moves the arm into a fixed initial position,  therefore the `_reset_` method sends corresponding `moveL` UR5 command to `URCommunicator` and sleeps sufficient amount of time for the command to be executed on a robot.

# Citing SenseAct

For the SenseAct computational model, please cite Mahmood et al. (2018a). For individual tasks, please cite Mahmood et al. (2018b).

* Mahmood, A. R., Korenkevych, D., Komer,B. J., Bergstra, J. (2018a). [Setting up a reinforcement learning task with a real-world robot](https://arxiv.org/abs/1803.07067). In *IEEE/RSJ International Conference on Intelligent Robots and Systems*.

* Mahmood A. R., Korenkevych, D., Vasan, G., Ma, W., Bergstra, J. (2018b). Benchmarking Reinforcement Learning Algorithms on Real-World Robots.




