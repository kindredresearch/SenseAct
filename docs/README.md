# Overview

This document gives some pointers on how to extend and run experiments with SenseAct.

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
 For example, the code below will run a random agent:

```python
import numpy as np
obs = env.reset()
while True:
    action = np.random.normal(size=env.action_space.shape)
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
```

More examples are provided in the [`examples`](../examples/) directory.  Our environment classes
are also [rllab](https://github.com/rll/rllab) compatible. For example, passing `rllab_box=True` as an argument to the
ReacherEnv makes it rllab compatible. Rllab uses object oriented abstractions for different
components required for their experiments. The environment should be constructed with the corresponding objects provided
by rllab.

# Understanding SenseAct tasks

A SenseAct task comprises implementations of environment and communicator classes.
The communicator class should be implemented on a per-device basis in a generic and task-agnostic way so that it can be reused for different tasks.
Any information the environment class and the agent need to know about the robot should only go through a communicator.
On the other hand, the 'environment' class fleshes out the reinforcement learning task specification by defining the observation space, the action space and the reward function among other things, based only on the sensorimotor information available from communicators.
In SenseAct, a communicator and an agent interact with each other in the following way:

![SenseAct](SenseAct.png)

The computations of the environment class are distributed among two processes: the experiment process and the task manager process as depicted above.

We provide an example of using a SenseAct task based on a simulated robot in [examples/advanced/sim_double_pendulum.py](../examples/advanced/sim_double_pendulum.py) so that the inner workings of SenseAct can be understood without requiring a real robot. The simulated robot is based on the [Double Inverted Pendulum environment](https://gym.openai.com/envs/InvertedDoublePendulum-v2/) from OpenAI Gym. The Gym environment is run asynchronously in a separate process than SenseAct processes, emulating how a real robot would work and be interfaced.



# Adding new tasks and robots

In order to add a new physical robot task to SenseAct, one needs to implement a 'communicator' class
facilitating communication with the robot and an 'environment' class defining the task.
The roles of communicators and environments are described in 'Understanding SenseAct' section.

### Implementing a communicator
The communicator class inherits from base [`Communicator`](../senseact/communicator.py) class. It should contain all the code required to read
sensory measurements from the robot and send actuation commands to the robot. For example,
reading packets from UR5 robot and sending it actuation commands are done via TCP/IP socket connection. Therefore a [`URCommunicator`](../senseact/devices/ur/ur_communicator.py) creates a socket objects that connects to
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
the robot, processing them into a numerical array and writing into a `sensor_buffer`, which is an object of [SharedBuffer](../senseact/sharedbuffer.py).

For example `_sensor_handler` method in the `URCommunicator` class receives packets from the
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

For example, the following fragment of `_actuation_handler` code in the URCommunicator class converts actions into `servoJ` UR5 commands and sends them over the socket connection:

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

An environment class inherits from base [`RTRLBaseEnv`](../senseact/rtrl_base_env.py) class and from `gym.core.Env` class defined in [Gym](https://github.com/openai/gym/). It contains the code specifying all the aspects of a task a reinforcement learning agent is to perform on the robot.
The environment class should define the reward function, and action and observation spaces for the task and implement corresponding properties. For example, the [`ReacherEnv`](../senseact/envs/ur/reacher_env.py) class defines continuous action and observation
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

The `_compute_actuation_` method converts an action produced by an agent into a numpy array representation of a corresponding actuation command and stores it into an `_actuation_packet_` dictionary, which has the format {communicator_string_name: actuation_numpy_array, ...}.

The `_reset_` method defines and executes the end of an episode reset function for a given task. For example, in UR5 Reacher reset moves the arm into a fixed initial position, therefore the `_reset_` method sends corresponding `moveL` UR5 command to `URCommunicator` and sleeps sufficient amount of time for the command to be executed on a robot.

The above three methods can be called on a different thread or process depends on the run mode.  Therefore, special care is required when implementing them.  We have added the trailing underscore to the methods' name as an caution indicator.
