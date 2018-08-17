import time
from senseact.communicator import Communicator
from senseact.sharedbuffer import SharedBuffer

class RodCommunicator(Communicator):
    """Communicator class for mujoco double inverted pendulum simulated cart sensor.

    This class implements cart specific simulated packet communicator.
    """

    def __init__(self,
                 simulator):

        """Inits RodCommunicator class with device- and task-specific parameters.

        Args:
            simulator: gym_simulator.GymSimulator object
        """

        self.simulator = simulator
        self.rod_obs_buffer = simulator.sim_rod_obs_buffer
        self.dt_tol = 0.0001

        sensor_args = {'buffer_len': SharedBuffer.DEFAULT_BUFFER_LEN,
                       'array_len': 6,
                       'array_type': 'd',
                       'np_array_type': 'd',
                      }

        self.read_start_time = self.read_end_time = time.time()

        super(RodCommunicator, self).__init__(use_sensor=True,
                                              use_actuator=False,
                                              sensor_args=sensor_args,
                                              actuator_args=None)

    def _sensor_handler(self):
        """Read values from rod sensor."""
        val = self.simulator.read_rod_sensor()
        #update sensor buffer
        self.sensor_buffer.write(val)
