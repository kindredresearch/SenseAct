# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
from senseact.communicator import Communicator
from senseact.sharedbuffer import SharedBuffer

class CartCommunicator(Communicator):
    """Communicator class for mujoco double inverted pendulum simulated cart sensor.

    This class implements cart specific simulated packet communicator.
    """

    def __init__(self,
                 simulator):

        """Inits CartCommunicator class with device- and task-specific parameters.

        Args:
            simulator: gym_simulator.GymSimulator object
        """
        self.simulator = simulator
        self.cart_obs_buffer = simulator.sim_cart_obs_buffer
        self.cart_action_buffer = simulator.sim_action_buffer
        self.dt_tol = 0.0001

        sensor_args = {'buffer_len': SharedBuffer.DEFAULT_BUFFER_LEN,
                       'array_len': 2,
                       'array_type': 'd',
                       'np_array_type': 'd',
                      }

        actuator_args = {'buffer_len': SharedBuffer.DEFAULT_BUFFER_LEN,
                         'array_len': 2,
                         'array_type': 'd',
                         'np_array_type': 'd',
                        }

        self.read_start_time = self.read_end_time = time.time()

        super(CartCommunicator, self).__init__(use_sensor=True,
                                               use_actuator=True,
                                               sensor_args=sensor_args,
                                               actuator_args=actuator_args
                                              )

    def _sensor_handler(self):
        """Reads cart values from cart sensor."""
        val = self.simulator.read_cart_sensor()
        #update sensor buffer
        self.sensor_buffer.write(val)

    def _actuator_handler(self):
        """Sends actuation command to robot."""
        if self.actuator_buffer.updated():
            recent_actuation, _, _ = self.actuator_buffer.read_update()
            self.simulator.write_cart_action(recent_actuation)
