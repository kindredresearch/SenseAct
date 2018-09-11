# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import senseact.devices.create2.create2_config as create2_config

from collections import deque


class Create2ObservationFactory(object):
    """Main helper factory class for spawning the proper observation dimension handler."""

    def __init__(self, **kwargs):
        """Constructor of the factory.

        Args:
            **kwargs: all the arguments required by any of the observation dimension
        """
        self._kwargs = kwargs

    def make_dim(self, name):
        """Creates the corresponding Create2ObservationDim object.

        Args:
            name: the name of the dimension

        Returns:
            An Create2ObservationDim object or one of its subclasses.
        """
        if name in ['light bump left signal', 'light bump front left signal', 'light bump center left signal',
                    'light bump center right signal', 'light bump front right signal', 'light bump right signal']:
            return Create2LightBumpDim(name, **self._kwargs)

        if name in ['infrared character omni', 'infrared character left', 'infrared character right']:
            return Create2InfraredCharacterDim(name, **self._kwargs)

        if name in ['bumps and wheel drops']:
            return Create2BumpsAndWheelDropsDim(name, **self._kwargs)

        if name in ['charging sources available']:
            return Create2ChargingSourcesAvailableDim(name, **self._kwargs)

        if name in ['previous action']:
            return Create2PrevActionDim(name, **self._kwargs)

        return Create2ObservationDim(name, **self._kwargs)


class Create2ObservationDim(object):
    """Base helper class for a single dimension of observation."""

    def __init__(self, name, **kwargs):
        """Constructor of the observation dimension.

        Args:
            name:     the name of the dimension
            **kwargs: all required parameters for calculation the dimension, such as dt, internal_timing,
                or obs_history
        """
        # initialize all passed in values under its own name
        for k, v in kwargs.items():
            setattr(self, '_' + k, v)

        self._name = name

        # since _computer_sensation_ runs faster than the cycle time, we need to keep enough history
        # so that when sensation is retrieved on step, it would include the proper history from last
        # step
        self._history = deque(maxlen=int(np.ceil(self._dt / self._internal_timing)) * self._obs_history)

    @property
    def packet_id(self):
        return create2_config.PACKET_NAME_TO_ID[self._name] if self._name in create2_config.PACKET_NAME_TO_ID else None

    @property
    def name(self):
        return self._name

    @property
    def lows(self):
        # assume normalized between -1 to 1
        return [-1 for i in self.ranges] * self._obs_history

    @property
    def highs(self):
        # assume normalized between -1 to 1
        return [1 for i in self.ranges] * self._obs_history

    @property
    def ranges(self):
        """Returns the ranges for each dimension within the object.

        Returns:
            An array of 2D array like [[0, 1], [0, 1], ...] specifying the low and the high.
        """
        try:
            return self._stored_ranges
        except AttributeError:
            self._stored_ranges = self._ranges()
            return self._stored_ranges

    def normalized_handler(self, sensor_window):
        """Calculates the observation value and normalize to within -1 and 1.

        Args:
            sensor_window:  the full sensor_window from _compute_sensation_

        Returns:
            A numpy array of length obs_history, each index a numpy array containing all values.
        """
        orig_values = self._handler(sensor_window)
        normal_values = []
        ranges = self.ranges
        for i, d in enumerate(orig_values):
            normal_values.append(float(d - ranges[i][0]) / (ranges[i][1] - ranges[i][0]))
            # shift between -1 to 1
            normal_values[-1] = normal_values[-1] * 2.0 - 1.0

        self._history.appendleft(normal_values)

        # append the correct history that is multiple of cycle time back
        results = []
        for i in range(self._obs_history):
            index = int(i * np.ceil(self._dt / self._internal_timing))
            if index >= len(self._history):
                results.append(np.zeros_like(normal_values))
            else:
                results.append(self._history[index])
        return np.concatenate(results)

    def reset(self):
        """Resets the observation history.

        Needs to be called manually each time an episode reset."""
        self._history.clear()

    def _ranges(self):
        """Returns the range for each dimensions.

        Returns:
            An array of 2D arrays like [[0, 1], [0, 1], ...] containing the range for each dimension.
        """
        return [create2_config.PACKET_INFO[create2_config.PACKET_NAME_TO_ID[self._name]]['range']]

    def _handler(self, sensor_window):
        """Handler of the dimension for processing incoming observation.

        Args:
            sensor_window: the complete sensor_window

        Returns:
            Numpy array containing the post-processed values for each dimension (default just return the full range)
        """
        return np.array([sensor_window[-1][0][self._name]])


class Create2LightBumpDim(Create2ObservationDim):
    """The Light Bump wall sensing dimensions.

    Applies inverse square root to the incoming signal to rescale the ranges.
    """
    def __init__(self, *args, **kwargs):
        super(Create2LightBumpDim, self).__init__(*args, **kwargs)
        self.__beta = 4.2

    def _ranges(self):
        signal_range = create2_config.PACKET_INFO[create2_config.PACKET_NAME_TO_ID[self._name]]['range']
        s_small = 1.0 / np.sqrt(signal_range[0] - self.__beta)
        s_large = 1.0 / np.sqrt(signal_range[1] - self.__beta)
        return [[s_small, s_large]]

    def _handler(self, sensor_window):
        s_curr = 1.0 / np.sqrt(max(1.0, sensor_window[-1][0][self._name] - self.__beta))
        return np.array([s_curr])


class Create2InfraredCharacterDim(Create2ObservationDim):
    """The Infrared Character dimension for detecting the docking station.

    Converts each infrared character into 3 dimensions for left buoy, force field, right buoy
    averaged over specific window size and keeping packet history.
    """
    def _ranges(self):
        return [[0, 1]] * 3 * self._ir_history

    def _handler(self, sensor_window):
        output = np.array([0, 0, 0] * self._ir_history)
        index = 0
        for h in range(self._ir_history):
            for p in range(self._ir_window):
                ir = sensor_window[-1 - index - p][0][self._name]
                output[3 * h:3 * h + 3] += np.array([ir >> 2 & 1, ir & 1, ir >> 3 & 1])
            index += self._ir_window

        return output / self._ir_window


class Create2BumpsAndWheelDropsDim(Create2ObservationDim):
    """The physical bump dimension.

    Converts the incoming flag into 2 binary dimension one for each physical bumper.
    """

    def __init__(self, *args, **kwargs):
        super(Create2BumpsAndWheelDropsDim, self).__init__(*args, **kwargs)
        self.__num_packets = max(1, int(self._dt / self._internal_timing))

    def _ranges(self):
        return [[0, 1]] * 2

    def _handler(self, sensor_window):
        bw = 0
        for p in range(self.__num_packets):
            bw |= sensor_window[-1 - p][0][self._name]
        return np.array([bw >> 1 & 1, bw & 1])


class Create2ChargingSourcesAvailableDim(Create2ObservationDim):
    """Handles the charging_sources_available observation to read only home base flag.

    Linear weight the observation over the cycle.
    """

    def __init__(self, *args, **kwargs):
        super(Create2ChargingSourcesAvailableDim, self).__init__(*args, **kwargs)
        self.__num_packets = max(1, int(self._dt / self._internal_timing))

    def _ranges(self):
        return [[0, sum((self.__num_packets - p) / self.__num_packets for p in range(self.__num_packets))]]

    def _handler(self, sensor_window):
        csa = 0
        for p in range(self.__num_packets):
            csa += (self.__num_packets - p) / self.__num_packets * (sensor_window[-1 - p][0]['charging sources available'] >> 1)
        return np.array([csa])


class Create2PrevActionDim(Create2ObservationDim):
    """Special dimension that returns the previous action."""
    def _ranges(self):
        return list(create2_config.OPCODE_INFO[
                        create2_config.OPCODE_NAME_TO_CODE[self._main_op]]['params'].values())

    def _handler(self, sensor_window):
        """
        Returns the last action as observation.

        :param sensor_window:     the complete sensor_window
        :return:                  the last action as numpy array
        """
        return np.array(self._prev_action.copy())
