# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from math import pi

setups = \
    {
    'dxl_gripper_default':
        {
            'angles_low'  : [-150.0 * pi/180.0],
            'angles_high' : [0.0 * pi/180.0],
            'high_load'   : [85],
        },
    'dxl_tracker_default':
        {
            'angles_low'  : [-pi/3.],
            'angles_high' : [pi/3.],
            'high_load'   : [85],
        }
    }
