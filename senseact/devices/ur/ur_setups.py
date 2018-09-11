# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Contains setups for UR reacher environments. Specifies safety
box dimensions, joint limits to avoid self-collision etc."""

import numpy as np

setups = {
    'UR5_default':
              {
                  'host': '192.168.2.152',  # put UR5 Controller address here
                  'end_effector_low': np.array([-0.2, -0.3, 0.5]),
                  'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  'angles_low':np.pi/180 * np.array(
                      [ 60,
                       -180,#-180
                       -120,
                       -50,
                        50,
                        50
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 90,
                       -60,
                        130,
                        25,
                        120,
                        175
                       ]
                  ),
                  'speed_max': 0.3,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  'q_ref': np.array([ 1.58724391, -2.4, 1.5, -0.71790582, 1.63685572, 1.00910473]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )
              },
    'UR5_6dof':
              {
                  'host': '192.168.2.152',  # put UR5 Controller address here
                  'end_effector_low': np.array([-0.3, -0.6, 0.5]),
                  'end_effector_high': np.array([0.2, 0.4, 1.0]),
                  'angles_low':np.pi/180 * np.array(
                      [ 30,
                       -180,#-180
                       -120,
                       -120,
                        30,
                        0
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 150,
                       0,
                        130,
                        30,
                        150,
                        180
                       ]
                  ),
                  'speed_max': 0.3,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  'q_ref': np.array([ 1.58724391, -2.6, 1.6, -0.71790582, 1.63685572, 1.00910473]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params':
                      (
                          0.089159, # d1
                          -0.42500, # a2
                          -0.39225, # a3
                          0.10915,  # d4
                          0.09465,  # d5
                          0.0823    # d6
                      )
              },
    'UR3_default':
              {
                  'host': '192.168.2.152',  # put UR3 Controller address here
                  'end_effector_low': np.array([-0.12, 0.2, 0.4]),
                  'end_effector_high': np.array([-0.12, 1.1, 0.9]),
                  'angles_low': np.array([-180, -120])*np.pi/180,
                  'angles_high': np.array([-90, 120])*np.pi/180,
                  'reset_speed_limit': 0.6,
                  'q_ref': np.array([1.49595487, -1.8, 1.19992781, -3.0167721, -1.54870445, 3.11713743]),
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.01,
                  'ik_params':
                      (
                          0.1273,   # d1
                          -0.612,    # a2
                          -0.5723,   # a3
                          0.163941, # d4
                          0.1157,   # d5
                          0.0922    # d6
                      )
              }
}