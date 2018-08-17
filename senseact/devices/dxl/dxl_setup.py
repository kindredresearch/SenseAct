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
