from math import sin, cos, fabs, asin, acos, sqrt, atan2
from math import pi as PI
import numpy as np

FIRMWARE_VERSION = 3.3 # put firmware version here
DASHBOARD_SERVER_PORT = 29999  # to unlock protective stop
PRIMARY_CLIENT_INTERFACE_PORT = 30001
SECONDARY_CLIENT_INTERFACE_PORT = 30002
REALTIME_COMM_CLIENT_INTERFACE_PORT = 30003
if FIRMWARE_VERSION >= 3.5:
    REALTIME_COMM_PACKET_SIZE = 1108
elif FIRMWARE_VERSION >= 3.2:
    REALTIME_COMM_PACKET_SIZE = 1060
elif FIRMWARE_VERSION >= 3.0:
    REALTIME_COMM_PACKET_SIZE = 1044
elif FIRMWARE_VERSION >= 1.8:
    REALTIME_COMM_PACKET_SIZE = 812
elif FIRMWARE_VERSION >= 1.7:
    REALTIME_COMM_PACKET_SIZE = 764
else:
    REALTIME_COMM_PACKET_SIZE = 756

COMMANDS = {
    'NOTHING':
        {
            'id': 0,
            'size': 0
        },
    'SERVOJ':
        {
            'id': 1,
            'size': 9,
            'default': # as suggested by URScript API
                {
                    't': 0.008,
                    'lookahead_time': 0.1,
                    'gain': 300
                }
        },
    'SPEEDJ':
        {
            'id': 2,
            'size': 8,
            'default':
                {
                    'a': 1.4,
                    't_min': 0.008, # for firmware < 3.1, 0.02 might work better
                }
        },
    'MOVEL':
        {
            'id': 3,
            'size': 7,
            'default':
                {
                    'a': 1.2,
                    'v': 0.25,
                    't': 0,
                    'r': 0
                }
        },
    'MOVEJ':
        {
            'id': 4,
            'size': 10,
            'default':
                {
                    'a': 1.4,
                    'v': 1.05,
                    't': 0,
                    'r': 0
                }
        },
    'STOPJ':
        {
            'id': 5,
            'size': 1,
        },
    'UNLOCK_PSTOP':
        {
            'id': -1,
            'size': 0
        },
}

USE_DEFAULT = np.finfo(np.float64).min

ACTUATOR_DT = 0.008

if FIRMWARE_VERSION >= 3.5:
    REALTIME_COMM_PACKET = np.dtype(
    [('message_size', '>i4'),
     ('time', '>f8'),
     ('q_target', '>f8', (6,)),
     ('qd_target', '>f8', (6,)),
     ('qdd_target', '>f8', (6,)),
     ('i_target', '>f8', (6,)),
     ('m_target', '>f8', (6,)),
     ('q_actual', '>f8', (6,)),
     ('qd_actual', '>f8', (6,)),
     ('i_actual', '>f8', (6,)),
     ('i_control', '>f8', (6,)),
     ('tool_vector_actual', '>f8', (6,)),
     ('tcp_speed_actual', '>f8', (6,)),
     ('tcp_force', '>f8', (6,)),
     ('tool_vector_target', '>f8', (6,)),
     ('tcp_speed_target', '>f8', (6,)),
     ('digital_input_bits', '>f8'),
     ('motor_temperatures', '>f8', (6,)),
     ('controller_timer', '>f8'),
     ('test_value', '>f8'),
     ('robot_mode', '>f8'),
     ('joint_modes', '>f8', (6,)),
     ('safety_mode', '>f8'),
     ('reserved_0', '>f8', (6,)),
     ('tool_accelerometer_values', '>f8', (3,)),
     ('reserved_1', '>f8', (6,)),
     ('speed_scaling', '>f8'),
     ('linear_momentum_norm', '>f8'),
     ('reserved_2', '>f8'),
     ('reserved_3', '>f8'),
     ('v_main', '>f8'),
     ('v_robot', '>f8'),
     ('i_robot', '>f8'),
     ('v_actual', '>f8', (6,)),
     ('digital_outputs', '>f8'),
     ('program_state', '>f8'),
     ('elbow_position', '>f8', (3,)),
     ('elbow_velocity', '>f8', (3,)),
     ])
elif FIRMWARE_VERSION >= 3.2:
    REALTIME_COMM_PACKET = np.dtype(
    [('message_size', '>i4'),
     ('time', '>f8'),
     ('q_target', '>f8', (6,)),
     ('qd_target', '>f8', (6,)),
     ('qdd_target', '>f8', (6,)),
     ('i_target', '>f8', (6,)),
     ('m_target', '>f8', (6,)),
     ('q_actual', '>f8', (6,)),
     ('qd_actual', '>f8', (6,)),
     ('i_actual', '>f8', (6,)),
     ('i_control', '>f8', (6,)),
     ('tool_vector_actual', '>f8', (6,)),
     ('tcp_speed_actual', '>f8', (6,)),
     ('tcp_force', '>f8', (6,)),
     ('tool_vector_target', '>f8', (6,)),
     ('tcp_speed_target', '>f8', (6,)),
     ('digital_input_bits', '>f8'),
     ('motor_temperatures', '>f8', (6,)),
     ('controller_timer', '>f8'),
     ('test_value', '>f8'),
     ('robot_mode', '>f8'),
     ('joint_modes', '>f8', (6,)),
     ('safety_mode', '>f8'),
     ('reserved_0', '>f8', (6,)),
     ('tool_accelerometer_values', '>f8', (3,)),
     ('reserved_1', '>f8', (6,)),
     ('speed_scaling', '>f8'),
     ('linear_momentum_norm', '>f8'),
     ('reserved_2', '>f8'),
     ('reserved_3', '>f8'),
     ('v_main', '>f8'),
     ('v_robot', '>f8'),
     ('i_robot', '>f8'),
     ('v_actual', '>f8', (6,)),
     ('digital_outputs', '>f8'),
     ('program_state', '>f8'),
     ])
elif FIRMWARE_VERSION >= 3.0:
    REALTIME_COMM_PACKET = np.dtype(
    [('message_size', '>i4'),
     ('time', '>f8'),
     ('q_target', '>f8', (6,)),
     ('qd_target', '>f8', (6,)),
     ('qdd_target', '>f8', (6,)),
     ('i_target', '>f8', (6,)),
     ('m_target', '>f8', (6,)),
     ('q_actual', '>f8', (6,)),
     ('qd_actual', '>f8', (6,)),
     ('i_actual', '>f8', (6,)),
     ('i_control', '>f8', (6,)),
     ('tool_vector_actual', '>f8', (6,)),
     ('tcp_speed_actual', '>f8', (6,)),
     ('tcp_force', '>f8', (6,)),
     ('tool_vector_target', '>f8', (6,)),
     ('tcp_speed_target', '>f8', (6,)),
     ('digital_input_bits', '>f8'),
     ('motor_temperatures', '>f8', (6,)),
     ('controller_timer', '>f8'),
     ('test_value', '>f8'),
     ('robot_mode', '>f8'),
     ('joint_modes', '>f8', (6,)),
     ('safety_mode', '>f8'),
     ('reserved_0', '>f8', (6,)),
     ('tool_accelerometer_values', '>f8', (3,)),
     ('reserved_1', '>f8', (6,)),
     ('speed_scaling', '>f8'),
     ('linear_momentum_norm', '>f8'),
     ('reserved_2', '>f8'),
     ('reserved_3', '>f8'),
     ('v_main', '>f8'),
     ('v_robot', '>f8'),
     ('i_robot', '>f8'),
     ('v_actual', '>f8', (6,)),
     ])
elif FIRMWARE_VERSION >= 1.8:
    REALTIME_COMM_PACKET = np.dtype(
    [('message_size', '>i4'),
     ('time', '>f8'),
     ('q_target', '>f8', (6,)),
     ('qd_target', '>f8', (6,)),
     ('qdd_target', '>f8', (6,)),
     ('i_target', '>f8', (6,)),
     ('m_target', '>f8', (6,)),
     ('q_actual', '>f8', (6,)),
     ('qd_actual', '>f8', (6,)),
     ('i_actual', '>f8', (6,)),
     ('tool_accelerometer_values', '>f8', (3,)),
     ('unused', '>f8', (15,)),
     ('tcp_force', '>f8', (6,)),
     ('tool_vector', '>f8', (6,)),
     ('tcp_speed', '>f8', (6,)),
     ('digital_input_bits', '>f8'),
     ('motor_temperatures', '>f8', (6,)),
     ('controller_timer', '>f8'),
     ('test_value', '>f8'),
     ('robot_mode', '>f8'),
     ('joint_modes', '>f8', (6,)),
     ])
elif FIRMWARE_VERSION >= 1.7:
    REALTIME_COMM_PACKET = np.dtype(
    [('message_size', '>i4'),
     ('time', '>f8'),
     ('q_target', '>f8', (6,)),
     ('qd_target', '>f8', (6,)),
     ('qdd_target', '>f8', (6,)),
     ('i_target', '>f8', (6,)),
     ('m_target', '>f8', (6,)),
     ('q_actual', '>f8', (6,)),
     ('qd_actual', '>f8', (6,)),
     ('i_actual', '>f8', (6,)),
     ('tool_accelerometer_values', '>f8', (3,)),
     ('unused', '>f8', (15,)),
     ('tcp_force', '>f8', (6,)),
     ('tool_vector', '>f8', (6,)),
     ('tcp_speed', '>f8', (6,)),
     ('digital_input_bits', '>f8'),
     ('motor_temperatures', '>f8', (6,)),
     ('controller_timer', '>f8'),
     ('test_value', '>f8'),
     ('robot_mode', '>f8'),
     ])
else:
    REALTIME_COMM_PACKET = np.dtype(
    [('message_size', '>i4'),
     ('time', '>f8'),
     ('q_target', '>f8', (6,)),
     ('qd_target', '>f8', (6,)),
     ('qdd_target', '>f8', (6,)),
     ('i_target', '>f8', (6,)),
     ('m_target', '>f8', (6,)),
     ('q_actual', '>f8', (6,)),
     ('qd_actual', '>f8', (6,)),
     ('i_actual', '>f8', (6,)),
     ('tool_accelerometer_values', '>f8', (3,)),
     ('unused', '>f8', (15,)),
     ('tcp_force', '>f8', (6,)),
     ('tool_vector', '>f8', (6,)),
     ('tcp_speed', '>f8', (6,)),
     ('digital_input_bits', '>f8'),
     ('motor_temperatures', '>f8', (6,)),
     ('controller_timer', '>f8'),
     ('test_value', '>f8'),
     ])

class SafetyModes(object):
    """
    UR5 Safety Modes (for firmware 3.3, 3.4)
    """

    FAULT = 9
    VIOLATION = 8
    ROBOT_EMERGENCY_STOP = 7
    SYSTEM_EMERGENCY_STOP = 6
    SAFEGUARD_STOP = 5
    RECOVERY = 4
    PROTECTIVE_STOP = 3
    REDUCED = 2
    NORMAL = 1


class ServoJ(object):
    """Represents ServoJ UR5 command.

    ServoJ command facilitates online control in joint space.
    Servo to position (linear in joint-space)
    Servo function used for online control of the robot. The lookahead time
    and the gain can be used to smoothen or sharpen the trajectory.
    Note: A high gain or a short lookahead time may cause instability.
    Prefered use is to call this function with a new setpoint (q) in each time
    step (thus the default t=0.008)

    Attributes:
        q: a numpy array of float joint positions in rad
        t: a float representing duration of the command in seconds
        lookahead time: a float representing parameter for smoothing
            the trajectory, range [0.03,0.2]
        gain: a float representing a proportional gain for following
            target position, range [100,2000]
    """

    def __init__(self, q,
                 t=COMMANDS['SERVOJ']['default']['t'],
                 lookahead_time=COMMANDS['SERVOJ']['default']['lookahead_time'],
                 gain=COMMANDS['SERVOJ']['default']['gain']):
        """Inits the ServoJ object with command parameters.

        Args:
            See class attributes description.
        """
        self.q = q
        self.t = t
        self.lookahead_time = lookahead_time
        self.gain = gain

    def __repr__(self):
        return 'servoj([{}, {}, {}, {}, {}, {}], t={}, lookahead_time={}, gain={})'.format(
            *(list(self.q) + [self.t, self.lookahead_time, self.gain]))


class SpeedJ(object):
    """Represents SpeedJ UR5 command.

    SpeedJ command accelerates to and moves the arm with constant
    joints speed.

    Attributes:
        qd: a numpy array of float joint speeds in rad/s
        a: a float specifying joint acceleration in rad/sˆ2 (of leading axis)
        t_min: a float specifying minimal time before function returns
    """

    def __init__(self, qd,
                 a=COMMANDS['SPEEDJ']['default']['a'],
                 t_min=COMMANDS['SPEEDJ']['default']['t_min']):
        """Inits the ServoJ object with command parameters.

        Args:
            See class attributes description.
        """
        self.qd = qd
        self.a = a
        self.t_min = t_min

    def __repr__(self):
        if FIRMWARE_VERSION >= 3.1 and FIRMWARE_VERSION < 3.3:
            return 'speedj([{}, {}, {}, {}, {}, {}], {})'.format(
                *(list(self.qd) + [self.a]))
        else:
            return 'speedj([{}, {}, {}, {}, {}, {}], {}, {})'.format(
                *(list(self.qd) + [self.a, self.t_min]))

class MoveJ(object):
    """Represents MoveJ UR5 command.

    MoveJ command moves thge arm to a given position
    (linear in joint-space). When using this command, the
    robot must be at standstill or come from a movej or movel commands with a
    blend. The speed and acceleration parameters control the trapezoid
    speed profile of the move. The $t$ parameters can be used in stead to
    set the time for this move. Time setting has priority over speed and
    acceleration settings. The blend radius can be set with the $r$
    parameters, to avoid the robot stopping at the point. However, if he
    blend region of this mover overlaps with previous or following regions,
    this move will be skipped, and an ’Overlapping Blends’ warning
    message will be generated.

    Attributes:
        q: a numpy array of float joint positions (q can also be
            specified as a pose, then inverse kinematics is used
            to calculate the corresponding joint positions)
        a: a float specifying joint acceleration of leading
            axis in rad/sˆ2
        v: a float specifying joint speed of leading axis
            in rad/s
        t: a float specifying duration of the command in s
        r: a float specifying blend radius in m
    """

    def __init__(self, q,
                 a=COMMANDS['MOVEJ']['default']['a'],
                 v=COMMANDS['MOVEJ']['default']['v'],
                 t=COMMANDS['MOVEJ']['default']['t'],
                 r=COMMANDS['MOVEJ']['default']['r']):
        """Inits the MoveJ object with command parameters.

        Args:
            See class attributes description.
        """
        self.q = q
        self.a = a
        self.v = v
        self.t = t
        self.r = r

    def __repr__(self):
        return 'movej([{}, {}, {}, {}, {}, {}], a={}, v={}, t={}, r={})'.format(
            *(list(self.q) + [self.a, self.v, self.t, self.r]))

class MoveL(object):
    """Represnts MoveL UR5 command.

    MoveL command moves the arm to position (linear in tool-space).
    See movej for analogous details.

    Attributes:
        pose: a float numpy array representing target pose (pose can
            also be specified as joint positions, then forward kinematics
            is used to calculate the corresponding pose)
        a: a float specifying tool acceleration in m/sˆ2
        v: a float specifying tool speed in m/s
        t: a float specifying duration of the commnd in s
        r: a float specifying blend radius in m
    """

    def __init__(self, pose,
                 a=COMMANDS['MOVEL']['default']['a'],
                 v=COMMANDS['MOVEL']['default']['v'],
                 t=COMMANDS['MOVEL']['default']['t'],
                 r=COMMANDS['MOVEL']['default']['r']):
        """Inits the MoveL object with command parameters.

        Args:
            See class attributes description.
        """
        self.pose = pose
        self.a = a
        self.v = v
        self.t = t
        self.r = r

    def __repr__(self):
        return 'movej([{}, {}, {}], a={}, v={}, t={}, r={})'.format(
            *(list(self.pose) + [self.a, self.v, self.t, self.r]))


class StopJ(object):
    """Represents StopJ UR5 command.

    StopJ decellerates joint speeds to zero.

    Attributes:
        a: a float specifying joint acceleration in rad/sˆ2 (of leading axis)
    """

    def __init__(self, a):
        """Inits the MoveL object with command parameters.

        Args:
            See class attributes description.
        """
        self.a = a

    def __repr__(self):
        return 'stopj(a={})'.format(self.a)


ZERO_THRESH = 0.00000001;

#UR5
# d1 =  0.089159;
# a2 = -0.42500;
# a3 = -0.39225;
# d4 =  0.10915;
# d5 =  0.09465;
# d6 =  0.0823;

def sign(x):
    return 1 if x > 0 else 0 if x == 0 else -1

def forward(q, params):
    """Computes forward kinematics solutions.

    Args:
        q: a numpy array representing joint positions in rad
            params: a tuple of UR5 arm physical parameters (e.g. links lengths)

    Returns:
        A 4x4 rigid body transformation matrix for world-to-gripper
        coordinate transform.
    """
    d1, a2, a3, d4, d5, d6 = params


    s1, s2, s3, s4, s5, s6 = np.sin(q)
    c1, c2, c3, c4, c5, c6 = np.cos(q)
    del s4, c4

    sum_q234 = q[1:4].sum()
    s234 = sin(sum_q234)
    c234 = cos(sum_q234)
    del q

    T = np.zeros(16)

    T[0] = ((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0;

    T[1] = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) -
            (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0);

    T[2] = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 -
            s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0));

    T[3] = ((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 -
            d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 -
            a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3);

    T[4] = c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0;

    T[5] = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) +
            s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0));

    T[6] = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) -
            s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0))

    T[7] = ((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 +
            (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 -
            a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3);

    T[8] = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0);

    T[9] = ((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6);

    T[10] = (s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0);

    T[11] = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 -
             (d6*(c234*c5+s234*s5))/2.0 - d5*c234);
    T[15] = 1.0
    return T.reshape(4, 4)

def inverse(T, wrist_desired, params):
    """Computes inverse kinematics solutions.

    Args:
        T: A 4x4 rigid body transformation matrix for
            world-to-gripper coordinate transform.
        wrist_desired: //TODO
        params: a tuple containing physical arm parameters

    Returns:
         A list containing joint-angle 6-vectors with solutions 
         to inverse kinematics problem
    """
    d1, a2, a3, d4, d5, d6 = params

    rval = []

    T = T.flatten()
    T02 = -T[0]
    T00 =  T[1]
    T01 =  T[2]
    T03 = -T[3]
    T12 = -T[4]
    T10 =  T[5]
    T11 =  T[6]
    T13 = -T[7]
    T22 =  T[8]
    T20 = -T[9]
    T21 = -T[10]
    T23 =  T[11]

    # Q1
    q1 = [0.0, 0.0]
    A = d6*T12 - T13;
    B = d6*T02 - T03;
    R = A*A + B*B;

    if (fabs(A) < ZERO_THRESH):
        if (fabs(fabs(d4) - fabs(B)) < ZERO_THRESH):
            div = -sign(d4)*sign(B);
        else:
            div = -d4/B;
        arcsin = asin(div);
        if (fabs(arcsin) < ZERO_THRESH):
            arcsin = 0.0;
        if(arcsin < 0.0):
            q1[0] = arcsin + 2.0*PI;
        else:
            q1[0] = arcsin;
        q1[1] = PI - arcsin;

    elif (fabs(B) < ZERO_THRESH):
        if (fabs(fabs(d4) - fabs(A)) < ZERO_THRESH):
            div = sign(d4)*sign(A);
        else:
            div = d4/A;
        arccos = acos(div);
        q1[0] = arccos;
        q1[1] = 2.0*PI - arccos;
    elif(d4*d4 > R):
        return []
    else:
        arccos = acos(d4 / sqrt(R)) ;
        arctan = atan2(-B, A);
        pos = arccos + arctan;
        neg = -arccos + arctan;

        if (fabs(pos) < ZERO_THRESH):
            pos = 0.0;
        if (fabs(neg) < ZERO_THRESH):
            neg = 0.0;
        if(pos >= 0.0):
            q1[0] = pos;
        else:
            q1[0] = 2.0*PI + pos;
        if (neg >= 0.0):
            q1[1] = neg;
        else:
            q1[1] = 2.0*PI + neg;

    # Q5
    q5 = [[0.0, 0.0], [0.0, 0.0]];
    for i in [0, 1]:
        numer = (T03*sin(q1[i]) - T13*cos(q1[i])-d4);
        if (fabs(fabs(numer) - fabs(d6)) < ZERO_THRESH):
            div = sign(numer) * sign(d6);
        else:
            div = numer / d6;
        arccos = acos(div);
        q5[i][0] = arccos;
        q5[i][1] = 2.0*PI - arccos;

    # Q234 here we go...
    for i in [0, 1]:
        for j in [0, 1]:
            c1 = cos(q1[i])
            s1 = sin(q1[i])
            c5 = cos(q5[i][j])
            s5 = sin(q5[i][j])

            if (fabs(s5) < ZERO_THRESH):
                q6 = wrist_desired;
            else:
                q6 = atan2(sign(s5)*-(T01*s1 - T11*c1),
                           sign(s5)*(T00*s1 - T10*c1));
                if (fabs(q6) < ZERO_THRESH):
                    q6 = 0.0
                if (q6 < 0.0):
                    q6 += 2.0 * PI

            q2 = [0.0, 0.0]
            q3 = [0.0, 0.0]
            q4 = [0.0, 0.0]

            c6 = cos(q6)
            s6 = sin(q6)
            x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1));
            x04y = c5*(T20*c6 - T21*s6) - T22*s5;
            p13x = (d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1))
                    - d6*(T02*c1 + T12*s1) + T03*c1 + T13*s1)
            p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6);
            c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3);
            if(fabs(fabs(c3) - 1.0) < ZERO_THRESH):
                c3 = sign(c3)
            elif(fabs(c3) > 1.0):
                continue
            arccos = acos(c3);
            q3[0] = arccos;
            q3[1] = 2.0*PI - arccos;
            denom = a2*a2 + a3*a3 + 2*a2*a3*c3;
            s3 = sin(arccos)
            A = (a2 + a3*c3)
            B = a3*s3
            q2[0] = atan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom);
            q2[1] = atan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom);
            c23_0 = cos(q2[0]+q3[0]);
            s23_0 = sin(q2[0]+q3[0]);
            c23_1 = cos(q2[1]+q3[1]);
            s23_1 = sin(q2[1]+q3[1]);
            q4[0] = atan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0);
            q4[1] = atan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1);
            for k in [0, 1]:
                if(fabs(q2[k]) < ZERO_THRESH):
                    q2[k] = 0.0;
                elif(q2[k] < 0.0):
                    q2[k] += 2.0*PI;
                if (fabs(q4[k]) < ZERO_THRESH):
                    q4[k] = 0.0
                elif(q4[k] < 0.0):
                    q4[k] += 2.0*PI;
                q_soln = [q1[i], q2[k], q3[k], q4[k], q5[i][j], q6]
                rval.append(np.asarray(q_soln))
    for solution in rval:
        for ii, joint_ii in enumerate(solution):
            while joint_ii < - np.pi:
                joint_ii += 2 * np.pi
            while joint_ii > np.pi:
                joint_ii -= 2 * np.pi
            solution[ii] = joint_ii
    return rval

# sorted according to distance from ref_pos
def inverse_near(T, wrist_desired, ref_pos, params):
    """Computes inverse kinematics solutions near given position.

    Args:
        T: A 4x4 rigid body transformation matrix for
            world-to-gripper coordinate transform.
        wrist_desired: //TODO
        ref_pos: a tuple containing reference joint positions in rad.
            The funciton will search solutions to ik problem near this
            position. 
        params: a tuple containing physical arm parameters

    Returns:
         A list containing joint-angle 6-vectors with solutions 
         to inverse kinematics problem
    """
    solutions = inverse(T, wrist_desired, params)
    return sorted(solutions, key=lambda x: np.linalg.norm(x-ref_pos))