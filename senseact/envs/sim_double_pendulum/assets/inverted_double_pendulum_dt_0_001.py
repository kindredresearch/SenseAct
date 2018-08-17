import os
from gym import utils
from gym.envs.mujoco import inverted_double_pendulum, mujoco_env


class InvertedDoublePendulumEnvSenseAct(inverted_double_pendulum.InvertedDoublePendulumEnv):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(__file__) + '/inverted_double_pendulum_dt_0_001.xml', 5)
        utils.EzPickle.__init__(self)


