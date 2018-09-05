from senseact.envs.sim_double_pendulum.sim_double_pendulum import DoubleInvertedPendulumEnv
import numpy as np


def main():
    # Create Asynchronous Simulation of InvertedDoublePendulum-v2 mujoco environment.
    env = DoubleInvertedPendulumEnv(agent_dt=0.005,
                                    sensor_dt=[0.01, 0.0033333],
                                    is_render=True,
                                   )

    # Start environment processes
    env.start()
    env.reset()

    for episode in range(1000):
        done = False
        while not done:
            action = np.random.uniform(-1.0, 1.0)
            obs, reward, done, _ = env.step(action)
        env.reset()

    # Shutdown the environment
    env.close()


if __name__ == '__main__':
    main()