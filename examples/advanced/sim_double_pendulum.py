import time
import baselines.common.tf_util as U
import numpy as np


from baselines.ppo1.pposgd_simple import learn
from baselines.ppo1.mlp_policy import MlpPolicy

from senseact.envs.sim_double_pendulum.sim_double_pendulum import DoubleInvertedPendulumEnv
from senseact.utils import tf_set_seeds
from helper import create_callback
from multiprocessing import Process, Value, Manager

def main():
    # use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    #Create Asynchronous Simulation of InvertedDoublePendulum-v2 mujoco environment.
    env = DoubleInvertedPendulumEnv(agent_dt=0.005,
                                    sensor_dt=[0.01, 0.0033333],
                                    is_render=False,
                                    random_state=rand_state
                                   )
    # Start environment processes
    env.start()

    # Create baselines ppo policy function
    sess = U.single_threaded_session()
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=64, num_hid_layers=2)

    # create and start plotting process
    plot_running = Value('i', 1)
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], })
    # Plotting process
    pp = Process(target=plot_returns, args=(env, 2048, shared_returns, plot_running))
    pp.start()

    # Create callback function for logging data from baselines PPO learn

    kindred_callback = create_callback(shared_returns)

    # Train baselines PPO
    learn(env,
          policy_fn,
          max_timesteps=1e6,
          timesteps_per_actorbatch=2048,
          clip_param=0.2,
          entcoeff=0.0,
          optim_epochs=10,
          optim_stepsize=0.0001,
          optim_batchsize=64,
          gamma=0.995,
          lam=0.995,
          schedule="linear",
          callback=kindred_callback,
         )

    # Safely terminate plotter process
    plot_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    # Shutdown the environment
    env.close()

def plot_returns(env, batch_size, shared_returns, plot_running):
    """Plots episodic returns

    Args:
        env: An instance of DoubleInvertedPendulumEnv
        batch_size: An int representing timesteps_per_batch provided to the PPO learn function
        shared_returns: A manager dictionary object containing `episodic returns` and `episodic lengths`
        plot_running: A multiprocessing Value object containing 0/1.
            1: Continue plotting, 0: Terminate plotting loop
    """

    print("Started plotting routine")
    import matplotlib.pyplot as plt
    plt.ion()
    time.sleep(5.0)
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    hl11, = ax.plot([], [])
    fig.suptitle("Simulated Double Pendulum", fontsize=14)
    ax.set_title("Learning Curve")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average Returns")
    count = 0
    old_size = len(shared_returns['episodic_returns'])
    returns = []
    while plot_running.value:
        if count % 20 == 0:
            if len(shared_returns['episodic_returns']) > old_size:
                returns.append(np.mean(shared_returns['episodic_returns'][-(len(shared_returns['episodic_returns']) - old_size):]))
                old_size = len(shared_returns['episodic_returns'])
                # plot learning curve
                hl11.set_ydata(returns)
                hl11.set_xdata(batch_size * np.arange(len(returns)))
                ax.set_ylim([np.min(returns), np.max(returns)])
                ax.set_xlim(
                    [0, int(len(returns) * batch_size)])
                fig.canvas.draw()
                fig.canvas.flush_events()
        time.sleep(0.01)
        fig.canvas.draw()
        fig.canvas.flush_events()
        count += 1

if __name__ == '__main__':
    main()
