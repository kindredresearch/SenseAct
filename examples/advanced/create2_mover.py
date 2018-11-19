# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import copy
import sys
import numpy as np
import pickle as pkl
import baselines.common.tf_util as U

import senseact.devices.create2.create2_config as create2_config

from multiprocessing import Process, Value, Manager
from baselines.trpo_mpi.trpo_mpi import learn
from baselines.ppo1.mlp_policy import MlpPolicy

from senseact.envs.create2.create2_mover_env import Create2MoverEnv
from senseact.utils import tf_set_seeds, NormalizedEnv
from helper import create_callback


def main():
    # optionally use a pretrained model
    load_model_data = None
    hidden_sizes = (32, 32)
    if len(sys.argv) > 1:
        load_model_path = sys.argv[1]
        load_model_data = pkl.load(open(load_model_path, 'rb'))
        hidden_sizes = load_model_data['hidden_sizes']

    # use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # Create the Create2 mover environment
    env = Create2MoverEnv(90, port='/dev/ttyUSB0', obs_history=1, dt=0.15, random_state=rand_state)
    env = NormalizedEnv(env)

    # Start environment processes
    env.start()

    # Create baselines TRPO policy function
    sess = U.single_threaded_session()
    sess.__enter__()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=hidden_sizes[0], num_hid_layers=len(hidden_sizes))

    # Create and start plotting process
    plot_running = Value('i', 1)
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], })
    # Spawn plotting process
    pp = Process(target=plot_create2_mover, args=(env, 2048, shared_returns, plot_running))
    pp.start()

    # Create callback function for logging data from baselines TRPO learn
    kindred_callback = create_callback(shared_returns, load_model_data)

    # Train baselines TRPO
    learn(env, policy_fn,
          max_timesteps=40000,
          timesteps_per_batch=2048,
          max_kl=0.05,
          cg_iters=10,
          cg_damping=0.1,
          vf_iters=5,
          vf_stepsize=0.001,
          gamma=0.995,
          lam=0.995,
          callback=kindred_callback
          )

    # Safely terminate plotter process
    plot_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    env.close()


def plot_create2_mover(env, batch_size, shared_returns, plot_running):
    """Helper process for visualize the learning curve and observations.

    Args:
        env: An instance of Create2MoverEnv
        batch_size: An int representing timesteps_per_batch provided to the PPO learn function
        shared_returns: A manager dictionary object containing `episodic returns` and `episodic lengths`
        plot_running: A multiprocessing Value object containing 0/1.
            1: Continue plotting, 0: Terminate plotting loop
    """
    print("Started plotting routine")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    plt.ion()
    time.sleep(5.0)

    fig = plt.figure(figsize=(20, 7))
    gs = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(gs[0, :-1])
    ax2 = plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[1, :])

    # setup main plot showing the current state
    sensor_array_type = env._sensor_comms[env._comm_name].sensor_buffer.np_array_type
    sensor_array_dim = len(sensor_array_type)
    action_array_dim = env._action_buffer.array_len
    ax1_bars = ax1.bar(list(range(sensor_array_dim + action_array_dim)),
                       [0] * (sensor_array_dim + action_array_dim),
                       tick_label=list(sensor_array_type.names) +
                                  ['requested left wheel speed', 'requested right wheel speed'])
    ax1_min_y = [create2_config.PACKET_INFO[create2_config.PACKET_NAME_TO_ID[packet]]['range'][0] for packet in
                 sensor_array_type.names]
    ax1_min_y.extend([r[0] for r in
                      create2_config.OPCODE_INFO[create2_config.OPCODE_NAME_TO_CODE[env._main_op]]['params'].values()])
    ax1_max_y = [create2_config.PACKET_INFO[create2_config.PACKET_NAME_TO_ID[packet]]['range'][1] for packet in
                 sensor_array_type.names]
    ax1_max_y.extend([r[1] for r in
                      create2_config.OPCODE_INFO[create2_config.OPCODE_NAME_TO_CODE[env._main_op]]['params'].values()])
    ax1.set_ylim([-1.25, 1.25])
    ax1.tick_params(axis='x', labelrotation=10, labelsize=7)
    ax1.set_title("Current Sensor & Action")

    ax1_texts = []
    for b in ax1_bars:
        ax1_texts.append(ax1.text(b.get_x() + b.get_width() / 2., 1.05 * b.get_height(), '0', ha='center', va='bottom'))

    hl11, = ax2.plot([], [])

    sensation_array_dim = env._sensation_buffer.array_len - 2
    ax3_bars = ax3.bar(list(range(sensation_array_dim)), [0] * sensation_array_dim)
    ax3_min_y = env.observation_space.low
    ax3_max_y = env.observation_space.high
    ax3_texts = []
    for b in ax3_bars:
        ax3_texts.append(ax3.text(b.get_x() + b.get_width() / 2., 1.05 * b.get_height(), '0', ha='center', va='bottom'))
    ax3.set_ylim([-1.25, 1.25])
    ax3.set_title("Current Sensation")

    count = 0
    old_size = len(shared_returns['episodic_returns'])
    while plot_running.value:
        sensor_buffer = env._sensor_comms[env._comm_name].sensor_buffer.read()
        action_buffer = env._action_buffer.read()
        sensation_buffer = env._sensation_buffer.read()
        for i, b in enumerate(ax1_bars):
            ax1_texts[i].remove()
            if i >= len(sensor_buffer[0][0][0]):
                raw_val = action_buffer[0][0][-1 * (len(ax1_bars) - i)]
            else:
                raw_val = sensor_buffer[0][0][0][i]
            if raw_val < 0:
                height = raw_val / abs(ax1_min_y[i])
            else:
                height = raw_val / ax1_max_y[i]
            b.set_height(max(-1.0, min(1.0, height)))
            ax1_texts[i] = ax1.text(b.get_x() + b.get_width() / 2., 1.05 * b.get_height(), "{:.2f}".format(raw_val),
                                    ha='center', va='bottom')

        for i, b in enumerate(ax3_bars):
            ax3_texts[i].remove()
            raw_val = sensation_buffer[0][0][i]
            if raw_val < 0:
                height = raw_val / abs(ax3_min_y[i])
            else:
                height = raw_val / ax3_max_y[i]
            b.set_height(max(-1.0, min(1.0, height)))
            ax3_texts[i] = ax3.text(b.get_x() + b.get_width() / 2., 1.05 * b.get_height(), "{:.2f}".format(raw_val),
                                    ha='center', va='bottom')
        ax3.set_title("Current Sensation (Reward: {:.2f})".format(sensation_buffer[0][0][-2]))

        # make a copy of the whole dict to avoid episode_returns and episodic_lengths getting desync
        # while plotting
        copied_returns = copy.deepcopy(shared_returns)
        if not copied_returns['write_lock'] and len(copied_returns['episodic_returns']) > old_size:
            # plot learning curve
            returns = np.array(copied_returns['episodic_returns'])
            old_size = len(copied_returns['episodic_returns'])
            window_size_steps = 5000
            x_tick = 1000

            if copied_returns['episodic_lengths']:
                ep_lens = np.array(copied_returns['episodic_lengths'])
            else:
                ep_lens = batch_size * np.arange(len(returns))
            cum_episode_lengths = np.cumsum(ep_lens)

            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)
                rets = []

                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))

                hl11.set_xdata(np.arange(1, len(rets) + 1) * x_tick)
                ax2.set_xlim([x_tick, len(rets) * x_tick])
                hl11.set_ydata(rets)
                ax2.set_ylim([np.min(rets), np.max(rets) + 1])
        time.sleep(0.01)
        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.tight_layout(rect=[0, 0.01, 1, 0.99])

        count += 1


if __name__ == '__main__':
    main()



