# Examples

This folder contains some example scripts for running SenseAct in various environments.

## Basic

Run [basic/sim_double_pendulum.py](basic/sim_double_pendulum.py) to start a simple random agent on the simulated Double Inverted Pendulum Environment for 1000 episodes. In addition to the standard SenseAct requirements, this environment also requires mujoco 1.50. You should see a _mujoco_py_ window pop up showing a 3D rendering of the pendulum when you run the example.

## Advanced

We also provided examples that use the OpenAI Baselines implementation of Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO) for actual learning. To run these examples, first install the Baselines prerequisites shown on OpenAI Baselines [README](https://github.com/openai/baselines), then install Baselines version 0.1.5 by running:

`pip install baselines==0.1.5`

Each examples also require its corresponding hardware, with the exception of [advanced/sim_double_pendulum.py](advanced/sim_double_pendulum.py), which requires mujoco 1.50.

### Example list
Below is a list of examples provided with a learning agent. Each is marked in terms of difficulty of hardware and physical environment setup.
* (Simple) Double Inverted Pendulum environment: [advanced/sim_double_pendulum.py](advanced/sim_double_pendulum.py)
* (Simple) DXL Reacher environment: [advanced/dxl_reacher.py](advanced/dxl_reacher.py)
* (Simple) DXL Tracker environment: [advanced/dxl_tracker.py](advanced/dxl_tracker.py)
* (Moderate) UR5 Reacher 2D environment: [advanced/ur5_reacher.py](advanced/ur5_reacher.py)
* (Moderate) UR5 Reacher 6D environment: [advanced/ur5_reacher_6D.py](advanced/ur5_reacher_6D.py)
* (Complex) Create2 Mover environment: [advanced/create2_mover.py](advanced/create2_mover.py)
* (Complex) Create2 Docker environment: [advanced/create2_docker.py](advanced/create2_docker.py)
