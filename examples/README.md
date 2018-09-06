## Examples

This folder contains some example scripts for running SenseAct in various environments.

### Basic

Run [basic/sim_double_pendulum.py](basic/sim_double_pendulum.py) to start a simple random agent on the simulated Double Inverted Pendulum Environment for 1000 episodes. In addition to the standard SenseAct requirements, this environment also requires mujoco 1.50. You should see a _mujoco_py_ window pop up showing a 3D rendering of the pendulum.

### Advance

We also provided examples that use the OpenAI Baselines implementation of Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO) for actual learning. To run these examples, first install the Baselines prerequisites shown on OpenAI Baselines [README](https://github.com/openai/baselines), then install Baselines version 0.1.5 by running `pip install baselines==0.1.5`. Each examples also require its corresponding hardware, with the exception of [advance/sim_double_pendulum.py](advance/sim_double_pendulum.py), which requires mujoco 1.50.

#### Example list
* Create2 Docker environment: [advance/create2_docker.py](advance/create2_docker.py)
* Create2 Mover environment: [advance/create2_mover.py](advance/create2_mover.py)
* DXL Reacher environment: [advance/dxl_reacher.py](advance/dxl_reacher.py)
* DXL Tracker environment: [advance/dxl_tracker.py](advance/dxl_tracker.py)
* Double Inverted Pendulum environment: [advance/sim_double_pendulum.py](advance/sim_double_pendulum.py)
* UR5 Reacher environment: [advance/ur5_reacher.py](advance/ur5_reacher.py)
