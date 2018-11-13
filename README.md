# SenseAct: A computational framework for real-world robot learning tasks

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

This repository provides the implementation of several reinforcement learning tasks with multiple real-world robots.
These tasks come with an interface similar to [OpenAI-Gym](https://github.com/openai/gym) so that learning algorithms can be plugged in easily and in a uniform manner across tasks.
All the tasks here are implemented based on a computational framework of robot-agent communication proposed by Mahmood et al. (2018a), which we call *SenseAct*.
In this computational framework, agent and environment-related computations are ordered and distributed among multiple concurrent processes in a specific way. By doing so, SenseAct enables the following:

- Timely communication between the learning agent and multiple robotic devices with reduced latency,
- Easy and systematic design of robotic tasks for reinforcement learning agents,
- Facilitate reproducible real-world reinforcement learning.

This repository provides the following real-world robotic tasks, which are proposed by Mahmood et al. (2018b) as benchmark tasks for reinforcement learning algorithms:

### Universal-Robots (UR) robotic arms:
Tested on UR Software v. 3.3.4.310
- [UR-Reacher](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/ur/reacher_env.py) (both 2 joint and 6 joint control)

| ![UR-Reacher-2](docs/ur-reacher-2-trpo.gif) <br> UR-Reacher-2 | ![UR-Reacher-6](docs/ur-reacher-6-trpo.gif) <br /> UR-Reacher-6 |
| --- | --- |

### Dynamixel (DXL) actuators:
Currently we only support MX-64AT.
- [DXL-Reacher](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/dxl/dxl_reacher_env.py)
- [DXL-Tracker](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/dxl/dxl_tracker_env.py)

| ![DXL-Reacher](docs/dxl-reacher-trpo.gif) <br/>DXL-Reacher | ![DXL-Tracker](docs/dxl-tracker-trpo.gif)<br /> DXL-Tracker |
| --- | --- |

### iRobot Create 2 robots:
- [Create-Mover](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/create2/create2_mover_env.py)
- [Create-Docker](https://github.com/kindredresearch/SenseAct/blob/master/senseact/envs/create2/create2_docker_env.py)

| ![Create-Mover](docs/create-mover-ppo.gif) <br />Create-Mover | ![Create-Docker](docs/create-docker-trpo.gif) <br /> Create-Docker |
| --- | --- |

Mahmood et al. (2018b) provide extensive results comparing multiple reinforcement learning algorithms on the above tasks, and Mahmood et al. (2018a) show the effect of different task-setup elements in learning. Their results can be reproduced by using this repository (see [documentation](docs/) for more information).

# Installation

SenseAct uses Python3 (>=3.5), and all other requirements are automatically installed via pip.

On Linux and Mac OS X, run the following:
1. `git clone https://github.com/kindredresearch/SenseAct.git`
1. `cd SenseAct`
1. `pip install -e .` or `pip3 install -e .` depends on your setup

Additional instruction for installing [OpenAI Baselines](https://github.com/openai/baselines) needed for running the [advanced examples](examples/advanced) is given in the [corresponding readme](examples/).

### Additional installation steps for Dynamixel-based tasks (Linux only)

Dynamixels can be controlled by drivers written using either ctypes by [Robotis](https://github.com/ROBOTIS-GIT/DynamixelSDK/releases/tag/3.5.4) or pyserial, which can be chosen by passing either `True` (ctypes) or `False` (pyserial) as an argument to the `use_ctypes_driver` parameter of a Dynamixel-based task (e.g., see `examples/advanced/dxl_reacher.py`). We found the ctypes-based driver to provide substantially more timely and precise communication compared to the pyserial-based one.

In order to use the CType-based driver, we need to install gcc and relevant packages for compiling the C libraries:

`sudo apt-get install gcc-5 build-essential gcc-multilib g++-multilib`

Then run the following script to download and compile the Dynamixel driver C libraries:

`sudo bash setup_dxl.sh`

For additional setup and troubleshooting information regarding Dynamixels, please see [DXL Docs](senseact/devices/dxl/).

### Tests

You can check whether SenseAct is installed correctly by running the included unit tests.

```bash
cd SenseAct
python -m unittest discover -b
```

# Support

Installation problems? Feature requests? General questions?
* read through examples and API [documentation](https://github.com/kindredresearch/SenseAct/tree/master/docs)
* create github issues on the [SenseAct project](https://github.com/kindredresearch/SenseAct)
* join the mailing list [https://groups.google.com/forum/#!forum/senseact](https://groups.google.com/forum/#!forum/senseact)

# Acknowledgments

This project is developed by the [Kindred](https://www.kindred.ai/) AI Research team. [Rupam Mahmood](https://github.com/armahmood), [Dmytro Korenkevych](https://github.com/dkorenkevych), and [Brent Komer](https://github.com/bjkomer) originally developed the computational framework and the UR tasks. [William Ma](https://github.com/williampma) developed the Create 2 tasks and contributed substantially by adding new features to SenseAct. [Gautham Vasan](https://github.com/gauthamvasan) developed the DXL tasks. [Francois Hogan](https://github.com/fhogan) developed the simulated task.

[James Bergstra](https://github.com/jaberg) provided support and guidance throughout the development. [Adrian Martin](https://github.com/adrianheron), [Scott Rostrup](https://github.com/sarostru), and [Jonathan Yep](https://github.com/JonathanYep) developed the pyserial DXL driver for a Kindred project, which was used for the SenseAct DXL Communicator. [Daniel Snider](https://github.com/danielsnider), [Oliver Limoyo](https://github.com/Olimoyo), [Dylan Ashley](https://github.com/dylanashley), and [Craig Sherstan](https://github.com/csherstan) tested the framework, provided thoughtful suggestions, and confirmed the reproducibility of learning by running experiments on real robots.

# Citing SenseAct

For the SenseAct computational framework and the UR-Reacher tasks, please cite Mahmood et al. (2018a). For the DXL and the Create 2 tasks, please cite Mahmood et al. (2018b).

* Mahmood, A. R., Korenkevych, D., Komer,B. J., Bergstra, J. (2018a). [Setting up a reinforcement learning task with a real-world robot](https://arxiv.org/abs/1803.07067). In *IEEE/RSJ International Conference on Intelligent Robots and Systems*.

* Mahmood, A. R., Korenkevych, D., Vasan, G., Ma, W., Bergstra, J. (2018b). [Benchmarking reinforcement learning algorithms on real-world robots](https://arxiv.org/abs/1809.07731). In *Proceedings of the 2nd Annual Conference on Robot Learning*.
