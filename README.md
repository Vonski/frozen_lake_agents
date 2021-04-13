# FrozenLake agents
Contains RL solutions for the FrozenLake8x8-v0 gym environment.

## Description
The task was to create an RL solution for the FrozenLake8x8-v0 gym environment.

I created two solutions for two different agents for two versions of this environment (the slippery one and the stable one).

QLearning with a lookup table was used to solve the slippery one. To solve the stable one I wrote Deep Q Network in torch (CPU only).

## Directory structure
- `configs` - contains configuration files for `train.py` script for both algorithms.
- `outs` - contains output files from scripts in `solvers`
        - logs - in this directory logs of `train.py` and `watch.py` scripts are generated.
        - plots - in this directory plots of learning progress from `train.py` script are generated.
- `solvers` - contains python files that define algorithms and other utilities. `train.py` and `watch.py` are located there.
- `trained_models` - contains serialized models. `train.py` settings of output path default to this directory.
- `requirements.txt` - contains the list of libraries needed to run scripts successfully.

## Setup
To prepare an environment to run scripts from this repo install requirement (I recommend using virtualenv):
```
virtualenv venv  # optional
. venv/bin/activate  # optional
pip install -r requirements.txt
```
**Note:** I used 3.7.10 version of python. Just in case.

## Running scripts
Scripts `train.py` and `watch.py` take command line arguments. You can check their meaning using `python train.py --help` and `python watch.py --help` respectively.

Example minimal commands to run these scripts with different parameters are shown below:
```
python train.py -a QLearning -c qlearning.yaml
python train.py -a DQN -c dqn.yaml
python watch.py -a QLearning -m Final_QLearning_best.npy
python watch.py -a DQN -m Final_DQN_best.pt -ns
```

## Code
It is documented with docstrings and I hope it is self-explanatory.

For an explanation of what different parameters in config files mean check scripts `train.py` and `watch.py` and also agent classes descriptions, especially section *Attributes* in `qlearning.py` and `dqn.py`. All these files are located in the `solvers` directory.

