import os
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import yaml
from dqn import dqnAgent
from qlearning import QLearningAgent

ALGORITHMS = {"QLearning": QLearningAgent, "DQN": dqnAgent}

PROJECT_PATH = Path(os.path.abspath(".")).parent
CONFIGS_PATH = PROJECT_PATH / "configs"
TRAINED_MODELS_PATH = PROJECT_PATH / "trained_models"
OUT_PATH = PROJECT_PATH / "out"


def train_config(cli_args: Namespace) -> Tuple:
    agent = ALGORITHMS[cli_args.algorithm]
    with open(str(CONFIGS_PATH / cli_args.config), "r") as f:
        config = yaml.safe_load(f)
    output_name = cli_args.algorithm if cli_args.output_path is None else cli_args.output_path
    output_path = TRAINED_MODELS_PATH / output_name
    return agent, config, output_path


def watch_config(cli_args: Namespace) -> Tuple:
    agent = ALGORITHMS[cli_args.algorithm]
    model_path = TRAINED_MODELS_PATH / cli_args.model
    episode_count = cli_args.episode_count if cli_args.episode_count is not None else 5
    max_steps_in_episode = (
        cli_args.max_steps_in_episode if cli_args.max_steps_in_episode is not None else 1000
    )
    return agent, model_path, episode_count, max_steps_in_episode
