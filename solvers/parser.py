import argparse


def train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        help="Name of algorithm which should be used in training " "[QLearning, DQN]",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        required=True,
        help="Path to configuration file with hyperparameters and other"
        "settings. Default directory for this option is <root>/configs/",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Name under which trained model should be saved. Default directory"
        "for this option is <root>/trained_models/",
    )
    return parser


def watch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        required=True,
        help="Name of algorithm which should be used in training" "[QLeraning, DQN]",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to trained model.")
    parser.add_argument("-e", "--episode-count", type=int, help="Count of episodes to rollout.")
    parser.add_argument(
        "--max-steps-in-episode",
        type=int,
        help="Maximal number of steps in signle episode before termination.",
    )
    return parser
