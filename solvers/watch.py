"""
Lets to watch how trained agents deal with environment.
"""
import sys
from datetime import datetime
from parser import watch_parser

import gym
from config import watch_config
from logger import get_logger
from RLAgent import RLAgent


def show_policy(rl_agent: RLAgent, size: int = 8) -> None:
    """
    Visualize agent policy for FrozenLake environments.

    Prints characters that looks like maximum action for each possible state. Description:
    '<' - left
    '^' - up
    '>' - right
    '.' - down

    Args:
        rl_agent: Trained agent.
        size: Size of area in chosen environment along single dimension.
    """
    actions_viz = {
        0: "<",
        1: ".",
        2: ">",
        3: "^",
    }
    actions = [rl_agent.get_action(i) for i in range(size ** 2)]
    viz = "".join(list(map(actions_viz.get, actions)))  # type: ignore
    for i in range(size):
        print(viz[i * size : (i + 1) * size])


if __name__ == "__main__":
    # Setting parsing and configuration
    timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
    logger = get_logger(sys.argv[0], timestamp)
    cli_args = watch_parser().parse_args()
    algorithm, model_path, episodes_count, max_steps_in_episode, is_slippery = watch_config(
        cli_args
    )

    # Environment and agent creation
    env = gym.make("FrozenLake8x8-v0", is_slippery=is_slippery)
    agent = algorithm.load(model_path)

    # Rollout
    total_rewards = 0
    for episode in range(episodes_count):
        current_state = env.reset()
        env.render()

        for _ in range(max_steps_in_episode):
            action = agent.get_action(current_state)
            print(action)
            current_state, reward, done, info = env.step(action)
            env.render()

            total_rewards += reward
            if done:
                if reward:
                    logger.info("Success! Frisbee has been recovered!")
                else:
                    logger.info("Failure! Blub, blub, blub...")
                break
    logger.info(
        f"Wins: {total_rewards}/{episodes_count} ({100 * total_rewards / episodes_count:.2f}%)"
    )
    show_policy(agent)
