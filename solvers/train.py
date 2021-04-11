import sys
from datetime import datetime
from parser import train_parser

import gym
import numpy as np
from config import train_config
from logger import get_logger
from plot import save_lineplot_with_best_epoch_marked


def main():
    # Setting parsing and configuration
    timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M_%S")
    logger = get_logger(sys.argv[0], timestamp)
    cli_args = train_parser().parse_args()
    algorithm, config, output_path = train_config(cli_args)

    # Environment setup
    env = gym.make("FrozenLake8x8-v0")

    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    logger.info(f"{action_space_size}, {state_space_size}")

    # Agent creation and training
    agent = algorithm(state_space_size, action_space_size, **config)

    epsilon = 0
    best_epoch = 0
    agent_from_best_epoch = agent.copy()
    win_ratio_over_time = []
    for epoch in range(config["epochs_count"]):
        current_state = env.reset()
        total_rewards = 0
        completed_episodes = 0

        for _ in range(config["steps_per_epoch"]):
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action = agent.get_action(current_state)

            old_state = current_state
            current_state, reward, done, info = env.step(action)
            agent.learn(reward, action, current_state, old_state)

            total_rewards += reward
            if done:
                completed_episodes += 1
                agent.learn(reward, action, current_state, old_state, end_of_episode=True)
                current_state = env.reset()

        epsilon = config["min_epsilon"] + (config["max_epsilon"] - config["min_epsilon"]) * np.exp(
            -config["epsilon_decay"] * epoch
        )

        win_ratio = total_rewards / completed_episodes
        win_ratio_over_time.append(win_ratio)
        if win_ratio > win_ratio_over_time[best_epoch]:
            best_epoch = epoch
            agent_from_best_epoch = agent.copy()

        logger.info(
            f"Epoch {epoch}: {100 * win_ratio:.2f}% ({int(total_rewards)}/{completed_episodes})"
        )

    # Saving results
    agent.save(output_path)
    best_agent_output_path = output_path.parent / f"{output_path.name}_best"
    agent_from_best_epoch.save(best_agent_output_path)

    save_lineplot_with_best_epoch_marked(win_ratio_over_time, best_epoch, timestamp)


if __name__ == "__main__":
    main()
