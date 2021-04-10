from parser import train_parser

import gym
import numpy as np
from config import train_config


def main():
    cli_args = train_parser().parse_args()
    algorithm, config, output_path = train_config(cli_args)

    env = gym.make("FrozenLake8x8-v0")

    state_space_size = env.observation_space.n
    action_space_size = env.action_space.n
    print(action_space_size, state_space_size)

    agent = algorithm(state_space_size, action_space_size, **config)

    epsilon = 0
    for epoch in range(config["epochs_count"]):
        current_state = env.reset()
        total_rewards = 0
        completed_epsiodes = 0

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
                completed_epsiodes += 1
                current_state = env.reset()
        print(
            f"Epoch {epoch}: {100 * total_rewards / (completed_epsiodes):.2f}% ({int(total_rewards)}/{completed_epsiodes})"
        )

        epsilon = config["min_epsilon"] + (config["max_epsilon"] - config["min_epsilon"]) * np.exp(
            -config["epsilon_decay"] * epoch
        )

    agent.save(output_path)


if __name__ == "__main__":
    main()
