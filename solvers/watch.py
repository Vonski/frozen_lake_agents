from parser import watch_parser

import gym
from config import watch_config


def main():
    cli_args = watch_parser().parse_args()
    algorithm, model_path, episodes_count, max_steps_in_episode = watch_config(cli_args)

    env = gym.make("FrozenLake8x8-v0")

    agent = algorithm.load(model_path)

    total_rewards = 0
    for episode in range(episodes_count):
        current_state = env.reset()
        env.render()

        for _ in range(max_steps_in_episode):
            action = agent.get_action(current_state)
            current_state, reward, done, info = env.step(action)
            env.render()

            total_rewards += reward
            if done:
                if reward:
                    print("Success! Frisbee has been recovered!")
                else:
                    print("Failure! Blub, blub, blub...")
                break
    print(f"Wins: {total_rewards}/{episodes_count} ({100 * total_rewards / episodes_count:.2f}%)")


if __name__ == "__main__":
    main()
