import argparse
import os
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libGLEW.so'
import torch
import gym
import numpy as np
from models_ac import Policy
import custom_envs.hopper
import custom_envs.walker2d

torch.set_default_tensor_type('torch.DoubleTensor')

def select_action(policy_net, state):
    state = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        action_mean, _, action_std = policy_net(state)
        action = torch.normal(action_mean, action_std)
    return action.squeeze(0).numpy()

def evaluate(policy_net, env, render=True, episodes=5):
    total_rewards = []
    custom_rewards = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_rew = 0

        while not done:
            if render:
                env.render()

            action = select_action(policy_net, state)
            next_state, reward, done, _ = env.step(action)

            episode_rew += reward


            state = next_state

        total_rewards.append(episode_rew)

    env.close()
    print(f"Avg Env Reward: {np.mean(total_rewards):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, required=True, help='Environment name (e.g. Hopper-v3)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to saved policy checkpoint')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.seed(123)
    torch.manual_seed(123)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    policy_net = Policy(num_inputs, num_actions)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    policy_net.load_state_dict(checkpoint['policy'])
    policy_net.eval()

    evaluate(policy_net, env, render=args.render)
