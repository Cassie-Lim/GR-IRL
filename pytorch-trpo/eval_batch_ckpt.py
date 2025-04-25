import argparse
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import re
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from models_ac import Policy
import custom_envs.hopper
import custom_envs.walker2d
from scipy.ndimage import gaussian_filter1d
torch.set_default_tensor_type('torch.DoubleTensor')

def select_action(policy_net, state):
    state = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        action_mean, _, action_std = policy_net(state)
        action = torch.normal(action_mean, action_std)
    return action.squeeze(0).numpy()

def evaluate(policy_net, env, render=False, episodes=5):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            if render:
                env.render()
            action = select_action(policy_net, state)
            state, reward, done, _ = env.step(action)
            total += reward
        rewards.append(total)
    return np.mean(rewards), np.std(rewards)

def extract_step(filename):
    match = re.search(r'epoch_(\d+)_rew', filename)
    return int(match.group(1)) if match else -1


def plot_with_shaded_std(ax, x, y, y_std, label):
    y_smoothed = gaussian_filter1d(y, sigma=1)
    y_std_smoothed = gaussian_filter1d(y_std, sigma=1)
    ax.plot(x, y_smoothed, label=label)
    ax.fill_between(x, y_smoothed - y_std_smoothed, y_smoothed + y_std_smoothed, alpha=0.2)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")  # avoids GUI backend issues

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--render-one', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env_name)
    env.reset(seed=123)
    torch.manual_seed(123)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    result_pairs = [
        ('pytorch-trpo/the_log_path/TREX/CustomHalfCheetah/0', 'TREX'),
        ('pytorch-trpo/the_log_path/GRIRL/CustomHalfCheetah/0', 'GRIRL'),
    ]
    for dir_path, label in result_pairs:
        ckpt_files = sorted(
            [f for f in os.listdir(dir_path) if f.endswith('.pth')],
            key=extract_step
        )[:150]

        steps, means, stds = [], [], []

        for i, fname in enumerate(ckpt_files):
            fpath = os.path.join(dir_path, fname)
            step = extract_step(fname)
            if step == -1:
                continue

            policy_net = Policy(num_inputs, num_actions)
            checkpoint = torch.load(fpath, map_location='cpu')
            policy_net.load_state_dict(checkpoint['policy'])
            policy_net.eval()

            mean_rew, std_rew = evaluate(policy_net, env, render=(args.render_one and i == len(ckpt_files)-1), episodes=args.episodes)

            print(f"[{label}] Step {step} | Mean: {mean_rew:.2f}, Std: {std_rew:.2f}")
            steps.append(step)
            means.append(mean_rew)
            stds.append(std_rew)

        if steps:
            sorted_indices = np.argsort(steps)
            steps = np.array(steps)[sorted_indices]
            means = np.array(means)[sorted_indices]
            stds = np.array(stds)[sorted_indices]
            plot_with_shaded_std(ax, steps, means, stds, label=label)

    ax.set_title(f'Evaluation Reward vs Training Step ({args.env_name})')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Return')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig('eval_reward_comparison_halfcheetah_150.png')
    print("Saved: eval_reward_comparison.png")