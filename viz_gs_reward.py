import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import torch
import dataset
import models
import gym
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_demo_files', nargs='+', required=True)
# parser.add_argument('--demo_labels', nargs='+', type=int, required=True)
parser.add_argument('--output_model_path', required=True)
parser.add_argument('--mode', default='state_only')
parser.add_argument('--dataset_mode', default='partial')
parser.add_argument('--traj_len', type=int, default=10)
parser.add_argument('--env_name', default='Reacher-v1')
args = parser.parse_args()

# ----- Environment -----
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# ----- Dataset -----
pair_num = 5
if args.dataset_mode == 'partial':
    pair_nums = [pair_num] * len(args.test_demo_files)
    test_dataset = dataset.TrajDataset(
        args.test_demo_files, pair_nums, num_inputs, num_actions, 
        mode=args.mode, traj_len=args.traj_len)
elif args.dataset_mode == 'traj':
    test_dataset = dataset.RankingTrajDataset(
        args.test_demo_files, None, num_inputs, num_actions, mode=args.mode)
else:
    raise NotImplementedError

input_dim = {
    'state_only': num_inputs,
    'state_action': num_inputs + num_actions,
    'state_pair': num_inputs * 2
}[args.mode]

# ----- Load Model -----
reward_net = models.GaussianRewardNet(input_dim).float()
reward_net.load_state_dict(torch.load(args.output_model_path, map_location='cpu'))
reward_net.eval()

# ----- Collect Embeddings and Rewards -----
raw_points, mus, vars_, labels = [], [], [], []
for demo_idx in range(len(test_dataset)):
    traj, label = test_dataset[demo_idx]
    for step in traj:
        x = step.unsqueeze(0).float()
        with torch.no_grad():
            mu, sigma = reward_net(x)
        raw_points.append(x.squeeze(0).numpy())
        mus.append(mu.item())
        vars_.append(sigma.item()**2)
        labels.append(label)

raw_points = np.array(raw_points)
mus = np.array(mus)
vars_ = np.array(vars_)
labels = np.array(labels)

# ----- PCA -----
pca = PCA(n_components=2)
pca_pts = pca.fit_transform(raw_points)
x, y = pca_pts[:, 0], pca_pts[:, 1]

# ----- Interpolation Grid -----
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), 100),
    np.linspace(y.min(), y.max(), 100)
)

mu_grid = griddata((x, y), mus, (grid_x, grid_y), method='cubic')
var_grid = griddata((x, y), vars_, (grid_x, grid_y), method='cubic')


# Clamp outliers to avoid spikes
mu_mean = np.mean(mus)
mu_std = np.std(mus)
mu_max_threshold = mu_mean + 3 * mu_std
mu_min_threshold = mu_mean - 3 * mu_std
mu_grid = np.clip(mu_grid, a_min=mu_min_threshold, a_max=mu_max_threshold)

var_mean = np.mean(vars_)
var_std = np.std(vars_)
var_max_threshold = var_mean + 3 * var_std
var_grid = np.clip(var_grid, a_min=0, a_max=var_max_threshold)

# ----- Common Plot Function -----
def plot_field(title, grid, cmap, cbar_label, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(grid_x, grid_y, grid, levels=20, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)

    sc = ax.scatter(x, y, c=labels, cmap='Set1', s=30, edgecolors='none', alpha=1)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title(title)
    ax.grid(True)
    ax.legend(*sc.legend_elements(), title="Demonstrators")
    fig.tight_layout()
    fig.savefig(filename)
    plt.show()
    plt.close()

# ----- Plot Reward Mean -----
plot_field(
    title='PCA of Demonstrations with Reward Mean Field',
    grid=mu_grid,
    cmap='coolwarm',
    cbar_label='Reward Mean',
    filename='reward_mean_field_250_test.png'
)

# ----- Plot Reward Variance -----
plot_field(
    title='PCA of Demonstrations with Reward Variance Field',
    grid=var_grid,
    cmap='coolwarm',
    cbar_label='Reward Variance',
    filename='reward_variance_field_250_test.png'
)