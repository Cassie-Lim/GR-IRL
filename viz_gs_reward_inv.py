import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import dataset
import models
import gym
import os

parser = argparse.ArgumentParser()
parser.add_argument('--test_demo_files', nargs='+', required=True)
parser.add_argument('--output_model_path', required=True)
parser.add_argument('--mode', default='state_only')
parser.add_argument('--dataset_mode', default='partial')
parser.add_argument('--traj_len', type=int, default=10)
parser.add_argument('--env_name', default='Reacher-v1')
args = parser.parse_args()

# ----- Environment Setup -----
env = gym.make(args.env_name)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# ----- Dataset Loading -----
pair_num = 6
if args.dataset_mode == 'partial':
    pair_nums = [pair_num] * len(args.test_demo_files)
    test_dataset = dataset.RankingLimitDataset(
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

# ----- Load Trained Model -----
reward_net = models.GaussianRewardNet(input_dim).float()
reward_net.load_state_dict(torch.load(args.output_model_path, map_location='cpu'))
reward_net.eval()

# ----- Collect Stepwise Data -----
raw_points, mus, vars_, labels = [], [], [], []
for demo_idx in range(len(test_dataset)):
    traj, _, _, _ = test_dataset[demo_idx]
    traj = traj.float()
    label = demo_idx % pair_num
    with torch.no_grad():
        mu_batch, sigma_batch = reward_net(traj)
    raw_points.extend(traj.numpy())
    mus.extend(mu_batch.squeeze(1).numpy())
    vars_.extend((sigma_batch.squeeze(1).numpy()) ** 2)
    labels.extend([label] * len(traj))

raw_points = np.array(raw_points)
mus = np.array(mus)
vars_ = np.maximum(np.array(vars_), 1e-6)  # clamp variance to be positive
labels = np.array(labels)

# ----- PCA Projection -----
pca = PCA(n_components=2)
pca_pts = pca.fit_transform(raw_points)
x, y = pca_pts[:, 0], pca_pts[:, 1]

# ----- Build Dense Grid in PCA Space -----
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), 100),
    np.linspace(y.min(), y.max(), 100)
)
grid_pca = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # shape (10000, 2)

# ----- Inverse Transform to Original Feature Space -----
grid_orig = pca.inverse_transform(grid_pca)
grid_orig_tensor = torch.tensor(grid_orig, dtype=torch.float32)

# ----- Predict Rewards on Grid -----
with torch.no_grad():
    mu_grid_raw, sigma_grid_raw = reward_net(grid_orig_tensor)

mu_grid = mu_grid_raw.view(100, 100).numpy()
var_grid = (sigma_grid_raw.view(100, 100).numpy()) ** 2
var_grid = np.clip(var_grid, a_min=0, a_max=None)

# ----- Common Plotting Utility -----
def plot_field(title, grid, cmap, cbar_label, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw field first
    contour = ax.contourf(grid_x, grid_y, grid, levels=20, cmap=cmap, alpha=0.5)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)

    # Draw points second
    sc = ax.scatter(x, y, c=labels, cmap='tab10', s=60, edgecolors='none', alpha=0.95)
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
    cmap='viridis',
    cbar_label='Reward Mean',
    filename='reward_mean_field_vewr2.png'
)

# ----- Plot Reward Variance -----
plot_field(
    title='PCA of Demonstrations with Reward Variance Field',
    grid=var_grid,
    cmap='Reds',
    cbar_label='Reward Variance',
    filename='reward_variance_field_ver2.png'
)
