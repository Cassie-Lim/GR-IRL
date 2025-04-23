import argparse, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import gym

import models  # assume GaussianRewardNet is in models.py
import dataset
from torch.utils.tensorboard import SummaryWriter
counter = 0
while True:
    log_dir = f'log/runs/grirl_{counter}'  # for tensorboard
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        break
    counter += 1
writer = SummaryWriter(log_dir=log_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Reacher-v1")
parser.add_argument('--seed', type=int, default=543)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=1)
parser.add_argument('--train_demo_files', nargs='+')
parser.add_argument('--test_demo_files', nargs='+')
parser.add_argument('--train_traj_nums', nargs='+', type=int)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--mode', default='state_only')
parser.add_argument('--dataset_mode', default='partial')
parser.add_argument('--output_model_path')
parser.add_argument('--traj_len', type=int)
args = parser.parse_args()
args.output_model_path = log_dir + '/model_ckpt'
os.makedirs(args.output_model_path, exist_ok=True)
use_gpu = torch.cuda.is_available()

env = gym.make(args.env_name)
test_env = gym.make(args.env_name)
env.seed(args.seed)
torch.manual_seed(args.seed)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# Load dataset
if args.dataset_mode == 'partial':
    train_dataset = dataset.RankingLimitDataset(args.train_demo_files, args.train_traj_nums, num_inputs, num_actions, mode=args.mode, traj_len=args.traj_len)
    test_dataset = dataset.RankingLimitDataset(args.test_demo_files, None, num_inputs, num_actions, mode=args.mode, traj_len=args.traj_len)
elif args.dataset_mode == 'traj':
    train_dataset = dataset.RankingTrajDataset(args.train_demo_files, args.train_traj_nums, num_inputs, num_actions, mode=args.mode)
    test_dataset = dataset.RankingTrajDataset(args.test_demo_files, None, num_inputs, num_actions, mode=args.mode)
else:
    raise NotImplementedError

train_loader = data_utils.DataLoader(train_dataset, collate_fn=dataset.rank_collate_func, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data_utils.DataLoader(test_dataset, collate_fn=dataset.rank_collate_func, batch_size=1, shuffle=True, num_workers=4)

# Reward model
input_dim = {
    'state_only': num_inputs,
    'state_action': num_inputs + num_actions,
    'state_pair': num_inputs * 2
}[args.mode]

reward_net = models.GaussianRewardNet(input_dim).float()
if use_gpu:
    reward_net = reward_net.cuda()

optimizer = optim.Adam(reward_net.parameters(), lr=1e-3, weight_decay=5e-4)

def compute_traj_reward(traj, net):
    mu_list, var_list = [], []
    for t in traj:
        mu, sigma = net(t)
        mu_list.append((mu).sum(dim=0, keepdim=True))
        var_list.append((sigma ** 2).sum(dim=0, keepdim=True)) 
    traj_mu = torch.cat(mu_list, dim=0)  # shape (B, 1), sum of means for each trajectory
    traj_var = torch.cat(var_list, dim=0)
    return traj_mu, traj_var

best_acc = 0
for epoch in range(args.num_epochs):
    # --- Eval ---
    if epoch % args.save_interval == 0:
        acc_counter, counter = 0, 0
        for iter_, data in enumerate(test_loader):
            traj1, rew1, traj2, rew2 = data
            if use_gpu:
                traj1, rew1, traj2, rew2 = [t.cuda() for t in traj1], rew1.cuda(), [t.cuda() for t in traj2], rew2.cuda()

            mu1, var1 = compute_traj_reward(traj1, reward_net)
            mu2, var2 = compute_traj_reward(traj2, reward_net)

            score1 = mu1 + 0.5 * var1
            score2 = mu2 + 0.5 * var2

            pred_rank = (score1 < score2)
            true_rank = (rew1 < rew2)
            acc_counter += torch.sum(pred_rank == true_rank).item()
            counter += len(traj1)

            if iter_ > 10000:
                break

        acc = acc_counter / counter
        print(f"[Eval] Epoch {epoch}, Accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(reward_net.state_dict(), os.path.join(args.output_model_path, f'reward_net_best_{epoch}.pth'))
        writer.add_scalar("Eval/Accuracy", acc, epoch)
        writer.add_scalar("Eval/Best_Accuracy", best_acc, epoch)

    # --- Train ---
    for iter_, data in enumerate(train_loader):
        traj1, rew1, traj2, rew2 = data
        if use_gpu:
            traj1, rew1, traj2, rew2 = [t.cuda() for t in traj1], rew1.cuda(), [t.cuda() for t in traj2], rew2.cuda()

        mu1, var1 = compute_traj_reward(traj1, reward_net)
        mu2, var2 = compute_traj_reward(traj2, reward_net)

        score1 = mu1 + 0.5 * var1
        score2 = mu2 + 0.5 * var2

        logit_diff = score1 - score2
        label      = (rew1 > rew2).float().view(-1, 1)

        loss = torch.nn.BCEWithLogitsLoss()(logit_diff, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_ % args.log_interval == 0:
            print(f"[Train] Epoch {epoch}, Iter {iter_}, Loss = {loss.item():.4f}")
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + iter_)
        if iter_ > 5000:
            break
    writer.close()