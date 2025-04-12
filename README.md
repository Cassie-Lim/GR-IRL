# TREX-pytorch

The code is a pytorch implementation for the paper ['Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations'](https://arxiv.org/abs/1904.06387).

We currently implement the version with partial trajectories.

## Installation

`pip install -r requirements.txt`

`
cd custom_envs

pip install -e .
`

## Collecting Demonstrations

To collect demonstrations, we use the reinforcement learning code [here](https://github.com/ikostrikov/pytorch-trpo) to learn an optimal policy.

Then we use the checkpoints at different episode to collect demonstrations with different reward and then we can derive the ranking.


## Training

### The structure of the demonstrations files
Each demonstration file is a pickle file of a dict `{'traj':[traj_1, traj_2, ..., traj_N], 'reward':[reward_1, reward_2, ..., reward_N]}`

`traj1=[[np.concatenate([s_0,a_0], axis=0)], [np.concatenate([s_1,a_1], axis=0)], ..., [np.concatenate([s_T,dummy_action], axis=0)]]`

`reward_1=[R(s_0,a_0), ..., R(s_{T-1}, a_{T-1})]`


### Use all the partial trajectories 
<!-- 
`python train_trex.py --env-name Hopper-v3 --train_demo_files ./demo/Hopper-v3_noise_0.0_interval_1_rew_45.43.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_162.11.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_410.32.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_529.22.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_947.06.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_1678.93.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2096.12.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2977.82.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_3281.01.pt --test_demo_files ./demo/Hopper-v3_noise_0.0_interval_1_rew_45.43.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_162.11.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_410.32.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_529.22.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_947.06.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_1678.93.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2096.12.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2977.82.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_3281.01.pt --batch-size 64 --log-interval 100 --num_epochs 2000 --mode state_action --save-interval 10 --output_model_path log/hopper.pth --traj_len 50` -->


`python train_trex.py --env-name Hopper-v3 --train_demo_files [a list of demo for training] --test_demo_files [a list of demo for eval] --batch-size 64 --log-interval 1 --num_epochs 10 --mode state_action --save-interval 1 --output_model_path log/hopper.pth --traj_len 50`

Note that you might need to tune `num_epochs` for different envs.

### Use some of the partial trajectories (set the parameters `--train_traj_num`)
[TODO] Haven't tried yet, cannot gaurantee if it works. Don't seem to be urgent tho.

`python train_trex.py --env-name Hopper-v3 --train_demo_files ./demo/Hopper-v3_noise_0.0_interval_1_rew_45.43.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_162.11.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_410.32.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_529.22.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_947.06.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_1678.93.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2096.12.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2977.82.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_3281.01.pt --test_demo_files ./demo/Hopper-v3_noise_0.0_interval_1_rew_45.43.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_162.11.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_410.32.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_529.22.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_947.06.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_1678.93.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2096.12.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_2977.82.pt ./demo/Hopper-v3_noise_0.0_interval_1_rew_3281.01.pt --batch-size 64 --log-interval 100 --num_epochs 2000 --mode state_action --train_traj_nums 500 500 500 500 --save-interval 10 --output_model_path log/hopper.pth --traj_len 50`





## Testing

### Reward Visualization


`python

python viz_gs_reward.py \
  --test_demo_files [a list of demo for visualization]  \
  --output_model_path log/runs/grirl_0/model_ckpt/reward_net_best_14.pth \
  --dataset_mode partial \
  --mode state_action \
  --traj_len 50 \
  --env_name Hopper-v3

`
Note: better to choose demos whose rewards are distinctive enough.

### Reward Quality Eval


[TODO] This part of test code is not working right now.

To test the learned reward, we use the reinforcement learning code modified from [here](https://github.com/ikostrikov/pytorch-trpo).

`
cd pytorch-trpo

python main_trex.py --env-name Hopper-v3 --test-env-name Hopper-v3 --batch-size 15000 --save-interval 5 --reward_model ../log/hopper_trex.pth   --prefix 1 --output_path the_log_path --render
`


