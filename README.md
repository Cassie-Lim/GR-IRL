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

Take HalfCheetah-v3 for exmaple. First train policy with
`
python pytorch-trpo/main_trex.py \
  --env-name HalfCheetah-v3 \
  --test-env-name HalfCheetah-v3 \
  --batch-size 15000 \
  --save-interval 1 \
  --prefix 1 \
  --output_path model_ckpt/
`
You will get a list of ckpts within `pytorch-trpo/model_ckpt/HalfCheetah`. Then manually select some ckpts that have distinguishable rewards, e.g.:

`python pytorch-trpo/generate_demos.py \
  --env-name HalfCheetah-v3 \
  --model-ckpts \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0000100_rew_660.51.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0000200_rew_987.16.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0000300_rew_1240.34.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0000400_rew_1391.88.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0000500_rew_1407.56.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0000700_rew_1790.42.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0001100_rew_2139.16.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0002200_rew_2481.77.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0003200_rew_2749.75.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0005300_rew_3147.82.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0006400_rew_3306.10.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0012200_rew_3670.78.pth \
  ./pytorch-trpo/model_ckpt/HalfCheetah/epoch_0019800_rew_4073.99.pth
`

This should give you a list of demos within `demo/HalfCheetah-v3`.

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


`python viz_gs_reward.py \
  --test_demo_files [a list of demo for visualization]  \
  --output_model_path log/runs/grirl_0/model_ckpt/reward_net_best_14.pth \
  --dataset_mode partial \
  --mode state_action \
  --traj_len 50 \
  --env_name Hopper-v3`

  
Note: better to choose demos whose rewards are distinctive enough.

### Reward Quality Eval



To test the learned reward, we use the reinforcement learning code modified from [here](https://github.com/ikostrikov/pytorch-trpo). We basically try to train an actor critic network with the learned reward, and compare the ground truth rewards.

To evaluate TREX model:

`
python pytorch-trpo/main_trex.py \
  --env-name CustomHalfCheetah-v0 \
  --test-env-name HalfCheetah-v3 \
  --batch-size 15000 \
  --save-interval 1 \
  --reward_model log/runs/HalfCheetah-v3/trex_0/model_ckpt/reward_net_best_8.pth \
  --mode state_action \
  --prefix 1 \
  --output_path pytorch-trpo/the_log_path`

To evaluate GRIRL model, change to `pytorch-trpo/main_grirl.py` accordingly.
You will then get a list of ckpts of policy networks in `pytorch-trpo/the_log_path/[METHOD]/CustomHalfCheetah/0`, change the path in `pytorch-trpo/eval_batch_ckpt.py` accordingly to get the result plots.

