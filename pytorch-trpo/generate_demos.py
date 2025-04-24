import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models_ac import *
from replay_memory import Memory
from running_state import ZFilter
import os
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Generate Demos with Pretrained TRPO')

parser.add_argument('--env-name', default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--model-ckpts', nargs='+')
args = parser.parse_args()

os.makedirs(f'demo/{args.env_name}', exist_ok=True)

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

TRAJ_PER_FILE = 500

for ckpt in args.model_ckpts:
    print(f"Loading checkpoint {ckpt}")
    checkpoint = torch.load(ckpt)
    policy_net.load_state_dict(checkpoint['policy'])
    value_net.load_state_dict(checkpoint['value'])
    all_trajs = [] 
    all_rewards = []
    for i in range(TRAJ_PER_FILE):
        num_steps = 0
        state = env.reset()
        state = running_state(state)

        traj = []
        reward_seq = []
        for t in range(100): # Don't infinite loop while learning
            action = select_action(state)
            action = action.data[0].numpy()

            next_state, reward, done, _ = env.step(action)
            reward_seq.append(reward) 

            traj.append(np.concatenate([state.copy(), action.copy()], axis=0))

            next_state = running_state(next_state)

            if done:
                break

            if args.render:
                env.render()
            state = next_state

        all_trajs.append(np.array(traj))
        all_rewards.append(np.array(reward_seq))

    demo = {
        'traj':   all_trajs,
        'reward': all_rewards
    }
    # demo_fname = (
    #     f"{args.env_name}_"
    #     f"noise_{args.noise:.1f}_"
    #     f"interval_{args.log_interval}_"
    #     f"batch_{file_count}.pt"
    # )
    demo_fname = ckpt.split('/')[-1].replace('epoch', args.env_name)
    out_path = os.path.join('demo', args.env_name, demo_fname)
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(demo, f)
    print(f"Saved demo file : {demo_fname}")
