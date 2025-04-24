import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import HalfCheetahEnv  # or AntEnv, HalfCheetahEnv, etc.
from gym.envs.mujoco import mujoco_env

class CustomHalfCheetahEnv(HalfCheetahEnv):
    def __init__(
        self,
        reward_mode='state_only',
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        reward_net=None
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.reward_net = reward_net
        self.reward_mode = reward_mode
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
    def set_reward_net(self, reward_net):
        self.reward_net = reward_net
    def step(self, action):
        observation_before = self._get_obs()
        observation, reward, done, info = super().step(action)
        
        if self.reward_net is not None:
            if self.reward_mode == 'state_only':
                net_reward = self.reward_net.compute_reward(observation)
            elif self.reward_mode == 'state_pair':
                net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, observation], axis=0))
            elif self.reward_mode == 'state_action':
                net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, action], axis=0))
            elif self.reward_mode == 'state_action_state':
                net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, action, observation], axis=0))
            reward = net_reward# - ctrl_cost
        else:
            reward = 0
        
        return observation, reward, done, info
