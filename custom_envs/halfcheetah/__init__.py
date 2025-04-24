from gym.envs.registration import register

register(
    id='CustomHalfCheetah-v0',
    entry_point='halfcheetah.halfcheetah_v3:CustomHalfCheetahEnv',
    max_episode_steps=1000,
)
