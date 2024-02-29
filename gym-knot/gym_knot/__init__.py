from gym.envs.registration import register

register(
    id='Slice-v0', 
    entry_point='gym_knot.envs:SliceEnv',
    max_episode_steps=1000,
    reward_threshold=0,)

