from gym.envs.registration import register

register(
    id='mc-v0',
    entry_point='gym_mc.envs:McEnv',
)