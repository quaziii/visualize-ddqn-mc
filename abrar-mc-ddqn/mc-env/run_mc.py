import gym

env = gym.make('gym_mc:mc-v0')
env.reset()


from ddqn.agent.ddqn_agent import Agent

# DDQN stuff here: Abrar


# DDQN stuff end: Abrar

for _ in range(3000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action, when we use DDQN, we have to pass the action defined by DDQN here
env.close()