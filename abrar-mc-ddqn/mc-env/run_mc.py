import gym

env = gym.make('gym_mc:mc-v0')
env.reset()


# from ddqn.agent.ddqn_agent import Agent
USE_OPTIMAL_POLICY = True
USE_RANDOM_POLICY = False


BOTTOM = -0.5251529683 # found by finding position where velocity of car is maximum, when only action is no_accelerate. WILL CHANGE WITH MC PARAMETERS.





next_action = 2

total_reward = 0
for _ in range(3000):
    env.render()

    if USE_RANDOM_POLICY:

        obs, reward, done, info = env.step(env.action_space.sample()) # take a random action, when we use DDQN, we have to pass the action defined by DDQN here
    elif USE_OPTIMAL_POLICY:

        obs, reward, done, info = env.step(next_action) # take a random action, when we use DDQN, we have to pass the action defined by DDQN here

        if obs[0] > BOTTOM and obs[1] > 0:
            # moving uphill to the right
            next_action = 2
        elif obs[0] < BOTTOM and obs[1] < 0:
            # moving uphill to the left

            next_action = 0
        else:
            # coast
            next_action = 1

        total_reward += reward        

        if obs[0] > 0.5:
            print('total reward: ', total_reward)
            break

    # obs, reward, done, info = env.step(1) # take a random action, when we use DDQN, we have to pass the action defined by DDQN here




env.close()