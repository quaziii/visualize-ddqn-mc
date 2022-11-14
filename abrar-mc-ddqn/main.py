import gym
from ddqn.my_ddqn import DQNAgent

from properties.properties import MeasureProperties

import torch
torch.manual_seed(100)

import matplotlib.pyplot as plt
import numpy as np



env_id = "gym_mc:mc-v0"
MAX_EPISODES = 200
MAX_STEPS = 1000
BATCH_SIZE = 32
N = 100

LOAD_FROM_FILE = False

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, calculate_properties=False, milestones = []):
    episode_rewards = []
    properties = MeasureProperties(N, agent.model.representation_size, env.observation_space.shape[0], env.action_space.n)

    L_reps = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)  

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state
        if episode in milestones and calculate_properties:
            print('--COMPUTING PROPERTIES-- FOR EPISODE ', episode)
            properties.initialize_data(agent.model)
            # print('complexity reduction: ', properties.get_L_rep())
            L_reps.append(properties.get_L_rep())

    
    complexity_reductions = MeasureProperties.complexity_reduction(L_reps)


    if calculate_properties:
        return episode_rewards, complexity_reductions
    else:

        return episode_rewards






if LOAD_FROM_FILE:
    agent.model.load_state_dict(torch.load('saved_models/model.pt'))
    agent.target_model.load_state_dict(torch.load('saved_models/target_model.pt'))
else:
    
    episode_rewards, L_reps = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, calculate_properties=True, milestones=[1, 20, 50, 100, 150, 190])

    plt.plot(L_reps)

    plt.show()

    torch.save(agent.model.state_dict(), 'saved_models/model.pt')
    torch.save(agent.target_model.state_dict(), 'saved_models/target_model.pt')


# print('tupels ', env.observation_space.shape)

properties = MeasureProperties(N, agent.model.representation_size, env.observation_space.shape[0], env.action_space.n)



properties.initialize_data(agent.model)

print('complexity reduction: ', properties.get_L_rep())

print('awareness: ', properties.awareness())
print('orthogonality: ', properties.orthogonality())
print('sparsity: ', properties.sparsity())
print('diversity: ', properties.diversity())



