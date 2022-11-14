import gym
from ddqn.my_ddqn import DQNAgent, mini_batch_train

from properties.properties import MeasureProperties

import torch



env_id = "gym_mc:mc-v0"
MAX_EPISODES = 200
MAX_STEPS = 1000
BATCH_SIZE = 32
N = 100

LOAD_FROM_FILE = True

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)

if LOAD_FROM_FILE:
    agent.model.load_state_dict(torch.load('saved_models/model.pt'))
    agent.target_model.load_state_dict(torch.load('saved_models/target_model.pt'))
else:
    
    episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

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



