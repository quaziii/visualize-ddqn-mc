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
BATCH_SIZE = 64
N = 100

LOAD_FROM_FILE = False

N_RUNS = 1




def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, calculate_properties=False, milestones = []):
    episode_rewards = []
    properties = MeasureProperties(N, agent.model.representation_size, env.observation_space.shape[0], env.action_space.n)

    L_rep_list = []
    awareness_list = []
    orthogonality_list = []
    sparsity_list = []
    diversity_list = []
    rewards_list = []

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
            properties.initialize_data(agent)
            # print('complexity reduction: ', properties.get_L_rep())
            L_rep_list.append(properties.get_L_rep())
            awareness_list.append(properties.awareness())
            orthogonality_list.append(properties.orthogonality())
            sparsity_list.append(properties.sparsity())
            diversity_list.append(properties.diversity())
            rewards_list.append(episode_reward)

    
    complexity_reductions = MeasureProperties.complexity_reduction(L_rep_list)

    
    



    if calculate_properties:
        return np.array(rewards_list), np.array(complexity_reductions), np.array(awareness_list), np.array(orthogonality_list), np.array(sparsity_list), np.array(diversity_list)
    else:

        return episode_rewards



env = gym.make(env_id)
# agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)
milestones = [1, 20, 50, 100, 150, 190]

episode_rewards_avg = np.zeros((len(milestones),))
complexity_reductions_avg = np.zeros((len(milestones),))
awareness_list_avg = np.zeros((len(milestones),))
orthogonality_list_avg = np.zeros((len(milestones),))
sparsity_list_avg = np.zeros((len(milestones),))
diversity_list_avg = np.zeros((len(milestones),))


if LOAD_FROM_FILE:
    agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)
    agent.model.load_state_dict(torch.load('saved_models/model.pt'))
    agent.target_model.load_state_dict(torch.load('saved_models/target_model.pt'))
else:
    

    for _ in range(N_RUNS):
        
        agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)




    
    
        episode_rewards, complexity_reductions, awareness_list, orthogonality_list, sparsity_list, diversity_list = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, calculate_properties=True, milestones=milestones)

        print('episode_rewards', episode_rewards)
        print('complexity_reductions', complexity_reductions)
        print('awareness_list', awareness_list)
        print('orthogonality_list', orthogonality_list)
        print('sparsity_list', sparsity_list)
        print('diversity_list', diversity_list)

        # maintaining running sums

        print('avg shape: ', episode_rewards_avg.shape)
        print('shape: ', episode_rewards.shape)
        episode_rewards_avg += episode_rewards
        complexity_reductions_avg += complexity_reductions
        awareness_list_avg += awareness_list
        orthogonality_list_avg += orthogonality_list
        sparsity_list_avg += sparsity_list
        diversity_list_avg += diversity_list

        print('adding')
        print('episode rewards: ', episode_rewards_avg)
        print('complexity_reductions_avg: ', complexity_reductions_avg)
        print('awareness_list_avg: ', awareness_list_avg)
        print('orthogonality_list_avg: ', orthogonality_list_avg)
        print('sparsity_list_avg: ', sparsity_list_avg)
        print('diversity_list_avg: ', diversity_list_avg)

        del agent

    torch.save(agent.model.state_dict(), 'saved_models/model.pt')
    torch.save(agent.target_model.state_dict(), 'saved_models/target_model.pt')

    episode_rewards_avg = episode_rewards_avg / N_RUNS
    complexity_reductions_avg = complexity_reductions_avg / N_RUNS
    awareness_list_avg = awareness_list_avg / N_RUNS
    orthogonality_list_avg = orthogonality_list_avg / N_RUNS
    sparsity_list_avg = sparsity_list_avg / N_RUNS
    diversity_list_avg = diversity_list_avg / N_RUNS


    plt.plot(milestones, episode_rewards_avg, label="Rewards")
    plt.plot(milestones, complexity_reductions_avg, label="Complexity reduction")
    plt.plot(milestones, awareness_list_avg, label="Awareness")
    plt.plot(milestones, orthogonality_list_avg, label="Orthogonality")
    plt.plot(milestones, sparsity_list_avg, label="Sparsity")
    plt.plot(milestones, diversity_list_avg, label="Diversity")

    print('episode rewards: ', episode_rewards_avg)
    print('complexity_reductions_avg: ', complexity_reductions_avg)
    print('awareness_list_avg: ', awareness_list_avg)
    print('orthogonality_list_avg: ', orthogonality_list_avg)
    print('sparsity_list_avg: ', sparsity_list_avg)
    print('diversity_list_avg: ', diversity_list_avg)


    plt.legend()

    plt.show()

    


# print('tupels ', env.observation_space.shape)

properties = MeasureProperties(N, agent.model.representation_size, env.observation_space.shape[0], env.action_space.n)



properties.initialize_data(agent)

print('complexity reduction: ', properties.get_L_rep())

print('awareness: ', properties.awareness())
print('orthogonality: ', properties.orthogonality())
print('sparsity: ', properties.sparsity())
print('diversity: ', properties.diversity()) # gives runtime warning debug LATER



