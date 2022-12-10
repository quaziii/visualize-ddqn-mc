import gym
from ddqn.my_ddqn import DQNAgent

from properties.properties import MeasureProperties, VisualizeRepresentation

import torch
# torch.manual_seed(100)

import matplotlib.pyplot as plt
import numpy as np



env_id = "gym_mc:mc-v0"
MAX_EPISODES = 200
MAX_STEPS = 1000
BATCH_SIZE = 64
N = 500
MEASUREMENT_INTERVAL = 500
REPRESENTATION_MEASUREMENT_INTERVAL = 30
N_RUNS = 1

LOAD_FROM_FILE = False




def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, calculate_properties=False, milestones = [], representation_milestones = []):
    episode_rewards = []
    properties = MeasureProperties(N, agent.model.representation_size, env.observation_space.shape[0], env.action_space.n, env)

    L_rep_list = []
    awareness_list = []
    orthogonality_list = []
    sparsity_list = []
    diversity_list = []
    rewards_list = []



    

    v_shape = VisualizeRepresentation(agent, env).return_visualization(layer='output')[2].shape

    representation_list = np.empty((len(representation_milestones), agent.model.representation_layer[2].weight.shape[0], agent.model.representation_layer[2].weight.shape[1]))
    xx_list = np.empty((len(representation_milestones), v_shape[0], v_shape[1]))
    yy_list = np.empty((len(representation_milestones), v_shape[0], v_shape[1]))
    zz_list = np.empty((len(representation_milestones), v_shape[0], v_shape[1]))



    i = 0
    max_episode_reward = -9999999
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

        if episode_reward > max_episode_reward:
            torch.save(agent.model.state_dict(), 'saved_models/best_model.pt')
            torch.save(agent.target_model.state_dict(), 'saved_models/best_target_model.pt')
            max_episode_reward = episode_reward
        
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
        

        if episode in representation_milestones:
            print('--COMPUTING REPRESENTATIONS-- FOR EPISODE ', episode)
            representation_list[i] = agent.model.representation_layer[2].weight.detach().numpy()

            xx_list[i], yy_list[i], zz_list[i] = VisualizeRepresentation(agent, env).return_visualization(layer='output')
            # yy_list[i] = VisualizeRepresentation(agent, env).visualize_env(layer='output')
            # zz_list[i] = VisualizeRepresentation(agent, env).visualize_env(layer='output')

           
            # print('WEIGHTS ', agent.model.representation_layer[2].weight.detach().numpy())
            # print('LISTT ', representation_list)

            i += 1

    # reducing learning rate every episode
    # agent.scheduler.step()

    
    complexity_reductions = MeasureProperties.complexity_reduction(L_rep_list)


    if calculate_properties:
        return np.array(rewards_list), np.array(complexity_reductions), np.array(awareness_list), np.array(orthogonality_list), np.array(sparsity_list), np.array(diversity_list), np.array(representation_list), np.array(xx_list), np.array(yy_list), np.array(zz_list), max_episode_reward
    else:

        return episode_rewards



env = gym.make(env_id)
# agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)
# milestones = [1, 20, 50, 100, 150, 190, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900]

milestones = np.arange(0, MAX_EPISODES-1, MEASUREMENT_INTERVAL)
representation_milestones = np.arange(0, MAX_EPISODES-1, REPRESENTATION_MEASUREMENT_INTERVAL)


# averages
episode_rewards_avg = np.zeros((len(milestones),))
complexity_reductions_avg = np.zeros((len(milestones),))
awareness_list_avg = np.zeros((len(milestones),))
orthogonality_list_avg = np.zeros((len(milestones),))
sparsity_list_avg = np.zeros((len(milestones),))
diversity_list_avg = np.zeros((len(milestones),))


# representation_list = np.zeros((len(representation_milestones),))



# collections
episode_rewards_collection = np.zeros((N_RUNS, len(milestones),))
complexity_reductions_collection = np.zeros((N_RUNS, len(milestones),))
awareness_list_collection = np.zeros((N_RUNS, len(milestones),))
orthogonality_list_collection = np.zeros((N_RUNS, len(milestones),))
sparsity_list_collection = np.zeros((N_RUNS, len(milestones),))
diversity_list_collection = np.zeros((N_RUNS, len(milestones),))








if LOAD_FROM_FILE:
    agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)
    agent.model.load_state_dict(torch.load('saved_models/model.pt'))
    agent.target_model.load_state_dict(torch.load('saved_models/target_model.pt'))
else:


    

    for run in range(N_RUNS):
        
        agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)

        episode_rewards, complexity_reductions, awareness_list, orthogonality_list, sparsity_list, diversity_list, representation_list, xx_list, yy_list, zz_list, max_episode_reward = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, calculate_properties=True, milestones=milestones, representation_milestones=representation_milestones)

        # print('episode_rewards', episode_rewards)
        # print('complexity_reductions', complexity_reductions)
        # print('awareness_list', awareness_list)
        # print('orthogonality_list', orthogonality_list)
        # print('sparsity_list', sparsity_list)
        # print('diversity_list', diversity_list)

        # maintaining running sums

        print('avg shape: ', episode_rewards_avg.shape)
        print('shape: ', episode_rewards.shape)

        # building averages
        episode_rewards_avg += episode_rewards
        complexity_reductions_avg += complexity_reductions
        awareness_list_avg += awareness_list
        orthogonality_list_avg += orthogonality_list
        sparsity_list_avg += sparsity_list
        diversity_list_avg += diversity_list

        # building collections
        episode_rewards_collection[run] = episode_rewards
        complexity_reductions_collection[run] = complexity_reductions
        awareness_list_collection[run] = awareness_list
        orthogonality_list_collection[run] = orthogonality_list
        sparsity_list_collection[run] = sparsity_list
        diversity_list_collection[run] = diversity_list

        # print('adding')
        # print('episode rewards: ', episode_rewards_avg)
        # print('complexity_reductions_avg: ', complexity_reductions_avg)
        # print('awareness_list_avg: ', awareness_list_avg)
        # print('orthogonality_list_avg: ', orthogonality_list_avg)
        # print('sparsity_list_avg: ', sparsity_list_avg)
        # print('diversity_list_avg: ', diversity_list_avg)


    torch.save(agent.model.state_dict(), 'saved_models/model.pt')
    torch.save(agent.target_model.state_dict(), 'saved_models/target_model.pt')


    # use best model
    agent.model.load_state_dict(torch.load('saved_models/best_model.pt'))
    agent.model.load_state_dict(torch.load('saved_models/best_target_model.pt'))

    print('BEST EPISODE REWARD: ', max_episode_reward)

    print('VISUALIZING HIDDEN LAYER')

    # print(agent.model.representation_layer[2].weight)
    # mapped_representation_layer = 
    # rep_weights = torch.where(agent.model.representation_layer[2].weight < 0.1, 0, )

    # plt.figure(figsize=(6,4))

    # print('REPRESENTATION LIST', representation_list)

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axs):
        print('i ', i)
        ax.imshow(representation_list[i])


        # plt.subplot(str(len(representation_list) // 2 + 1) + '2' + str(i))
        # plt.imshow(rep, cmap='gray', norm='linear')
    plt.show()

    

    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    for i, ax in enumerate(axs):
        print('i ', i)
        # ax.imshow(representation_list[i])
        print('ZZZZ for ', i, ' ', np.flip(zz_list[i], axis=0))
        ax.contourf(xx_list[i], yy_list[i], zz_list[i])


        # plt.subplot(str(len(representation_list) // 2 + 1) + '2' + str(i))
        # plt.imshow(rep, cmap='gray', norm='linear')
    plt.show()

    vr = VisualizeRepresentation(agent, env)

    vr.visualize_env(layer='output')



    episode_rewards_avg = episode_rewards_avg / N_RUNS
    complexity_reductions_avg = complexity_reductions_avg / N_RUNS
    awareness_list_avg = awareness_list_avg / N_RUNS
    orthogonality_list_avg = orthogonality_list_avg / N_RUNS
    sparsity_list_avg = sparsity_list_avg / N_RUNS
    diversity_list_avg = diversity_list_avg / N_RUNS
    fig = plt.figure(figsize=(6, 4))
    plt.subplot(321)
    # plt.plot(milestones, episode_rewards_avg, label="Rewards")

    for i in range(N_RUNS):
        plt.plot(milestones, episode_rewards_collection[i], label="Rewards")
    
    plt.ylabel('Rewards')
    plt.xlabel('Episode')
    plt.subplot(322)
    # plt.plot(milestones, complexity_reductions_avg, label="Complexity reduction")

    for i in range(N_RUNS):
        plt.plot(milestones, complexity_reductions_collection[i], label="Complexity reduction")

    plt.ylabel('Complexity Reduction')
    plt.xlabel('Episode')
    plt.subplot(323)

    for i in range(N_RUNS):
        plt.plot(milestones, awareness_list_collection[i], label="Awareness")
    # plt.plot(milestones, awareness_list_avg, label="Awareness")
    plt.ylabel('Awareness')
    plt.xlabel('Episode')
    plt.subplot(324)

    for i in range(N_RUNS):
        plt.plot(milestones, orthogonality_list_collection[i], label="Orthogonality")

    
    # plt.plot(milestones, orthogonality_list_avg, label="Orthogonality")
    plt.ylabel('Orthogonality')
    plt.xlabel('Episode')
    plt.subplot(325)
    # plt.plot(milestones, sparsity_list_avg, label="Sparsity")

    for i in range(N_RUNS):
        plt.plot(milestones, sparsity_list_collection[i], label="Sparsity")

    plt.ylabel('Sparsity')
    plt.xlabel('Episode')
    plt.subplot(326)

    for i in range(N_RUNS):
        plt.plot(milestones, diversity_list_collection[i], label="Diversity")
    
    
    plt.xlabel('Episode')
    # plt.plot(milestones, diversity_list_avg, label="Diversity")
    plt.ylabel('Diversity')
    


    print('--AVERAGES--')
    print('episode rewards: ', episode_rewards_avg)
    print('complexity_reductions_avg: ', complexity_reductions_avg)
    print('awareness_list_avg: ', awareness_list_avg)
    print('orthogonality_list_avg: ', orthogonality_list_avg)
    print('sparsity_list_avg: ', sparsity_list_avg)
    print('diversity_list_avg: ', diversity_list_avg)



    print('--COLLECTIONS--')
    print('episode rewards: ', episode_rewards_collection)
    print('complexity_reductions_avg: ', complexity_reductions_collection)
    print('awareness_list_avg: ', awareness_list_collection)
    print('orthogonality_list_avg: ', orthogonality_list_collection)
    print('sparsity_list_avg: ', sparsity_list_collection)
    print('diversity_list_avg: ', diversity_list_collection)

    # plt.legend()

    plt.show()

    


# print('tupels ', env.observation_space.shape)

# properties = MeasureProperties(N, agent.model.representation_size, env.observation_space.shape[0], env.action_space.n)



# properties.initialize_data(agent)

# print('complexity reduction: ', properties.get_L_rep())

# print('awareness: ', properties.awareness())
# print('orthogonality: ', properties.orthogonality())
# print('sparsity: ', properties.sparsity())
# print('diversity: ', properties.diversity()) # gives runtime warning debug LATER



