import gym
from ddqn.my_ddqn import DQNAgent

from properties.properties import AgentPropertiesWrapper

import torch
# torch.manual_seed(100)

import matplotlib.pyplot as plt
import numpy as np

import random

from hypers import *


# env_id = "gym_mc:mc-v0"
# MAX_EPISODES = 300
# MAX_STEPS = 1000
# BATCH_SIZE = 64
# N = 500
# MEASUREMENT_INTERVAL = 10
# N_RUNS = 1
# LEARNING_RATE = 1e-3

# LOAD_FROM_FILE = False

# TSNE_COLOR = 'max_action_values' # or 'actions', 'velocity_sign', 'velocity', 'position_and_velocity', 'max_action_value'

# RANDOM_SEED = 0


LOOK_FOR_CONTINUALLY_INCREASING_REWARD = False
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)



env = gym.make(env_id)

env.seed(0)
# agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)
# milestones = [1, 20, 50, 100, 150, 190, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900]

milestones = np.arange(0, MAX_EPISODES-1, MEASUREMENT_INTERVAL)

n_milestones = len(milestones)


# averages
episode_rewards_avg = np.zeros((len(milestones),))
complexity_reductions_avg = np.zeros((len(milestones),))
awareness_list_avg = np.zeros((len(milestones),))
orthogonality_list_avg = np.zeros((len(milestones),))
sparsity_list_avg = np.zeros((len(milestones),))
diversity_list_avg = np.zeros((len(milestones),))




# collections
episode_rewards_collection = np.zeros((N_RUNS, len(milestones),))
complexity_reductions_collection = np.zeros((N_RUNS, len(milestones),))
awareness_list_collection = np.zeros((N_RUNS, len(milestones),))
orthogonality_list_collection = np.zeros((N_RUNS, len(milestones),))
sparsity_list_collection = np.zeros((N_RUNS, len(milestones),))
diversity_list_collection = np.zeros((N_RUNS, len(milestones),))








# if LOAD_FROM_FILE:
#     agent = DQNAgent(env, use_conv=False, gamma=1, tau=0.01, learning_rate=0.005)
#     agent.model.load_state_dict(torch.load('saved_models/model.pt'))
#     agent.target_model.load_state_dict(torch.load('saved_models/target_model.pt'))


for run in range(N_RUNS):
    
    # agent = DQNAgent(env, gamma=1, tau=0.01, learning_rate=0.0005)
    


    restart_flag = True
    while restart_flag:
        agent = DQNAgent(env, gamma=1, tau=0.01, learning_rate=LEARNING_RATE)
        agent_model_state_dicts, episode_rewards, eval_milestone_rewards, restart_flag = agent.mini_batch_train(MAX_EPISODES, MAX_STEPS, BATCH_SIZE, milestones=milestones, look_for_continually_increasing_reward=LOOK_FOR_CONTINUALLY_INCREASING_REWARD)

    # PLOTTING EPISODE REWARDS

    plt.plot(episode_rewards)

    plt.title('Episode Rewards')
    plt.show()

    agent_properties  = AgentPropertiesWrapper(env, agent.gamma, agent.tau, agent.learning_rate, N)

    all_properties = agent_properties.return_properties(agent_model_state_dicts, tsne_colors=TSNE_COLOR)



    # milestone_rewards = np.array(episode_rewards)[milestones]

    milestone_rewards = eval_milestone_rewards


    # building averages
    episode_rewards_avg += milestone_rewards

    # building collections
    episode_rewards_collection[run] = milestone_rewards
    complexity_reductions_collection[run] = all_properties['milestone_properties']['complexity_reduction']
    awareness_list_collection[run] = all_properties['milestone_properties']['awareness']
    orthogonality_list_collection[run] = all_properties['milestone_properties']['orthogonality']
    sparsity_list_collection[run] = all_properties['milestone_properties']['sparsity']
    diversity_list_collection[run] = all_properties['milestone_properties']['diversity']


torch.save(agent.model.state_dict(), 'saved_models/model.pt')
torch.save(agent.target_model.state_dict(), 'saved_models/target_model.pt')






# PLOTTING HEATMAPS OF LAST LAYER WEIGHTS
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
# fig, axs = plt.subplots(1, 1, figsize=(10, 3))
for i, ax in enumerate(axs):

    if i < n_milestones:
        ax.imshow(all_properties['milestone_heatmaps'][i])


    # plt.subplot(str(len(representation_list) // 2 + 1) + '2' + str(i))
    # plt.imshow(rep, cmap='gray', norm='linear')
plt.show()


# PLOTTING OUTPUT DECISION BOUNDARIES
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axs):

    # ax.imshow(representation_list[i])
    if i < n_milestones:
        # print('ZZZZ for ', i, ' ', np.flip(zz_list[i], axis=0))
        ax.contourf(all_properties['milestone_decision_boundaries']['state_x'][i], all_properties['milestone_decision_boundaries']['state_y'][i], all_properties['milestone_decision_boundaries']['class'][i])
        # ax.set_title('Output decision boundary for agent with reward: ' + str(milestone_rewards[i]))
        ax.set_title('Reward: ' + str(milestone_rewards[i]))


    # plt.subplot(str(len(representation_list) // 2 + 1) + '2' + str(i))
    # plt.imshow(rep, cmap='gray', norm='linear')
plt.show()


# PLOTTING HIDDEN LAYER TSNE PLOTS
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i, ax in enumerate(axs):
    # ax.imshow(representation_list[i])

    # i += n_milestones - 10

    
    if i < n_milestones:
        # print('ZZZZ for ', i, ' ', np.flip(zz_list[i], axis=0))

        if TSNE_COLOR == 'actions' or TSNE_COLOR == 'velocity_sign':

            colors = ['red', 'green', 'yellow'] if TSNE_COLOR == 'actions' else ['blue', 'red', 'green']

            # for actions: red = left, yellow = coast, green = right

            # for velocity_sign: red = negative, blue = zero, green = positive


            scatter = ax.scatter(all_properties['milestone_tsne_class_clusters']['tsne_x'][i], all_properties['milestone_tsne_class_clusters']['tsne_y'][i], c=[colors[0] if l == 0 else (colors[2] if l == 1 else colors[1]) for l in  all_properties['milestone_tsne_class_clusters']['class'][i]])

            
            classes = ['Left', 'Coast', 'Right'] if TSNE_COLOR == 'actions' else ['Negative', 'Zero', 'Positive']
            ax.set_title('Reward: ' + str(milestone_rewards[i]))
            ax.legend(handles=scatter.legend_elements()[0], labels=classes)
        elif TSNE_COLOR == 'max_action_values':

            if PROCESS_TSNE_POSITIVES:

                # process tnse plots to remove > 0 action values.
                tsne_x = all_properties['milestone_tsne_class_clusters']['tsne_x'][i]
                tsne_y = all_properties['milestone_tsne_class_clusters']['tsne_y'][i]
                tsne_z = all_properties['milestone_tsne_class_clusters']['class'][i]

                processed_tsne_x = tsne_x[tsne_z < 0]
                processed_tsne_y = tsne_y[tsne_z < 0]
                processed_tsne_z = tsne_z[tsne_z < 0]
            else:
                processed_tsne_x = all_properties['milestone_tsne_class_clusters']['tsne_x'][i]
                processed_tsne_y = all_properties['milestone_tsne_class_clusters']['tsne_y'][i]
                processed_tsne_z = all_properties['milestone_tsne_class_clusters']['class'][i]

            scatter = ax.scatter(processed_tsne_x, processed_tsne_y, c=processed_tsne_z, cmap='plasma')
            # ax.set_title('max action value for agent with Reward: ' + str(milestone_rewards[i]))
            ax.set_title('Reward: ' + str(milestone_rewards[i]))
            fig.colorbar(scatter)
        elif TSNE_COLOR == 'velocity' or TSNE_COLOR == 'position':
            scatter = ax.scatter(all_properties['milestone_tsne_class_clusters']['tsne_x'][i], all_properties['milestone_tsne_class_clusters']['tsne_y'][i], c=all_properties['milestone_tsne_class_clusters']['class'][i], cmap='plasma')
            # ax.set_title('Velocity for agent with Reward: ' + str(milestone_rewards[i]))
            ax.set_title('Reward: ' + str(milestone_rewards[i]))

            fig.colorbar(scatter)

        elif TSNE_COLOR == 'position_and_velocity':

            scatter = ax.scatter(all_properties['milestone_tsne_class_clusters']['tsne_x'][i], all_properties['milestone_tsne_class_clusters']['tsne_y'][i], c=all_properties['milestone_tsne_class_clusters']['class'][i], s=all_properties['milestone_tsne_class_clusters']['class2'][i], cmap='plasma')
            # ax.set_title('Velocity for agent with Reward: ' + str(milestone_rewards[i]))
            ax.set_title('Reward: ' + str(milestone_rewards[i]))

            fig.colorbar(scatter)



        

            # ax.legend(handles=scatter.legend_elements()[0], labels=classes)


        
        
plt.show()

exit()

# PLOTTING HIDDEN LAYER TSNE PLOTS, WITH SELECTED ACTION VALUES

selected_action_tsne_class_clusters = agent_properties.return_properties(agent_model_state_dicts, tsne_colors='actions')['milestone_tsne_class_clusters']
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i, ax in enumerate(axs):
    # ax.imshow(representation_list[i])
    if i < n_milestones:
        # print('ZZZZ for ', i, ' ', np.flip(zz_list[i], axis=0))

        
        scatter = ax.scatter(selected_action_tsne_class_clusters['tsne_x'][i], selected_action_tsne_class_clusters['tsne_y'][i], c=selected_action_tsne_class_clusters['class'][i])
        classes = ['Left', 'Coast', 'Right']
        # ax.set_title('Action the agent selects with Reward: ' + str(milestone_rewards[i]))
        ax.set_title('Reward: ' + str(milestone_rewards[i]))
        ax.legend(handles=scatter.legend_elements()[0], labels=classes)

        
        
plt.show()


# PLOT TSNE FOR BEST MODEL

# use best model
agent.model.load_state_dict(torch.load('saved_models/best_model.pt'))
agent.target_model.load_state_dict(torch.load('saved_models/best_target_model.pt'))

print('BEST EPISODE REWARD: ', max(episode_rewards))


fig, axs = plt.subplots(1, 1, figsize=(10, 10))


best_agent_properties = agent_properties.return_properties([agent.model.state_dict()], tsne_colors=TSNE_COLOR)
i = 0
# process tnse plots to remove > 0 action values.

tsne_x = best_agent_properties['milestone_tsne_class_clusters']['tsne_x'][i]
tsne_y = best_agent_properties['milestone_tsne_class_clusters']['tsne_y'][i]
tsne_z = best_agent_properties['milestone_tsne_class_clusters']['class'][i]

processed_tsne_x = tsne_x[tsne_z < 0]
processed_tsne_y = tsne_y[tsne_z < 0]
processed_tsne_z = tsne_z[tsne_z < 0]

scatter = axs.scatter(processed_tsne_x, processed_tsne_y, c=processed_tsne_z, cmap='plasma')

# print('HEAT VALUESS ', best_agent_properties['milestone_tsne_class_clusters']['class'][i])


axs.set_title('Best Reward: ' + str(max(episode_rewards)))
fig.colorbar(scatter)

plt.show()





# vr = VisualizeRepresentation(agent, env)

# vr.visualize_env(layer='output')

# vr.return_tsne_visualization()



episode_rewards_avg = episode_rewards_avg / N_RUNS
complexity_reductions_avg = complexity_reductions_avg / N_RUNS
awareness_list_avg = awareness_list_avg / N_RUNS
orthogonality_list_avg = orthogonality_list_avg / N_RUNS
sparsity_list_avg = sparsity_list_avg / N_RUNS
diversity_list_avg = diversity_list_avg / N_RUNS

# PLOTTING MEASURED PROPERTIES
fig, axs = plt.subplots(1, 5, figsize=(10, 3))

for i, ax in enumerate(axs):
    if i == 0:
        for j in range(N_RUNS):
            ax.plot(milestones, episode_rewards_collection[j], label="Rewards")

            ax.set_ylabel('Rewards')
            ax.set_xlabel('Episode')
    elif i == 1:
        for j in range(N_RUNS):
            ax.plot(milestones, complexity_reductions_collection[j], label="Complexity reduction")
            ax.set_ylabel('Complexity Reduction')
            ax.set_xlabel('Episode')
    elif i == 2:
        for j in range(N_RUNS):
            ax.plot(milestones, awareness_list_collection[j], label="Awareness")
            ax.set_ylabel('Awareness')
            ax.set_xlabel('Episode')
    elif i == 3:
        for j in range(N_RUNS):
            ax.plot(milestones, orthogonality_list_collection[j], label="Orthogonality")
            ax.set_ylabel('Orthogonality')
            ax.set_xlabel('Episode')

    elif i == 4:
        for j in range(N_RUNS):
            ax.plot(milestones, sparsity_list_collection[j], label="Sparsity")
            ax.set_ylabel('Sparsity')
            ax.set_xlabel('Episode')
    elif i == 5:
        for j in range(N_RUNS):
            ax.plot(milestones, diversity_list_collection[j], label="Diversity")
            ax.set_xlabel('Episode')
            ax.set_ylabel('Diversity')


# fig = plt.figure(figsize=(6, 4))
# plt.subplot(321)
# # plt.plot(milestones, episode_rewards_avg, label="Rewards")

# for i in range(N_RUNS):
#     plt.plot(milestones, episode_rewards_collection[i], label="Rewards")

# plt.ylabel('Rewards')
# plt.xlabel('Episode')
# plt.subplot(322)
# # plt.plot(milestones, complexity_reductions_avg, label="Complexity reduction")

# for i in range(N_RUNS):
#     plt.plot(milestones, complexity_reductions_collection[i], label="Complexity reduction")

# plt.ylabel('Complexity Reduction')
# plt.xlabel('Episode')
# plt.subplot(323)

# for i in range(N_RUNS):
#     plt.plot(milestones, awareness_list_collection[i], label="Awareness")
# # plt.plot(milestones, awareness_list_avg, label="Awareness")
# plt.ylabel('Awareness')
# plt.xlabel('Episode')
# plt.subplot(324)

# for i in range(N_RUNS):
#     plt.plot(milestones, orthogonality_list_collection[i], label="Orthogonality")


# # plt.plot(milestones, orthogonality_list_avg, label="Orthogonality")
# plt.ylabel('Orthogonality')
# plt.xlabel('Episode')
# plt.subplot(325)
# # plt.plot(milestones, sparsity_list_avg, label="Sparsity")

# for i in range(N_RUNS):
#     plt.plot(milestones, sparsity_list_collection[i], label="Sparsity")

# plt.ylabel('Sparsity')
# plt.xlabel('Episode')
# plt.subplot(326)

# for i in range(N_RUNS):
#     plt.plot(milestones, diversity_list_collection[i], label="Diversity")


# plt.xlabel('Episode')
# # plt.plot(milestones, diversity_list_avg, label="Diversity")
# plt.ylabel('Diversity')



# print('--AVERAGES--')
# print('episode rewards: ', episode_rewards_avg)
# print('complexity_reductions_avg: ', complexity_reductions_avg)
# print('awareness_list_avg: ', awareness_list_avg)
# print('orthogonality_list_avg: ', orthogonality_list_avg)
# print('sparsity_list_avg: ', sparsity_list_avg)
# print('diversity_list_avg: ', diversity_list_avg)



# print('--COLLECTIONS--')
# print('episode rewards: ', episode_rewards_collection)
# print('complexity_reductions_avg: ', complexity_reductions_collection)
# print('awareness_list_avg: ', awareness_list_collection)
# print('orthogonality_list_avg: ', orthogonality_list_collection)
# print('sparsity_list_avg: ', sparsity_list_collection)
# print('diversity_list_avg: ', diversity_list_collection)

# plt.legend()

plt.show()

    