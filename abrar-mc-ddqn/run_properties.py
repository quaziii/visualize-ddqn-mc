import gym
from ddqn.my_ddqn import DQNAgent

from properties.properties import AgentPropertiesWrapper

import torch
# torch.manual_seed(100)

import matplotlib.pyplot as plt
import numpy as np

import random

from hypers import *

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)


env = gym.make(env_id)

env.seed(0)

milestones = np.arange(0, MAX_EPISODES-1, MEASUREMENT_INTERVAL)

n_milestones = len(milestones)

for episode in EPISODES_TO_ADD_TO_MILESTONES:
    if episode not in milestones:
        milestones = np.append(milestones, episode)


# sort milestones

milestones = np.sort(milestones)
# collections
episode_rewards_collection = np.zeros((N_RUNS, len(milestones),))
complexity_reductions_collection = np.zeros((N_RUNS, len(milestones),))
awareness_list_collection = np.zeros((N_RUNS, len(milestones),))
orthogonality_list_collection = np.zeros((N_RUNS, len(milestones),))
sparsity_list_collection = np.zeros((N_RUNS, len(milestones),))
diversity_list_collection = np.zeros((N_RUNS, len(milestones),))

for run in range(N_RUNS):

    agent = DQNAgent(env, gamma=1, tau=0.01, learning_rate=LEARNING_RATE)
    agent_model_state_dicts, episode_rewards, eval_milestone_rewards, _ = agent.mini_batch_train(MAX_EPISODES, MAX_STEPS, BATCH_SIZE, milestones=milestones, look_for_continually_increasing_reward=LOOK_FOR_CONTINUALLY_INCREASING_REWARD)

    agent_properties  = AgentPropertiesWrapper(env, agent.gamma, agent.tau, agent.learning_rate, N)

    all_properties = agent_properties.return_properties(agent_model_state_dicts, tsne_colors=TSNE_COLOR, skip_tsne=True, skip_properties=False)

    milestone_rewards = eval_milestone_rewards

    # building collections
    episode_rewards_collection[run] = milestone_rewards
    complexity_reductions_collection[run] = all_properties['milestone_properties']['complexity_reduction']
    awareness_list_collection[run] = all_properties['milestone_properties']['awareness']
    orthogonality_list_collection[run] = all_properties['milestone_properties']['orthogonality']
    sparsity_list_collection[run] = all_properties['milestone_properties']['sparsity']
    diversity_list_collection[run] = all_properties['milestone_properties']['diversity']


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


plt.show()

