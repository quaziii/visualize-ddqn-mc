import torch
import torch.nn.functional as F

import copy

# torch.manual_seed(100)



import numpy as np
import random
from collections import deque

from ddqn.model import DQN

# taken from https://github.com/cyoon1729/deep-Q-networks/blob/master/doubleDQN/ddqn.py


class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
      experience = (state, action, np.array([reward]), next_state, done)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, action, reward, next_state, done = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)

      return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)



class DQNAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.device = "cpu"


        self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [200, 400, 600, 800] , gamma=0.1, verbose=True)
        
        
    def normalize_state(self, state):
        return state

        # print('state shape ', state)
        normalized_state = state
        normalized_state = (normalized_state - np.array([self.env.min_position, 0])) / np.array([self.env.max_position - self.env.min_position, self.env.max_speed])
        # normalized_state[0] = state[0] - self.env.min_position / (self.env.max_position - self.env.min_position)
        # normalized_state[1] = state[1] / (self.env.max_speed)

        normalized_state = normalized_state.to(state.dtype)

        normalized_state = state



        # print('norm shape ', normalized_state)
        return normalized_state
    def get_action(self, state, eps=-1.5):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)



        # normalize state features
        # state shape: (1,2)
        # state: (position, velocity)

        # print('state shape ', state.shape)
        normalized_state = self.normalize_state(state)

        # print('normsa shape ', normalized_state.shape)

        qvals = self.model.forward(normalized_state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() < eps):
            # print('take rand')
            return self.env.action_space.sample()

        return action

    def compute_loss(self, batch):   
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1).to(self.device)


        # states shape: (n_buffer, 2)
        # compute loss
        normalized_states = self.normalize_state(states)
        normalized_next_states = self.normalize_state(next_states)
        curr_Q = self.model.forward(normalized_states).gather(1, actions)
        next_Q = self.target_model.forward(normalized_next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        # for p in self.model.parameters():
        #     print(p.grad.norm())
        self.optimizer.step()

        # self.scheduler.step()
        
        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


    def mini_batch_train(self, env, agent, max_episodes, max_steps, batch_size, milestones = []):

        agent_model_state_dicts = []

        episode_rewards = []

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
            
          
            

            if episode in milestones:
                agent_model_state_dicts.append(copy.deepcopy(agent.model.state_dict()))

        # reducing learning rate every episode
        # agent.scheduler.step()

        return agent_model_state_dicts, episode_rewards