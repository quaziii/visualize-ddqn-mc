import torch
import torch.nn.functional as F

import copy
import matplotlib.pyplot as plt

# torch.manual_seed(100)



import numpy as np
import random
from collections import deque

from ddqn.model import DQN

from hypers import *



# taken from https://github.com/cyoon1729/deep-Q-networks/blob/master/doubleDQN/ddqn.py




class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

      np.random.seed(RANDOM_SEED)
      torch.manual_seed(RANDOM_SEED)
      random.seed(RANDOM_SEED)

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

    def __init__(self, env, learning_rate=3e-4, gamma=0.99, tau=0.01, buffer_size=10000, epsilon=1, epsilon_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
	
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

    def get_eval_action(self, state):
        '''
        Greedy policy to follow during evaluation
        '''

        # trying something! Maybe remove later

        return self.get_train_action(state)


        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        # normalize state features
        # state shape: (1,2)
        # state: (position, velocity)

        # print('state shape ', state.shape)
        normalized_state = self.normalize_state(state)

        # print('normsa shape ', normalized_state.shape)

        qvals = self.model.forward(normalized_state)
        action = np.argmax(qvals.cpu().detach().numpy())
        return action


    
    def get_train_action(self, state):
        '''
        More exploratory policy to follow during training
        '''

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)

        # normalize state features
        # state shape: (1,2)
        # state: (position, velocity)

        # print('state shape ', state.shape)
        normalized_state = self.normalize_state(state)

        # print('normsa shape ', normalized_state.shape)

        qvals = self.model.forward(normalized_state)
        # action = np.argmax(qvals.cpu().detach().numpy())
        vals = qvals.cpu().detach().numpy()

        # print('vals ', vals)
        
        action = np.random.choice(np.flatnonzero(np.isclose(vals, vals.max())))

        if(np.random.rand() < self.epsilon):
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

        # print('curr_Q ', curr_Q)


        next_Q = self.target_model.forward(normalized_next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q


        # if torch.any(curr_Q > 0):
            # print('curr_Q ', curr_Q)
            # print('max_next_Q ', max_next_Q)
        # print('expected_Q ', expected_Q)
        
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

        return loss.detach().cpu().item()

    def evaluate(self, max_steps):
        print('--- EVALUATING MODEL ---')
        state = self.env.eval_reset()
        episode_reward = 0
        done = False
        step = 0
        while not done and step < max_steps:
            step += 1
            action = self.get_eval_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            state = next_state
        print("Evaluation reward: ", episode_reward)
        return episode_reward


    def mini_batch_train(self, max_episodes, max_steps, batch_size, milestones = [], look_for_continually_increasing_reward=False):

        agent_model_state_dicts = []

        training_episode_rewards = []
        evaluate_milestone_rewards = []

        last_episode_losses = []

        all_losses = []

        max_episode_reward = -9999999
        for episode in range(max_episodes):

            
            state = self.env.reset()
            episode_reward = 0
            

            for step in range(max_steps):
                action = self.get_train_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward

                if len(self.replay_buffer) > batch_size:
                    loss = self.update(batch_size)  
                    # if episode == max_episodes-1:
                    #     last_episode_losses.append(loss)
                    # if loss > 10:

                        # print('loss ', loss)

                    all_losses.append(loss)

                if done or step == max_steps-1:
                    training_episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    break

                state = next_state

            if episode_reward > max_episode_reward:
                torch.save(self.model.state_dict(), 'saved_models/best_model.pt')
                torch.save(self.target_model.state_dict(), 'saved_models/best_target_model.pt')
                max_episode_reward = episode_reward
            
          
            

            if episode in milestones:
                # eval_reward = self.evaluate(max_steps)
                # trying something, maybe change later
                eval_reward = episode_reward

                if look_for_continually_increasing_reward:
                    if len(evaluate_milestone_rewards) > 0:
                        if eval_reward < evaluate_milestone_rewards[-1]:
                            print('Continually increasing reward not found. Restarting')
                            return None, None, None, True
                            
                evaluate_milestone_rewards.append(eval_reward)

                agent_model_state_dicts.append(copy.deepcopy(self.model.state_dict()))

        # reducing learning rate every episode
        # agent.scheduler.step()
        # print('ALL LOSSES ', all_losses[:100])
        plt.plot(all_losses)
        plt.show()

        return agent_model_state_dicts, training_episode_rewards, evaluate_milestone_rewards, False