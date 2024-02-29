import gym
import gym_knot
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
# import pandas as pd  <-- For some reason, Pandas is not working on the supercomputer.
import os
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from gym import logger as gymlogger
# from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import glob
import io
import base64

def get_action_dqn(network, state, epsilon, epsilon_decay):
  """Select action according to e-greedy policy and decay epsilon

    Args:
        network (QNetwork): Q-Network
        state (list): current state, size (state_size)
        epsilon (float): probability of choosing a random action
        epsilon_decay (float): amount by which to decay epsilon

    Returns:
        action (int): chosen action [0, action_size)
        epsilon (float): decayed epsilon
  """
  # Step 1: Turn the state from list to tensor and pass to gpu.
  tensor_state = torch.FloatTensor(state).cuda()

  # Step 2: Get a random number to see if we choose random step or not.
  random_chance = np.random.random()

  # Step 3.0: If we choose random step, get the q-values from that
  if random_chance <= epsilon:
    # Note: we have np.random.random_sample((2,)) because there are only two actions for our cartpole environment.
      # So np.random.random_sample((2,)) will return two random 'probabilities.'
    next_choice = torch.from_numpy(np.random.random_sample((2,)).astype('float32'))

  # Step 3.1: If we do not choose a random step, plug the tensor_state into the network to get the best choice from q-table. 
    # Note: Make sure to unsqueeze because we need a 'batch.'
  elif random_chance > epsilon:
    next_choice = network(tensor_state.unsqueeze(0))
  
  # Step 4: Take the argmax of next_choice to get the actual choice we will take.
  final_choice = torch.argmax(next_choice)

  # Step 5: Calculate the new epsilon with its decay
  new_epsilon = epsilon*epsilon_decay
  return final_choice.item(), new_epsilon


def prepare_batch(memory, batch_size):
  """Randomly sample batch from memory
     Prepare cuda tensors

    Args:
        memory (list): state, action, next_state, reward, done tuples
        batch_size (int): amount of memory to sample into a batch

    Returns:
        state (tensor): float cuda tensor of size (batch_size x state_size)
          This is size (32 x 4)
        action (tensor): long tensor of size (batch_size)
          This is size (32) because there are 32 opportunities to take actions.
        next_state (tensor): float cuda tensor of size (batch_size x state_size)
          This is size (32 x 4)
        reward (tensor): float cuda tensor of size (batch_size)
          This is size (32)
        done (tensor): float cuda tensor of size (batch_size)
          This is size (32)
  """
  # state_size = 4
  # batch_size = 32
  # random_sample_batch is a list of size 32 that is one of the elements of the memory list (non-repeating).
  random_sample_batch = random.sample(memory, batch_size)

  # Get necessary tensors
  state      = [item[0] for item in random_sample_batch]
  action     = [item[1] for item in random_sample_batch]
  next_state = [item[2] for item in random_sample_batch]
  reward     = [item[3] for item in random_sample_batch]
  done       = [item[4] for item in random_sample_batch]
  #print(f"done {done}\n")
  #print(f"done as int {done.int()}")

  # Turn those tensors into the needed tensors as perscribed above.
  state_t      = torch.FloatTensor(state).cuda()
  action_t     = torch.LongTensor(action).cuda()
  next_state_t = torch.FloatTensor(next_state).cuda()
  reward_t     = torch.FloatTensor(reward).cuda()
  done_t       = torch.FloatTensor(done).cuda()

  return state_t, action_t, next_state_t, reward_t, done_t

  
def learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update):
  """Update Q-Network according to DQN Loss function
     Update Target Network every target_update global steps

    Args:
        batch (tuple): tuple of state, action, next_state, reward, and done tensors
        optim (Adam): Q-Network optimizer
        q_network (QNetwork): Q-Network
        target_network (QNetwork): Target Q-Network
        gamma (float): discount factor
        global_step (int): total steps taken in environment
        target_update (int): frequency of target network update
  """
  # Step 1: Separate tuple
  state, action, next_state, reward, done_tensor = batch

  # Step 2: Clear gradients.
  optim.zero_grad()

  # Step 3: Get y_truth from the target_network (pass in next_state to get prediction)
  y_truth = target_network(next_state)  # This gives us a bunch of predicted values
  #print(f"y_truth.shape\t\t\t {y_truth.shape}\n")

  # Step 4: Get y_hat from the q_network
  y_hat = q_network(state)
  action_onehot = F.one_hot(action, 13).bool()

  
  # Step 5: Get the reward + gamma*argmax(future)*(1-dones)
  second_half_of_equation = reward + (gamma * torch.max(y_truth, dim=1)[0] * (1-done_tensor.long()))

  # Step 5: Get loss
  loss = F.mse_loss(y_hat[action_onehot], second_half_of_equation)

  # Step 6: Do normal nn stuff
  loss.backward()
  optim.step()

  
  if global_step % target_update == 0:
    # Question: How do I update target network?
    target_network.load_state_dict(q_network.state_dict())  # ?
  pass

class QNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    super().__init__()
    hidden_size = 8
    
    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, action_size))  
    
  def forward(self, x):
    """Estimate q-values given state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          q-values (tensor): estimated q-values, size (batch x action_size)
    """
    return self.net(x)

def dqn_main():
  # Hyper parameters
  lr = 1e-3
  epochs = 500
  start_training = 1000
  gamma = 0.99
  batch_size = 256
  epsilon = 1
  epsilon_decay = .9999
  target_update = 1000
  learn_frequency = 2

  # Init environment
  state_size = 227
  action_size = 13
  env = gym.make('Slice-v0')

  # Init networks
  q_network = QNetwork(state_size, action_size).cuda()
  target_network = QNetwork(state_size, action_size).cuda()
  target_network.load_state_dict(q_network.state_dict())

  writer = SummaryWriter()

  # Init optimizer
  optim = torch.optim.Adam(q_network.parameters(), lr=lr)

  # Init replay buffer
  memory = []

  # Begin main loop
  results_dqn = []
  global_step = 0
  loop = tqdm(total=epochs, position=0, leave=False)
  for epoch in range(epochs):
    ######################
    last_epoch = (epoch+1 == epochs)
    # Record the last epoch, not the previous epochs
    # if last_epoch:
    #   env = wrap_env(env)
    ######################

    # Reset environment
    state = env.reset()
    done = False
    cum_reward = 0  # Track cumulative reward per episode

    # Begin episode
    while not done and cum_reward < 200:  # End after 200 steps 
      # Select e-greedy action
      action, epsilon = get_action_dqn(q_network, state, epsilon, epsilon_decay)

      # Take step
      next_state, reward, done, _ = env.step(action)
      # env.render()

      # Store step in replay buffer
      memory.append((state, action, next_state, reward, done))

      cum_reward += reward
      global_step += 1  # Increment total steps
      state = next_state  # Set current state

      # If time to train
      if global_step > start_training and global_step % learn_frequency == 0:

        # Sample batch
        batch = prepare_batch(memory, batch_size)
        
        # Train
        learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update)

    writer.add_scalar("Cum_reward", cum_reward, epoch)
    ######################
    env.close()
    ######################
    # Print results at end of episode
    results_dqn.append(cum_reward)
    loop.update(1)
    loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))
    writer.flush()
  
  writer.close()
  
  return results_dqn

results_dqn = dqn_main()
######################
# show_video()
######################
