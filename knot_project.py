
import gym
import gym_knot
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

####################
from gym import logger as gymlogger
#from gym.wrappers import Monitor
gymlogger.set_level(40) #error only

import glob
import io
import base64

def calculate_return(memory, rollout, gamma):
  """Return memory with calculated return in experience tuple

    Args:
        memory (list): (state, action, action_dist, return) tuples
        rollout (list): (state, action, action_dist, reward) tuples from last rollout
        gamma (float): discount factor
          Gamma = 0.9

    Returns:
        list: memory updated with (state, action, action_dist, return) tuples from rollout
  """
  # Calculating the return.
    # All we are doing is appending what has happened in the rollout to past memory,
    # and calculating the true reward.
  calc_return = 0
  for state, action, action_dist, reward in reversed(rollout):
    calc_return = reward + calc_return*gamma
    memory.append((state, action, action_dist, calc_return))

  return memory


def get_action_ppo(network, state):
  """Sample action from the distribution obtained from the policy network

    Args:
        network (PolicyNetwork): Policy Network
        state (np-array): current state, size (state_size)
          State size = 4

    Returns:
        int: action sampled from output distribution of policy network
        array: output distribution of policy network
  """
  # with torch.no_grad() stops this stuff from wanting a gradient
    # (This was done to stop an error I was getting. Thanks, Connor!)
  with torch.no_grad():
    # Converting the current state from list to Float32 tensor and putting it on the GPU.
    state_t = torch.from_numpy(state.astype('float32')).cuda()

    # Sampling an action from the network based on the current state.
    sample_action = network(state_t.unsqueeze(0))  # Distribution

    # Chosing an action from the sample_action distribution,
      # torch.multinomial(sample_action, 1) will sample one action from the sample action distribution.
    chosen_action = torch.multinomial(sample_action, 1)

  return chosen_action.item(), sample_action


def learn_ppo(optim, policy, value, memory_dataloader, epsilon, policy_epochs):
  """Implement PPO policy and value network updates. Iterate over your entire 
     memory the number of times indicated by policy_epochs (policy_epochs = 5).    

    Args:
        optim (Adam): value and policy optimizer
        policy (PolicyNetwork): Policy Network
          policy_network = PolicyNetwork(state_size, action_size).cuda()
        value (ValueNetwork): Value Network
          value_network = ValueNetwork(state_size).cuda()
        memory_dataloader (DataLoader): dataloader with (state, action, action_dist, return) tensors
        epsilon (float): trust region
        policy_epochs (int): number of times to iterate over all memory
  """
  # Go through all epochs
  for epoch in range(policy_epochs):
    for batch in memory_dataloader:
      optim.zero_grad()
      
      # Get the all variables from memory (by batch)
      state, action, action_dist, return_v = batch

      # Convert each of the above variables into their appropriate dtype and putting them on the GPU
      state_t       = state.type(torch.FloatTensor).cuda()
      action_t      = action.type(torch.LongTensor).cuda()
      action_dist_t = action_dist.type(torch.FloatTensor).cuda()
      return_v_t    = return_v.type(torch.FloatTensor).cuda()

      # Calculate advantage Â
      advantage = (return_v_t - value(state_t).cuda()).detach()

      # Calculate value loss
      value_loss = F.mse_loss(return_v_t, value(state_t).squeeze())

      # Calculate π(s,a)
      policy_norm       = action_dist_t.squeeze(1)
      action_onehot     = F.one_hot(action_t, 13).bool()
      taken_policy_norm = policy_norm[action_onehot]
      
      #Calculate π'(s,a)
      policy_prime       = policy(state_t)
      taken_policy_prime = policy_prime[action_onehot]
      
      #Calculate the ratio between π'(s,a)/π(s,a)
      prim_div_norm = taken_policy_prime / taken_policy_norm

      # Clipping π'(s,a)/π(s,a)
      clip = torch.clip(prim_div_norm, 1-epsilon, 1+epsilon)
      
      # Left part of the policy loss (π'(s,a)/π(s,a))*Â
      left_part = prim_div_norm * advantage

      # Right part of the policy loss clip*Â
      right_part = clip * advantage

      # Calculating policy loss
      policy_loss = torch.mean(torch.min(left_part, right_part))

      # Total loss
      total_loss = value_loss - policy_loss
      
      total_loss.backward()
      optim.step()



    # Go through entire dataset
      # Do everything else

# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
  def __init__(self, data):
    super().__init__()
    self.data = []
    for d in data:
      self.data.append(d)
  
  def __getitem__(self, index):
    return self.data[index]
 
  def __len__(self):
    return len(self.data)


# Policy Network
class PolicyNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    super().__init__()
    hidden_size = 8
    
    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, action_size),
                             nn.Softmax(dim=1))
  
  def forward(self, x):
    """Get policy from state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          action_dist (tensor): probability distribution over actions (batch x action_size)
    """
    return self.net(x)
  

# Value Network
class ValueNetwork(nn.Module):
  def __init__(self, state_size):
    super().__init__()
    hidden_size = 8
  
    self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, hidden_size),
                             nn.ReLU(),
                             nn.Linear(hidden_size, 1))
    
  def forward(self, x):
    """Estimate value given state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          value (tensor): estimated value, size (batch)
    """
    return self.net(x)

def ppo_main():
  # Hyper parameters
  lr = 1e-3
  epochs = 20
  env_samples = 100
  gamma = 0.9
  batch_size = 256
  epsilon = 0.2
  policy_epochs = 5

  # Init environment 
  state_size = 227
  action_size = 13
  env = gym.make("Slice-v0")

  # Init networks
  policy_network = PolicyNetwork(state_size, action_size).cuda()
  value_network = ValueNetwork(state_size).cuda()

  # Init optimizer
  optim = torch.optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr)

  # Start main loop
  results_ppo = []
  loop = tqdm(total=epochs, position=0, leave=False)
  for epoch in range(epochs):
    ######################
    last_epoch = (epoch+1 == epochs)
    # Record only last epoch
    # if last_epoch:
    #   env = wrap_env(env)
    ######################
    
    memory = []  # Reset memory every epoch
    rewards = []  # Calculate average episodic reward per epoch

    # Begin experience loop
    for episode in range(env_samples):
      
      # Reset environment
      state = env.reset()
      done = False
      rollout = []
      cum_reward = 0  # Track cumulative reward

      # Begin episode
      while not done and cum_reward < 200:  # End after 200 steps   
        # Get action
        action, action_dist = get_action_ppo(policy_network, state)
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        # env.render()

        # Store step
        rollout.append((state, action, action_dist, reward))

        cum_reward += reward
        state = next_state  # Set current state

      # Calculate returns and add episode to memory
      memory = calculate_return(memory, rollout, gamma)

      rewards.append(cum_reward)
      ######################
      env.close()
      ######################
    # Train
    dataset = RLDataset(memory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    learn_ppo(optim, policy_network, value_network, loader, epsilon, policy_epochs)
    
    # Print results
    results_ppo.extend(rewards)  # Store rewards for this epoch
    loop.update(1)
    loop.set_description("Epochs: {} Reward: {}".format(epoch, results_ppo[-1]))

  return results_ppo

results_ppo = ppo_main()
######################
# show_video()
######################
