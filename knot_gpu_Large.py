
import gym
import gym_knot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import torch.multiprocessing as mp
import shutil
import pandas as pd
import ast
from adabelief_pytorch import AdaBelief

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
    #state_t = torch.from_numpy(state.astype('float32')).cuda()
    state_t = torch.FloatTensor(state).cuda()

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
      
      # Do state.stack because we have a list of tensors, and we need a tensor of stuff.
      
      state = torch.stack(state, dim=1)      

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


# Create customized penalized tanh
class PenalizedTanH(nn.Module):
    def __init__(self):
        super(PenalizedTanH, self).__init__()

    def forward(self, x):
        condition1 = x > 0
        return torch.where(condition1, torch.tanh(x), 0.25*torch.tanh(x))


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
    
    self.net = nn.Sequential(nn.Linear(state_size, hidden_size*3),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*3, hidden_size*4),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*4, hidden_size*6),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*6, hidden_size*8),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*8, hidden_size*12),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*12, hidden_size*8),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*8, hidden_size*6),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*6, hidden_size*4),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*4, hidden_size*2),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*2, action_size),
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
  
    self.net = nn.Sequential(nn.Linear(state_size, hidden_size*3),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*3, hidden_size*4),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*4, hidden_size*6),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*6, hidden_size*8),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*8, hidden_size*12),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*12, hidden_size*8),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*8, hidden_size*6),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*6, hidden_size*4),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*4, hidden_size*2),
                             PenalizedTanH(),
                             nn.Linear(hidden_size*2, 1))
    
  def forward(self, x):
    """Estimate value given state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          value (tensor): estimated value, size (batch)
    """
    return self.net(x)

def main():
  # Helper stuff to parse worldsize and rank and whatnot from commandline.
  parser = argparse.ArgumentParser()
  parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
  parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of gpus per node')
  parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
  parser.add_argument('--epochs', default=2, type=int, metavar='N', help='Number of total epochs to run')
  args = parser.parse_args()

  args.world_size = args.gpus * args.nodes
  #os.environ['MASTER_ADDR'] = "$(scontrol show job $SLURM_JOBID | awk -F= '/BatchHost/ {print $2}')"
  os.environ['MASTER_ADDR'] = "127.0.0.1"
  os.environ['MASTER_PORT'] = "54263"
 
  mp.spawn(ppo_main, nprocs=args.gpus, args=(args,))

def knot_sampler(df, low, upper):
    # Right now, just get knots where lower == upper
    filtered = df[df.euler.apply(lambda x: x[0] == x[1])]
    filtered = filtered[filtered.word.apply(lambda x: low <= len(x) <= upper)]

    chosen = filtered.sample(1)
    index = chosen['index'].values[0]
    word = chosen['word'].values[0]
    euler_lower = chosen['euler'].values[0][0]
    euler_upper = chosen['euler'].values[0][1]
    return word, euler_lower, euler_upper

  
def ppo_main(gpu, args): 
  # Calculate rank to get things done.
  rank = args.nr * args.gpus + gpu

  #init_method = "tcp://" + master_addr + ":54263"
 
  # This needs to be called in order to run data parallelization.
    # backend='nccl' is the necessary parameter when working with GPUs.
  #torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)
  torch.distributed.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:54263", world_size=args.world_size, rank=rank)
  # Hyper parameters
  lr = 5e-3
  epochs = 200
  env_samples = 100
  # gamma = 0.9221
  gamma = 1
  batch_size = 256
  epsilon = 0.15
  policy_epochs = 30
  tqdm_epoch = epochs * env_samples
  load = True
  low_knot, up_knot = 8, 14

  # Read in .parquet file with all the good stuff.
  knot_info = pd.read_parquet('knot_info.parquet')
  start, lower, upper = knot_sampler(knot_info, low_knot, up_knot)

  # Init environment 
  state_size = 227
  action_size = 13
  #start, lower, upper = knot_sampler(knot_info)
  #start, lower, upper = '4_1', -10, 10
  env = gym.make("Slice-v0", knot=start, lower=lower, upper=upper)

  torch.cuda.set_device(gpu)

  checkpoint = torch.load(f'training_files/knot_gpu_optim_Large_{up_knot-1}.pt')

  # Init networks
  policy_network = PolicyNetwork(state_size, action_size).cuda(gpu)
  policy_network = nn.parallel.DistributedDataParallel(policy_network, device_ids=[rank])
  #policy_network.load_state_dict(checkpoint['policy_network'])

  value_network = ValueNetwork(state_size).cuda(gpu)
  value_network = nn.parallel.DistributedDataParallel(value_network, device_ids=[rank])
  #value_network.load_state_dict(checkpoint['value_network'])


  # Init optimizer
  #optim = torch.optim.Adam(chain(policy_network.parameters(),     value_network.parameters()), lr=lr)

  optim = AdaBelief(chain(policy_network.parameters(),
                          value_network.parameters()), lr=lr)
  #optim.load_state_dict(checkpoint['optim'])
  #completed_epochs = checkpoint['epoch']
  #epochs += completed_epochs
  
  # If load is true, load in previously trained training files.
  if load:
    policy_network.load_state_dict(checkpoint['policy_network'])
    value_network.load_state_dict(checkpoint['value_network'])
    optim.load_state_dict(checkpoint['optim'])
    completed_epochs = checkpoint['epoch']
    epochs += completed_epochs  

  else:
    completed_epochs = 0  


  # Start main loop
  results_ppo = []
  return_list = pd.DataFrame(columns=['Braid','Epoch','Step', 'Reward', 'Action_list'])
  chosen_action = []
  loop = tqdm(total=tqdm_epoch, position=0, leave=True, colour='YELLOW')
  solved = False
  attempted_epochs = 0
  knot_list = ['4_1', '7_1', '7_2', '8_1', '8_4', '9_1', "9_9", "10_1", "10_2", "11a_1", "11a_14", "11a_34",
               '11n_5', '11n_17', '11n_23', '11n_185', '12a_4', '12a_13', '12n_14', '12n_22', 'H1', 'K1']
  #knot_list = ['K1', 'K2', 'K3', 'K4', 'K5']
  correct_solved = 0
  current_try = 0
  current_knot = 0
  #writer = SummaryWriter(log_dir='runs/' + str(knot_list[current_knot % len(knot_list)]))
  #print(f'info: {env.info()}')
  writer = SummaryWriter(log_dir=f'runs/optim_euler_Large_{low_knot}_{up_knot}_{rank}')
  # for epoch in range(completed_epochs, epochs):
  for epoch in range(epochs):
    ######################
    last_epoch = (epoch+1 == epochs)
    # Record only last epoch
    # if last_epoch:
    #   env = wrap_env(env)
    ######################
    
    memory = []  # Reset memory every epoch
    rewards = []  # Calculate average episodic reward per epoch
    chosen_action = []

    # Begin experience loop
    for episode in range(env_samples):
      
      # Reset environment
      state = env.reset()
      done = False
      rollout = []
      cum_reward = 0  # Track cumulative reward
      chosen_action = []

      

      '''
      Trying something new for selecting new knots. Keeping this just in case.

      # If it solved the braid, reset the environment to choose a new knot.
      if solved:
        solved = False
        env = gym.make("Slice-v0")
        state = env.reset()
        attempted_epochs = 0

      # If the agent has been trying for 50 epochs and hasn't solved it, try a different knot
      if attempted_epochs == 50:
        solved = False
        env = gym.make("Slice-v0")
        state = env.reset()
        attempted_epochs = 0
      '''

      # If we solved the knot 20 times, move onto the next knot.
      if correct_solved == 20:
        writer.flush()
        current_knot += 1
        env.close()
        start, lower, upper = knot_sampler(knot_info, low_knot, up_knot)
        #env = gym.make("Slice-v0", knot=knot_list[current_knot % len(knot_list)], lower=lower, upper=upper)
        env = gym.make("Slice-v0", knot=start, lower=lower, upper=upper)
        #env.starting_braid = knot_list[current_knot % len(knot_list)]  # This way it will loop through all the knots continuously.
        #env.starting_braid, env.euler_lower, env.euler_upper = knot_sampler(knot_info)
        state = env.reset()
        #print(f'info (correct_solved):')
        #env.info()
        correct_solved = 0
        #writer = SummaryWriter(log_dir='runs/' + str(knot_list[current_knot % len(knot_list)]))
        writer = SummaryWriter(log_dir=f'runs/optim_euler_Large_{low_knot}_{up_knot}_{rank}')


      # Begin episode
      while not done and cum_reward < 200:  # End after 200 steps   
        # Get action
        action, action_dist = get_action_ppo(policy_network, state)
        chosen_action.append(action)
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        
        # env.render()

        # Store step
        rollout.append((state, action, action_dist, reward))

        cum_reward += reward
        state = next_state  # Set current state

      if done and cum_reward > -50:
        # Save and update the files.
        new = pd.DataFrame([[env.starting_braid, epoch, episode, round(cum_reward, 3), chosen_action]], columns = ['Braid', 'Epoch','Step', 'Reward', 'Action_list'])
        return_list = pd.concat([return_list, new])

        # Put files on my computer.
        os.makedirs('Result_csv/', exist_ok=True)
        return_list.to_csv(f"Result_csv/results_L_{low_knot}_{up_knot}_{rank}.csv")
        # print(f"Return_list (should have the good stuff) {return_list}")
        solved = True
        correct_solved += 1

      # Increment total tries. If it has tried a bunch and can't get it, get a new knot.
      current_try += 1
      if current_try >= 500:
        writer.flush()
        current_knot += 1
        env.close()
        start, lower, upper = knot_sampler(knot_info, low_knot, up_knot)
        #env = gym.make("Slice-v0", knot=knot_list[current_knot % len(knot_list)], lower=lower, upper=upper)
        env = gym.make("Slice-v0", knot=start, lower=lower, upper=upper)
        #env.starting_braid = knot_list[current_knot % len(knot_list)]  # This way it will loop through all the knots continuously.
        #env.starting_braid, env.euler_lower, env.euler_upper = knot_sampler(knot_info)
        state = env.reset()
        #print(f'info (correct_solved):')
        #env.info()
        correct_solved = 0
        current_try = 0
        #writer = SummaryWriter(log_dir='runs/' + str(knot_list[current_knot % len(knot_list)]))
        writer = SummaryWriter(log_dir=f'runs/optim_euler_Large_{low_knot}_{up_knot}_{rank}')

      # Calculate returns and add episode to memory
      memory = calculate_return(memory, rollout, gamma)

      rewards.append(cum_reward)
      # logging_df.loc[len(logging_df)] = [epoch, cum_reward]
      writer.add_scalar("Total_Reward", cum_reward, epoch+completed_epochs)
      #writer.add_scalar('Current knot', env.starting_braid)

      writer.flush()
      loop.update(1)
      loop.set_description("Knot: {} Epoch: {} Step: {} Reward: {}".format(len(env.starting_braid), epoch, episode, round(cum_reward, 3)))
      ######################
      env.close()
      ######################

    # Train
    dataset = RLDataset(memory)

    # Use "distributed aware" sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    learn_ppo(optim, policy_network, value_network, loader, epsilon, policy_epochs)

    '''
    Put the checkpoint saver here!
    '''
    if correct_solved == 1:
        torch.save({'policy_network': policy_network.state_dict(),
                    'value_network': value_network.state_dict(),
                    'optim': optim.state_dict(),
                    'epoch': completed_epochs + epoch+1


        }, f'training_files/knot_gpu_optim_Large_{up_knot}.pt')  # Changed to knot_gpu_12 because I am introducing a 12 crossing knot. 
                                             # I am doing this to make sure our original agent does not break.
    
    
    # Print results
    results_ppo.extend(rewards)  # Store rewards for this epoch
    writer.flush()
    loop.update(1)
    loop.set_description("Knot: {} Epochs: {} Reward: {}".format(len(env.starting_braid), epoch, round(results_ppo[-1], 3)))
    attempted_epochs += 1
  #os.makedir('fslhome/dskinne3', exist_ok=True)
  #logging_df.to_csv('fsl/dskinne/knot_gpu.csv')
  writer.close()
  #np.savetxt("results.csv", return_list, delimiter=',')

  return results_ppo

if  __name__ == '__main__':
    print('Starting main')
    main()

# results_ppo = main()
######################
# show_video()
######################
