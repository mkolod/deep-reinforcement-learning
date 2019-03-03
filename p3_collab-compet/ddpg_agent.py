import numpy as np
import copy
from collections import namedtuple, deque
import random
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

class Agent():
    def __init__(self, 
                 state_size,
                 action_size,
                 device,
                 agent_id,
                 seed=0,
                 buffer_size=int(1e6),
                 batch_size=1024,
                 gamma=0.99,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=3e-4,
                 relu_leak=0.01,
                 num_updates=10,
                 update_every=20,
                 init_noise=1.0,
                 end_noise=0.1,
                 noise_annealing=(1.0 - 1e-4),
                 writer=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            device (str): 'cpu' or 'cuda:device_id'
            seed (int): random seed
            buffer_size (int): reply buffer size
            batch_size (int): batch size for learning
            gamma (float): discount rate
            tau (float): interpolation parameter for soft_update
            lr_actor (float): learning rate for actor network
            lr_critic (float): learning rate for critic network
            relu_leak (float): the leak rate for leaky ReLU, i.e. the alpha in (x < 0) * alpha * x + (x >= 0) * x
            num_updates (int): number of samples and updates when updating update_every calls to step() 
            update_every (int): call self.learn every update_every steps - this allows for replay buffer changes
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_updates = num_updates
        self.update_every = update_every
        self.tau = tau
        self.agent_id = agent_id
        

        # Actor Network
        self.actor_local = Actor(state_size, action_size, leak=relu_leak, seed=seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, leak=relu_leak, seed=seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size*2, action_size, leak=relu_leak, seed=seed).to(self.device)
        self.critic_target = Critic(state_size*2, action_size, leak=relu_leak, seed=seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, self.device)
        self.steps = 0  
        self.init_noise = init_noise
        self.end_noise = end_noise
        self.noise_mul = init_noise
        self.noise_annealing = noise_annealing
        self.writer = writer

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.steps += 1
        self.memory.add(states, actions, rewards, next_states, dones)

        if (len(self.memory) > self.batch_size) and (self.steps % self.update_every == 0):
            # reset steps so we don't overflow at some point
            # self.steps = 0
            for _ in range(self.num_updates):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True, train=True):
        break_idx = self.state_size
        if self.agent_id == 0:
            start, stop = 0, break_idx
        else:
            start, stop = break_idx, break_idx*2
        local_states = states[:, start:stop]
        if local_states.ndim == 1:
            local_states = local_states.reshape(1,-1)
        local_states = torch.from_numpy(local_states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(local_states).cpu().data.numpy()
        if train:
            self.actor_local.train()
            if add_noise:
                actions += self.noise.sample() * self.noise_mul
        return np.clip(actions, -1, 1)
    
    def target_act(self, states, add_noise=True, train=True):
        break_idx = self.state_size
        if self.agent_id == 0:
            start, stop = 0, break_idx
        else:
            start_stop = break_idx, break_idx*2        
        local_states = states[:, start:stop]
        if local_states.ndim == 1:
            local_states = local_states.reshape(1,-1)        
        local_states = torch.from_numpy(local_states).float().to(self.device)
        self.actor_target.eval()
        with torch.no_grad():
            actions = self.actor_target(local_states).cpu().data.numpy()
        if train:
            self.actor_target.train()
            if add_noise:
                actions += self.noise.sample() * self.noise_mul
        return np.clip(actions, -1, 1)    

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        break_idx = self.state_size
        if self.agent_id == 0:
            start, stop = 0, break_idx
        else:
            start, stop = break_idx, break_idx*2                

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states[:, :, start:stop])
        Q_targets_next = self.critic_target(next_states, actions_next).squeeze()
        # Compute Q targets for current states (y_i)        
        dones = dones[:, self.agent_id]
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        
        break_idx = self.action_size
        if self.agent_id == 0:
            start, stop = 0, break_idx
        else:
            start, stop = break_idx, break_idx*2 
            
        this_actor_actions = actions[:, :, start:stop]
        Q_expected = self.critic_local(states, this_actor_actions)
        
        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(Q_expected.squeeze(), Q_targets.detach())
    
        # critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        
        break_idx = self.state_size
        if self.agent_id == 0:
            start, stop = 0, break_idx
        else:
            start, stop = break_idx, break_idx*2        
                    
        actions_pred = self.actor_local(states[:, :, start:stop])
        # There was a minus here
        actor_loss = self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
        self.noise_mul = max(self.end_noise, self.noise_mul*self.noise_annealing)
        
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        self.writer.add_scalar("agent_{}/actor_loss".format(self.agent_id), al, self.steps)
        self.writer.add_scalar("agent_{}/critic_loss".format(self.agent_id), cl, self.steps)
        self.writer.file_writer.flush()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)            
