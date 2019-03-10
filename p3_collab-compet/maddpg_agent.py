import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ou_noise import OUNoise
from replay_buffer import ReplayBuffer
from utils import hard_update, soft_update

import random

from model import Actor, Critic

def rand_from_seed(seed):
    random.seed(seed)
    return random.randint(0, (1 << 32) - 1)

class MADDPG:
    def __init__(
        self,
        state_size,
        action_size,
        num_agents,
        init_noise_scale=1.0,
        noise_annealing=1.0,
        lr_actor=1e-3,
        lr_critic=1e-3,
        device=torch.device('cpu'),
        replay_buffer_size=int(1e6),
        batch_size=1024,
        seed=42,
        tau=1e-3,
        gamma=0.99,
        update_every=1,
        tensorboard_writer=None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.init_noise_scale = init_noise_scale
        self.noise_scale = init_noise_scale
        self.noise_annealing = noise_annealing
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.tau = tau
        self.step_idx = 0
        self.gamma = gamma
        self.update_every = update_every
        self.tensorboard_writer = tensorboard_writer
        
        seed = rand_from_seed(seed)
        self.replay_buffer = ReplayBuffer(
            action_size, replay_buffer_size, batch_size, seed, device)                    
        
        self.agents = []
        
        for agent_idx in range(num_agents):
            seed = rand_from_seed(seed)
            self.agents.append(
                DDPG(state_size, action_size, tau, lr_actor, \
                     lr_critic, num_agents, agent_idx, seed, \
                     device, gamma, tensorboard_writer).to(self.device)
            )
        
        #         self.agents = [
#             DDPG(state_size, action_size, tau, lr_actor, \
#                  lr_critic, num_agents, agent_idx, None, \
#                  device, gamma, tensorboard_writer).to(self.device) for agent_idx in range(num_agents)]  

    def act(self, all_actor_states, noise_scale=1.0, use_noise=True):        
        all_actor_actions = []
        for agent_idx, agent in enumerate(self.agents):
            curr_action = agent.act(all_actor_states[agent_idx], noise_scale, use_noise)
            all_actor_actions.append(curr_action)
        return np.array(all_actor_actions)
    
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        
        self.replay_buffer.add(
            all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        if len(self.replay_buffer) >= self.batch_size and self.step_idx % self.update_every == 0:
            experiences = [self.replay_buffer.sample() for _ in range(self.num_agents)]
            self.learn(experiences)
            
        self.step_idx += 1
        
    
    def learn(self, experiences):
        
        all_curr_pred_actions = []
        all_next_pred_actions = []
                      
        for agent in self.agents:
            agent_device_idx = torch.tensor([agent.agent_idx]).to(self.device)
            states, actions, rewards, next_states, dones = experiences[agent.agent_idx]
            curr_action = agent.actor_local(states.index_select(1, agent_device_idx).squeeze(1))
            all_curr_pred_actions.append(curr_action)
            next_action = agent.actor_target(next_states.index_select(1, agent_device_idx).squeeze(1))
            all_next_pred_actions.append(next_action)
            
        for agent in self.agents:
            agent.learn(experiences[agent.agent_idx], all_curr_pred_actions, all_next_pred_actions)            

    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def save(self):
        for agent in self.agents:
            torch.save(agent.actor_local.state_dict(), "agent_{}_actor.pt".format(agent.agent_idx))
            torch.save(agent.critic_local.state_dict(), "agent_{}_critic.pt".format(agent.agent_idx))

            
class DDPG:
    def __init__(self,
                state_size,
                action_size,                
                tau,
                lr_actor,
                lr_critic,
                num_agents,
                agent_idx,
                seed,
                device,
                gamma,
                tensorboard_writer=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_agents = num_agents
        self.agent_idx = agent_idx
        self.seed = seed       
        self.device = device
        self.gamma = gamma
        random.seed(seed)
        self.tensorboard_writer = tensorboard_writer        
        
        self.actor_local = Actor(state_size, action_size, seed)
        self.actor_target = Actor(state_size, action_size, seed)
        
        critic_state_size = (state_size + action_size) * num_agents
        
        self.critic_local = Critic(critic_state_size, seed)
        self.critic_target = Critic(critic_state_size, seed)
        
        hard_update(self.actor_local, self.actor_target)
        hard_update(self.critic_local, self.critic_target) 
        
        self.actor_optim = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        self.noise = OUNoise(action_size, seed)
        
        self.iteration = 0
        
    def to(self, device):
        self.actor_local.to(device)
        self.actor_target.to(device)
        self.critic_local.to(device)
        self.critic_target.to(device)
        return self
                             
    def act(self, state, noise_scale, use_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if use_noise:
            action += self.noise.sample() * noise_scale
        return np.clip(action, -1, 1)
    
    def learn(self, experiences, all_curr_pred_actions, all_next_pred_actions):
        
        agent_idx_device = torch.tensor(self.agent_idx).to(self.device)
        
        states, actions, rewards, next_states, dones = experiences

        rewards = rewards.index_select(1, agent_idx_device)
        dones = dones.index_select(1, agent_idx_device)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
                
        batch_size = next_states.shape[0]
        
        actions_next = torch.cat(all_next_pred_actions, dim=1).to(self.device)
        next_states = next_states.reshape(batch_size, -1)      
        
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        states = states.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)
        
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_optim.zero_grad()
        predicted_actions = torch.cat([action if idx == self.agent_idx \
                   else action.detach()
                   for idx, action in enumerate(all_curr_pred_actions)],
                   dim=1).to(self.device)

        actor_loss = -self.critic_local(states, predicted_actions).mean()
        # minimize loss
        actor_loss.backward()
        self.actor_optim.step()
        
        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        
        if self.tensorboard_writer is not None:            
            self.tensorboard_writer.add_scalar("agent{}/actor_loss".format(self.agent_idx), al, self.iteration)
            self.tensorboard_writer.add_scalar("agent{}/critic_loss".format(self.agent_idx), cl, self.iteration)
            self.tensorboard_writer.file_writer.flush()
            
        self.iteration += 1

        # ----------------------- update target networks ----------------------- #
        soft_update(self.critic_target, self.critic_local, self.tau)
        soft_update(self.actor_target, self.actor_local, self.tau)           

    
    def reset(self):
        self.noise.reset()
        
