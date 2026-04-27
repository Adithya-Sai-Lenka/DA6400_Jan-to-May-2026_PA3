import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

# ==========================================
# 1. NETWORKS & REPARAMETERIZATION TRICK
# ==========================================

def weight_init(m):
    """Custom weight initialization from pytorch_sac"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class DoubleQCritic(nn.Module):
    """Critic: Learns Action-Value (Q) only. Clipped Double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1 Architecture
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 Architecture
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class SquashedGaussianActor(nn.Module):
    """Actor: Squashed Gaussian Policy (tanh) using Reparameterization Trick"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_bounds=(-20, 2)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = log_std.exp()
        
        # Reparameterization trick: rsample() allows gradients to flow back
        dist = Normal(mu, std)
        x_t = dist.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Enforcing Action Bound via tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob

# ==========================================
# 2. SAC AGENT
# ==========================================

class SACAgent:
    def __init__(self, obs_dim, action_dim, device, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
        self.critic = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Adam optimizer for both actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)

        # CRITIC UPDATE (Clipped Double Q-Learning)
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            # Take the minimum of the two target Q-values
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + (not_done * self.gamma * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ACTOR UPDATE
        actor_action, log_prob = self.actor(obs)
        Q1, Q2 = self.critic(obs, actor_action)
        Q_pi = torch.min(Q1, Q2)
        
        # Maximize Q while maximizing entropy (minimize negative)
        actor_loss = (self.alpha * log_prob - Q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # SOFT UPDATE TARGET NETWORKS
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ==========================================
# 3. UTILITIES & REPLAY BUFFER
# ==========================================

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=1000000, device='cpu'):
        self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        self.obses[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obses[self.idx] = next_obs
        self.not_dones[self.idx] = not done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        return (
            torch.as_tensor(self.obses[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.next_obses[idxs], device=self.device),
            torch.as_tensor(self.not_dones[idxs], device=self.device)
        )

def get_env(env_id):
    """Instantiate the correct environment wrapper based on task"""
    if env_id == "pendulum":
        return gym.make("Pendulum-v1")
    elif env_id == "lunar_lander":
        return gym.make("LunarLander-v3", continuous=True)
    elif env_id == "reacher_easy":
        # Requires shimmy[dm-control]
        return gym.make("dm_control/reacher-easy-v0")
    elif env_id == "reacher_hard":
        return gym.make("dm_control/reacher-hard-v0")
    else:
        raise ValueError("Invalid environment specified.")

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train(env_id="pendulum", num_steps=1000000, random_steps=10000):
    env = get_env(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # gamma = 0.99 is set here as requested
    agent = SACAgent(obs_dim, action_dim, device, gamma=0.99)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, device=device)
    
    obs, _ = env.reset()
    episode_reward = 0
    
    for step in range(num_steps):
        # 10K STEPS RANDOM ACTION EXPLORATION PHASE
        if step < random_steps:
            action = env.action_space.sample()
            scaled_action = action / max_action # Normalize for the buffer
        else:
            scaled_action = agent.select_action(obs)
            action = scaled_action * max_action 
            
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Add unsquashed/normalized action to buffer so critic processes [-1, 1]
        replay_buffer.add(obs, scaled_action, reward, next_obs, terminated)
        
        obs = next_obs
        episode_reward += reward
        
        if step >= random_steps:
            agent.update(replay_buffer)
            
        if done:
            print(f"Step: {step+1} | Env: {env_id} | Reward: {episode_reward:.2f}")
            obs, _ = env.reset()
            episode_reward = 0

if __name__ == "__main__":
    # Choose environment: "pendulum", "lunar_lander", "reacher_easy", "reacher_hard"
    print("Starting Training...")
    train(env_id="lunar_lander")