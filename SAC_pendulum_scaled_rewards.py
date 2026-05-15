import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import concurrent.futures
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_num_threads(1)

# ==========================================
# 1. NETWORKS
# ==========================================
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.apply(weight_init)
    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_bounds=(-20, 2)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, obs, deterministic=False):
        x = self.net(obs)
        mu = self.mu_layer(x)
        if deterministic:
            return torch.tanh(mu), None
        log_std = torch.clamp(self.log_std_layer(x), self.log_std_bounds[0], self.log_std_bounds[1])
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample() 
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(-1, keepdim=True)

# ==========================================
# 2. UNIFIED SAC AGENT
# ==========================================
class SACAgent:
    def __init__(self, obs_dim, action_dim, device, alpha_type='auto', fixed_alpha=0.005, gamma=0.99, tau=0.005, lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_type = alpha_type
        
        if self.alpha_type == 'auto':
            self.target_entropy = -action_dim 
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.fixed_alpha = fixed_alpha

        self.actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
        self.critic = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach() if self.alpha_type == 'auto' else self.fixed_alpha

    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + (not_done * self.gamma * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action, log_prob = self.actor(obs)
        Q1, Q2 = self.critic(obs, actor_action)
        actor_loss = (self.alpha * log_prob - torch.min(Q1, Q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.alpha_type == 'auto':
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ==========================================
# 3. UTILITIES & ENVIRONMENT
# ==========================================
class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=100000, device='cpu'):
        self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.capacity, self.idx, self.full, self.device = capacity, 0, False, device

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
        return (torch.as_tensor(self.obses[idxs], device=self.device), torch.as_tensor(self.actions[idxs], device=self.device),
                torch.as_tensor(self.rewards[idxs], device=self.device), torch.as_tensor(self.next_obses[idxs], device=self.device),
                torch.as_tensor(self.not_dones[idxs], device=self.device))

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TargetPendulumWrapper(gym.Wrapper):
    def __init__(self, env, target_angle_deg, reward_scale=1.0):
        super().__init__(env)
        self.target_theta = np.deg2rad(target_angle_deg)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cos_th, sin_th, th_dot = obs
        current_theta = np.arctan2(sin_th, cos_th)
        diff = current_theta - self.target_theta
        angle_err = ((diff + np.pi) % (2 * np.pi)) - np.pi
        torque = np.clip(action[0], -2.0, 2.0)
        base_reward = -(angle_err**2 + 0.1 * th_dot**2 + 0.001 * torque**2)
        return obs, base_reward * self.reward_scale, terminated, truncated, info

def evaluate_policy(agent, target_angle, eval_episodes=20, seed=0):
    # Important: Evaluate with UN-SCALED rewards (scale=1.0) so plots are comparable
    eval_env = TargetPendulumWrapper(gym.make("Pendulum-v1", max_episode_steps=1000), target_angle, reward_scale=1.0)
    max_action = float(eval_env.action_space.high[0])
    avg_reward = 0.
    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True) * max_action
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
            done = terminated or truncated

    eval_env.close()
    return avg_reward / eval_episodes

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def run_experiment(args):
    target_angle, alpha_type, reward_scale, seed, total_steps, eval_freq = args
    set_seed(seed)
    
    env = TargetPendulumWrapper(gym.make("Pendulum-v1", max_episode_steps=1000), target_angle, reward_scale=reward_scale)
    device = torch.device('cpu')
    
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], device, 
                     alpha_type=alpha_type, fixed_alpha=0.005)
    replay_buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], device=device)
    
    obs, _ = env.reset(seed=seed)
    eval_metrics = []
    
    for step in range(0, total_steps + 1):
        if step <= 2000:
            action = env.action_space.sample() / float(env.action_space.high[0])
        else:
            action = agent.select_action(obs, deterministic=False)
            
        next_obs, reward, terminated, truncated, _ = env.step(action * float(env.action_space.high[0]))
        replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs
        
        if step > 2000:
            agent.update(replay_buffer)
            
        if terminated or truncated:
            obs, _ = env.reset()

        if step % eval_freq == 0:
            eval_reward = evaluate_policy(agent, target_angle, seed=seed)
            eval_metrics.append((step, eval_reward))
            print(f"Target Angle: {target_angle}° | Alpha Type: {alpha_type} | Reward Scale: {reward_scale} | Seed: {seed} | Step: {step} | Eval Reward: {eval_reward:.2f}")
            
    env.close()
    return alpha_type, reward_scale, seed, eval_metrics

# ==========================================
# 5. EXECUTION & PLOTTING
# ==========================================
if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
        
    target_angle = 90
    reward_scales = [10.0, 0.1]
    alpha_types = ['auto', 'constant']
    num_seeds = 15  
    total_steps = 80000
    eval_freq = 5000
    
    tasks = [(target_angle, a_type, scale, seed, total_steps, eval_freq) 
             for a_type in alpha_types for scale in reward_scales for seed in range(1, num_seeds + 1)]
    
    results = {scale: {a_type: {} for a_type in alpha_types} for scale in reward_scales}
    
    print(f"Running Scaled Reward Experiment | Tasks: {len(tasks)}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(run_experiment, task) for task in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            a_type, scale, seed, metrics = future.result()
            results[scale][a_type][seed] = metrics

    np.save('sac_pendulum_scaled_reward_results.npy', results, allow_pickle=True)
    print("✅ Experiment logs saved successfully to 'sac_scaled_reward_results.npy'")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {'auto': 'blue', 'constant': 'red'}
    labels = {'auto': 'Auto-Tuned α', 'constant': 'Constant α = 0.005'}

    for idx, scale in enumerate(reward_scales):
        ax = axes[idx]
        ax.set_title(f"Reward Scale: {scale}x", fontsize=14, fontweight='bold')
        
        for a_type in alpha_types:
            seeds_data = results[scale][a_type]
            steps = [m[0] for m in seeds_data[1]]
            returns_matrix = np.array([[m[1] for m in metrics] for metrics in seeds_data.values()])
            
            mean_returns = np.mean(returns_matrix, axis=0)
            ci = 1.96 * (np.std(returns_matrix, axis=0) / np.sqrt(num_seeds))
            
            ls = '-' if a_type == 'auto' else '--'
            ax.plot(steps, mean_returns, label=labels[a_type], color=colors[a_type], lw=2.5, linestyle=ls)
            ax.fill_between(steps, mean_returns - ci, mean_returns + ci, color=colors[a_type], alpha=0.15)
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel('Environment Steps', fontsize=12)
        if idx == 0: ax.set_ylabel('True Unscaled Return', fontsize=12)
        ax.legend(loc='lower right')

    plt.suptitle("SAC Adaptability to Reward Scaling (Target θ = 90°)", fontsize=16)
    plt.tight_layout()
    plt.savefig('sac_scaled_reward_experiment.png', dpi=300)
    print("Plot saved as 'sac_scaled_reward_experiment.png'")