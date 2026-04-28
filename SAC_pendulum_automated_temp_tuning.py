import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

import concurrent.futures
import multiprocessing as mp

# PREVENT THREAD OVERSUBSCRIPTION
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)


def set_seed(seed):
    """Enforces reproducibility across all libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    def forward(self, obs, deterministic=False):
        x = self.net(obs)
        mu = self.mu_layer(x)
        
        # --- Evaluation Mode (No Exploration) ---
        if deterministic:
            return torch.tanh(mu), None
        # ---------------------------------------------

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
    def __init__(self, obs_dim, action_dim, device, gamma=0.99, tau=0.005, lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        ## Automated Temperature Tuning Setup
        self.target_entropy = -action_dim 
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
        self.critic = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Adam optimizer for both actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @property
    def alpha(self):
        """Dynamically fetch the current temperature value"""
        return self.log_alpha.exp().detach()
    
    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs, deterministic=deterministic)
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

        ## TEMPERATURE UPDATE (Automated Tuning)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()


        # SOFT UPDATE TARGET NETWORKS
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ==========================================
# 3. UTILITIES & ENVIRONMENT WRAPPER
# ==========================================

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=100000, device='cpu'):
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

class TargetPendulumWrapper(gym.Wrapper):
    """Modifies Pendulum-v1 to target a specific angle."""
    def __init__(self, env, target_angle_deg):
        super().__init__(env)
        self.target_theta = np.deg2rad(target_angle_deg)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # obs = [cos(theta), sin(theta), theta_dot]
        cos_th, sin_th, th_dot = obs
        current_theta = np.arctan2(sin_th, cos_th)
        
        # Calculate angular error wrapped to [-pi, pi]
        diff = current_theta - self.target_theta
        angle_err = ((diff + np.pi) % (2 * np.pi)) - np.pi
        
        # Extract torque applied
        torque = np.clip(action[0], -2.0, 2.0)
        
        # Calculate new custom reward
        reward = -(angle_err**2 + 0.1 * th_dot**2 + 0.001 * torque**2)
        
        return obs, reward, terminated, truncated, info

# ==========================================
# 4. OFFLINE EVALUATION FUNCTION
# ==========================================

def evaluate_policy(agent, target_angle, eval_episodes=20, seed=0):
    """Runs isolated episodes purely for evaluation without exploration."""
    eval_env = TargetPendulumWrapper(gym.make("Pendulum-v1", max_episode_steps=1000), target_angle)
    max_action = float(eval_env.action_space.high[0])
    
    avg_reward = 0.
    for ep in range(eval_episodes):
        obs, _ = eval_env.reset(seed=seed + ep) # Unique seed per eval episode
        done = False
        while not done:
            # Deterministic action selection (mean of Gaussian)
            scaled_action = agent.select_action(obs, deterministic=True)
            action = scaled_action * max_action
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            avg_reward += reward
            
    avg_reward /= eval_episodes
    eval_env.close()
    return avg_reward

# ==========================================
# 5. TRAINING LOOP
# ==========================================

def train_target_pendulum(target_angle, seed, num_steps=80000, random_steps=2000, eval_freq=5000):
    set_seed(seed)
    
    base_env = gym.make("Pendulum-v1", max_episode_steps=1000)
    env = TargetPendulumWrapper(base_env, target_angle)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0]) 
    
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    agent = SACAgent(obs_dim, action_dim, device, gamma=0.99)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, device=device)
    
    obs, _ = env.reset(seed=seed)
    
    # Store evaluation metrics: format [(step, mean_reward), ...]
    eval_metrics = []
    
    for step in range(0, num_steps + 1):
        if step <= random_steps:
            action = env.action_space.sample()
            scaled_action = action / max_action
        else:
            scaled_action = agent.select_action(obs, deterministic=False)
            action = scaled_action * max_action 
            
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        replay_buffer.add(obs, scaled_action, reward, next_obs, terminated)
        obs = next_obs
        
        if step > random_steps:
            agent.update(replay_buffer)
            
        if done:
            obs, _ = env.reset()

        # OFFLINE EVALUATION TRIGGER
        if step % eval_freq == 0:
            eval_reward = evaluate_policy(agent, target_angle, eval_episodes=20, seed=seed)
            eval_metrics.append((step, eval_reward))
            print(f"Angle: {target_angle}° | Seed: {seed:2d} | Step: {step:5d} | Eval Avg Return: {eval_reward:.2f}")

    env.close()
    return eval_metrics

def run_seed_task(args):
    """Helper function to unpack arguments for the parallel executor."""
    angle, seed, total_steps, eval_freq = args
    
    # Run the existing training function
    metrics = train_target_pendulum(
        target_angle=angle, 
        seed=seed, 
        num_steps=total_steps, 
        eval_freq=eval_freq
    )
    
    return angle, seed, metrics

if __name__ == "__main__":
    # CRITICAL: Enforce 'spawn' context for PyTorch multiprocessing safety
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    target_angles = [0, -10, 30, -60, 90, -90, 120, -150]
    num_seeds = 15
    eval_freq = 5000 
    total_steps = 80000 
    
    # Nested dictionary to aggregate logs
    experiment_logs = {angle: {} for angle in target_angles}
    
    # 1. Prepare the list of all tasks (8 angles * 15 seeds = 120 tasks)
    tasks = []
    for angle in target_angles:
        for seed in range(1, num_seeds + 1):
            tasks.append((angle, seed, total_steps, eval_freq))
            
    # 2. Determine optimal worker count 
    # Reserve 1 core for the OS. If using GPU, limit this so you don't OOM (Out of Memory).
    # Since Pendulum is tiny, running entirely on CPU often works best for mass parallelism.
    
    # max_workers = min(mp.cpu_count() - 1, len(tasks))
    max_workers = 15
    
    print(f"{'='*50}")
    print(f"Starting Parallel SAC Evaluation")
    print(f"Total Tasks: {len(tasks)} | Max Workers: {max_workers}")
    print(f"{'='*50}\n")
    
    # 3. Execute tasks in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(run_seed_task, task) for task in tasks]
        
        # As each task finishes, collect the results
        for future in concurrent.futures.as_completed(futures):
            try:
                angle, seed, metrics = future.result()
                experiment_logs[angle][seed] = metrics
                print(f"✅ Completed -> Target Angle: {angle:4d}° | Seed: {seed:2d}")
            except Exception as exc:
                print(f"❌ Task generated an exception: {exc}")
                
    print("\nAll parallel training completed successfully.")
    
    # Save the aggregated results to disk for plotting
    np.save('sac_pendulum_automated_temp_tuning_eval_results.npy', experiment_logs, allow_pickle=True)