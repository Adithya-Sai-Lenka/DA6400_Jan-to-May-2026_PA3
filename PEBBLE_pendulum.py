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
from tqdm import tqdm

def init_worker(tqdm_lock):
    tqdm.set_lock(tqdm_lock)

from collections import deque

# Restrict internal threading to allow efficient multi-process parallelization
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

# ==========================================
# 1. CORE NETWORKS (SAC & REWARD MODEL)
# ==========================================

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
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
        if deterministic: return torch.tanh(mu), None
        log_std = torch.clamp(self.log_std_layer(x), *self.log_std_bounds)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(-1, keepdim=True)

class RewardModel(nn.Module):
    """Learns the reward function from human/teacher preferences."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))
    
# ==========================================
# 2. BUFFERS & TEACHER
# ==========================================

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=100000, device='cpu'):
        self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.capacity, self.idx, self.full, self.device = capacity, 0, False, device

    def add(self, obs, action, next_obs, done):
        self.obses[self.idx] = obs
        self.actions[self.idx] = action
        self.next_obses[self.idx] = next_obs
        self.not_dones[self.idx] = not done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        return (torch.as_tensor(self.obses[idxs], device=self.device),
                torch.as_tensor(self.actions[idxs], device=self.device),
                torch.as_tensor(self.next_obses[idxs], device=self.device),
                torch.as_tensor(self.not_dones[idxs], device=self.device))

class PreferenceBuffer:
    """Stores human/teacher preferences over trajectory pairs."""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, seg1_obs, seg1_act, seg2_obs, seg2_act, label):
        self.buffer.append((seg1_obs, seg1_act, seg2_obs, seg2_act, label))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s1_obs, s1_act, s2_obs, s2_act, labels = zip(*batch)
        return (np.array(s1_obs), np.array(s1_act), 
                np.array(s2_obs), np.array(s2_act), 
                np.array(labels, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class SimulatedTeacher:
    """Provides ground-truth preferences to train the Reward Model."""
    def __init__(self, target_angle_deg):
        self.target_theta = np.deg2rad(target_angle_deg)

    def evaluate_segment(self, obs_seq, act_seq):
        """Calculates the true cumulative reward for a sequence."""
        total_reward = 0
        for obs, act in zip(obs_seq, act_seq):
            cos_th, sin_th, th_dot = obs
            current_theta = np.arctan2(sin_th, cos_th)
            diff = current_theta - self.target_theta
            angle_err = ((diff + np.pi) % (2 * np.pi)) - np.pi
            torque = np.clip(act[0], -2.0, 2.0)
            reward = -(angle_err**2 + 0.1 * th_dot**2 + 0.001 * torque**2)
            total_reward += reward
        return total_reward

    def query(self, seg1, seg2):
        """Returns 1 if seg2 is strictly better than seg1, else 0."""
        r1 = self.evaluate_segment(seg1['obs'], seg1['acts'])
        r2 = self.evaluate_segment(seg2['obs'], seg2['acts'])
        return 1.0 if r2 > r1 else 0.0

# ==========================================
# 3. PEBBLE AGENT
# ==========================================

class PEBBLEAgent:
    def __init__(self, obs_dim, action_dim, device, gamma=0.99, tau=0.005, lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # SAC Components
        self.actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
        self.critic = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Auto Temperature Tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        
        # Reward Model (PEBBLE specific)
        self.reward_model = RewardModel(obs_dim, action_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3, weight_decay=1e-4)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def update_reward_model(self, pref_buffer, batch_size=64, epochs=5):
        """Trains the reward model using the Bradley-Terry model."""
        if len(pref_buffer) == 0: return

        for _ in range(epochs):
            s1_o, s1_a, s2_o, s2_a, labels = pref_buffer.sample(batch_size)
            
            s1_o = torch.FloatTensor(s1_o).to(self.device)
            s1_a = torch.FloatTensor(s1_a).to(self.device)
            s2_o = torch.FloatTensor(s2_o).to(self.device)
            s2_a = torch.FloatTensor(s2_a).to(self.device)
            labels = torch.FloatTensor(labels).to(self.device).unsqueeze(-1)
            
            # Predict rewards for each step in the segments
            r1 = self.reward_model(s1_o, s1_a).squeeze(-1) # Shape: [batch, seg_len]
            r2 = self.reward_model(s2_o, s2_a).squeeze(-1)
            
            # Sum rewards over the segment length
            r1_sum = r1.sum(dim=1, keepdim=True)
            r2_sum = r2.sum(dim=1, keepdim=True)
            
            # Bradley-Terry BCE Loss using logits
            # If label=1 (seg2 preferred), we want r2_sum > r1_sum
            logits = r2_sum - r1_sum
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()

    def update_sac(self, replay_buffer, batch_size=256):
        """Standard SAC update, but dynamically uses the LEARNED reward model."""
        obs, action, next_obs, not_done = replay_buffer.sample(batch_size)

        # 1. Compute Reward internally using the current Reward Model
        with torch.no_grad():
            learned_reward = self.reward_model(obs, action)
            # Normalize or clip reward to maintain SAC stability
            learned_reward = torch.clamp(learned_reward, -10, 10)

        # 2. Critic Update
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = learned_reward + (not_done * self.gamma * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Actor Update
        actor_action, log_prob = self.actor(obs)
        Q1, Q2 = self.critic(obs, actor_action)
        actor_loss = (self.alpha * log_prob - torch.min(Q1, Q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Temperature Update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 5. Soft Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ==========================================
# 4. EXPERIMENT LOOP
# ==========================================
from SAC_pendulum_automated_temp_tuning import TargetPendulumWrapper

def evaluate_true_return(agent, target_angle, eval_episodes=5):
    """Evaluates the agent using the exact same environment wrapper as base SAC."""
    base_env = gym.make("Pendulum-v1", max_episode_steps=1000)
    eval_env = TargetPendulumWrapper(base_env, target_angle)
    
    max_action = float(eval_env.action_space.high[0])
    avg_reward = 0.
    
    for _ in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True) * max_action
            
            # The wrapper naturally returns the ground-truth target reward!
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            avg_reward += reward
            done = terminated or truncated
            
    eval_env.close()
    return avg_reward / eval_episodes

def run_pebble_instance(args):
    target_angle, budget, seed, total_steps, task_id, max_workers = args
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cpu')
    env = gym.make("Pendulum-v1", max_episode_steps=1000)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PEBBLEAgent(obs_dim, action_dim, device) # Assumes PEBBLEAgent from Part 1
    teacher = SimulatedTeacher(target_angle)
    replay_buffer = ReplayBuffer(obs_dim, action_dim)
    pref_buffer = PreferenceBuffer(capacity=budget)
    
    obs, _ = env.reset(seed=seed)
    recent_segments = deque(maxlen=20)
    ep_obs, ep_acts = [], []
    queries_made = 0
    logs = []

    worker_pos = (task_id % max_workers) + 1 
    
    step_iterator = tqdm(range(0, total_steps + 1), 
                         desc=f"Ang:{target_angle:4d} | Bud:{budget:4d} | S:{seed:2d}",
                         position=worker_pos,
                         leave=False,
                         ncols=100)

    queries_per_interval = max(1, budget // (total_steps // 500))

    for step in step_iterator:
        # Exploration vs Exploitation
        if step <= 2000: action = env.action_space.sample() / 2.0
        else: action = agent.select_action(obs)
        
        next_obs, _, term, trunc, _ = env.step(action * 2.0)
        replay_buffer.add(obs, action, next_obs, term or trunc)
        
        ep_obs.append(obs); ep_acts.append(action)
        obs = next_obs
        
        if (term or trunc) or len(ep_obs) >= 50:
            if len(ep_obs) >= 50:
                recent_segments.append({'obs': ep_obs[-50:], 'acts': ep_acts[-50:]})
            if term or trunc: obs, _ = env.reset(); ep_obs, ep_acts = [], []

        # Query Teacher
        if step % 500 == 0 and len(recent_segments) >= 2 and queries_made < budget:
            queries_this_round = 0
            
            # Ask multiple questions at once to utilize the budget
            while queries_this_round < queries_per_interval and queries_made < budget:
                s1, s2 = random.sample(list(recent_segments), 2)
                label = teacher.query(s1, s2)
                pref_buffer.add(s1['obs'], s1['acts'], s2['obs'], s2['acts'], label)
                
                queries_made += 1
                queries_this_round += 1
                
            # Train the reward model with the newly enriched dataset
            agent.update_reward_model(pref_buffer)

        if step > 2000: agent.update_sac(replay_buffer)
            
        if step % 5000 == 0:
            # Eval using ground truth wrapper
            eval_score = evaluate_true_return(agent, target_angle)
            logs.append((step, eval_score))

    env.close()
    return target_angle, budget, seed, logs

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    target_angles = [0, -60, 90, 120, -150]
    budgets = [100, 200, 500, 1000]
    num_seeds = 15
    total_steps = 80000

    max_workers = 15
    print("\n" * (max_workers + 2))

    tdqm_lock = mp.RLock()
    
    combinations = [(ang, bud, seed) for ang in target_angles for bud in budgets for seed in range(num_seeds)]
    tasks = [(ang, bud, seed, total_steps, task_id, max_workers) for task_id, (ang, bud, seed) in enumerate(combinations)]
    
    experiment_results = {ang: {bud: {} for bud in budgets} for ang in target_angles}


    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(tdqm_lock,)) as executor:
        futures = [executor.submit(run_pebble_instance, t) for t in tasks]
        overall_progress = tqdm(concurrent.futures.as_completed(futures), 
                                total=len(futures), 
                                desc="Total PEBBLE Runs", 
                                position=0, 
                                leave=True, ncols=100)
                                
        for f in overall_progress:
            ang, bud, seed, logs = f.result()
            experiment_results[ang][bud][seed] = logs

    np.save('pebble_results_combined.npy', experiment_results, allow_pickle=True)