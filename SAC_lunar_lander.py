import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

torch.set_num_threads(1)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        module.bias.data.fill_(0.0)


class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_bounds=(-20, 2)):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.apply(weight_init)

    def forward(self, obs, deterministic=False):
        x = self.net(obs)
        mu = self.mu_layer(x)

        if deterministic:
            return torch.tanh(mu), None

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = log_std.exp()

        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return y_t, log_prob


class SACAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        automatic_entropy_tuning=True,
        fixed_alpha=None,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.fixed_alpha = fixed_alpha

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.actor = SquashedGaussianActor(obs_dim, action_dim).to(device)
        self.critic = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target = DoubleQCritic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @property
    def alpha(self):
        if not self.automatic_entropy_tuning:
            return torch.tensor(float(self.fixed_alpha), device=self.device)
        return self.log_alpha.exp().detach()

    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (not_done * self.gamma * target_v)

        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action, log_prob = self.actor(obs)
        q1, q2 = self.critic(obs, actor_action)
        q_pi = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=100000, device="cpu"):
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
            torch.as_tensor(self.not_dones[idxs], device=self.device),
        )


@dataclass
class LunarLanderConfig:
    env_id: str = "LunarLander-v3"
    seed: int = 1
    total_steps: int = 500_000
    random_steps: int = 10_000
    batch_size: int = 256
    replay_capacity: int = 100_000
    eval_freq: int = 5_000
    eval_episodes: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    fixed_alpha: float | None = None
    enable_wind: bool = False
    hover_reward: float | None = None
    hover_switch_step: int | None = None
    switched_hover_reward: float = -100.0
    log_dir: str = "logs"


class HoverRewardWrapper(gym.Wrapper):
    """One-shot LunarLander hover-box bonus/penalty for section 2.2(3)."""

    def __init__(self, env, hover_reward=200.0):
        super().__init__(env)
        self.hover_reward = hover_reward
        self.hover_reward_paid = False

    def set_hover_reward(self, hover_reward):
        self.hover_reward = hover_reward

    def reset(self, **kwargs):
        self.hover_reward_paid = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = float(obs[0])
        y_pos = float(obs[1])

        in_hover_box = abs(x_pos) < 0.1 and 0.4 < abs(y_pos) < 0.6
        if in_hover_box and not self.hover_reward_paid:
            reward += self.hover_reward
            self.hover_reward_paid = True
            info = dict(info)
            info["hover_reward"] = self.hover_reward

        return obs, reward, terminated, truncated, info


def make_lunar_lander_env(cfg: LunarLanderConfig, continuous=True):
    env = gym.make(
        cfg.env_id,
        continuous=continuous,
        enable_wind=cfg.enable_wind,
    )
    if cfg.hover_reward is not None:
        env = HoverRewardWrapper(env, hover_reward=cfg.hover_reward)
    return env


def scale_action(action, action_space):
    low = action_space.low
    high = action_space.high
    return low + 0.5 * (action + 1.0) * (high - low)


def evaluate_policy(agent, cfg: LunarLanderConfig, eval_seed):
    eval_env = make_lunar_lander_env(cfg, continuous=True)
    eval_env.action_space.seed(eval_seed)
    returns = []

    for ep in range(cfg.eval_episodes):
        obs, _ = eval_env.reset(seed=eval_seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            normalized_action = agent.select_action(obs, deterministic=True)
            env_action = scale_action(normalized_action, eval_env.action_space)
            obs, reward, terminated, truncated, _ = eval_env.step(env_action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    eval_env.close()
    return float(np.mean(returns))


def experiment_name(cfg: LunarLanderConfig):
    alpha_name = "auto_alpha" if cfg.fixed_alpha is None else f"fixed_alpha_{cfg.fixed_alpha:g}"
    reward_name = "default" if cfg.hover_reward is None else "hover_switch"
    return f"sac_lunar_lander_{reward_name}_{alpha_name}_seed_{cfg.seed}"


def make_csv_logger(cfg: LunarLanderConfig):
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_name(cfg)}.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "step",
        "eval_avg_return",
        "alpha",
        "seed",
        "total_steps",
        "random_steps",
        "replay_capacity",
        "batch_size",
        "eval_freq",
        "eval_episodes",
        "fixed_alpha",
        "hover_reward_initial",
        "hover_switch_step",
        "hover_reward_after_switch",
        "reward_phase",
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    return log_path, log_file, writer


def train_continuous_sac(cfg: LunarLanderConfig):
    set_seed(cfg.seed)
    env = make_lunar_lander_env(cfg, continuous=True)
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = SACAgent(
        obs_dim,
        action_dim,
        device,
        gamma=cfg.gamma,
        tau=cfg.tau,
        lr=cfg.lr,
        automatic_entropy_tuning=cfg.fixed_alpha is None,
        fixed_alpha=cfg.fixed_alpha,
    )
    replay_buffer = ReplayBuffer(obs_dim, action_dim, cfg.replay_capacity, device)

    obs, _ = env.reset(seed=cfg.seed)
    eval_metrics = []
    log_path, log_file, csv_writer = make_csv_logger(cfg)
    print(
        f"Experiment: {experiment_name(cfg)} | seed={cfg.seed} | "
        f"total_steps={cfg.total_steps} | log={log_path}"
    )

    try:
        for step in range(cfg.total_steps + 1):
            if cfg.hover_switch_step is not None and step == cfg.hover_switch_step:
                unwrapped = env
                while hasattr(unwrapped, "env") and not isinstance(unwrapped, HoverRewardWrapper):
                    unwrapped = unwrapped.env
                if isinstance(unwrapped, HoverRewardWrapper):
                    unwrapped.set_hover_reward(cfg.switched_hover_reward)

            if step < cfg.random_steps:
                env_action = env.action_space.sample()
                normalized_action = 2.0 * (env_action - env.action_space.low) / (
                    env.action_space.high - env.action_space.low
                ) - 1.0
            else:
                normalized_action = agent.select_action(obs)
                env_action = scale_action(normalized_action, env.action_space)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            replay_buffer.add(obs, normalized_action, reward, next_obs, terminated)
            obs = next_obs

            if step >= cfg.random_steps:
                agent.update(replay_buffer, cfg.batch_size)

            if done:
                obs, _ = env.reset()

            if step % cfg.eval_freq == 0:
                eval_return = evaluate_policy(agent, cfg, eval_seed=cfg.seed + 10_000)
                alpha = float(agent.alpha.cpu())
                reward_phase = "default"
                if cfg.hover_reward is not None:
                    reward_phase = "after_switch" if (
                        cfg.hover_switch_step is not None and step >= cfg.hover_switch_step
                    ) else "before_switch"

                eval_metrics.append((step, eval_return, alpha, reward_phase))
                csv_writer.writerow(
                    {
                        "step": step,
                        "eval_avg_return": eval_return,
                        "alpha": alpha,
                        "seed": cfg.seed,
                        "total_steps": cfg.total_steps,
                        "random_steps": cfg.random_steps,
                        "replay_capacity": cfg.replay_capacity,
                        "batch_size": cfg.batch_size,
                        "eval_freq": cfg.eval_freq,
                        "eval_episodes": cfg.eval_episodes,
                        "fixed_alpha": "" if cfg.fixed_alpha is None else cfg.fixed_alpha,
                        "hover_reward_initial": "" if cfg.hover_reward is None else cfg.hover_reward,
                        "hover_switch_step": "" if cfg.hover_switch_step is None else cfg.hover_switch_step,
                        "hover_reward_after_switch": cfg.switched_hover_reward,
                        "reward_phase": reward_phase,
                    }
                )
                log_file.flush()
                print(
                    f"Seed: {cfg.seed:3d} | Step: {step:7d} | Eval Avg Return: {eval_return:8.2f} | "
                    f"Alpha: {alpha:.4f}"
                )
    finally:
        log_file.close()

    env.close()
    return eval_metrics


def print_env_summary(cfg: LunarLanderConfig):
    continuous_env = make_lunar_lander_env(cfg, continuous=True)
    discrete_env = make_lunar_lander_env(cfg, continuous=False)

    print("Continuous LunarLander setup for 2.2(1)")
    print(f"  Env ID: {cfg.env_id}")
    print(f"  Observation space: {continuous_env.observation_space}")
    print(f"  Action space: {continuous_env.action_space}")
    print(f"  Random exploration steps: {cfg.random_steps}")
    print(f"  Gamma: {cfg.gamma}")
    print(f"  Alpha: {'auto' if cfg.fixed_alpha is None else cfg.fixed_alpha}")
    print()
    print("Discrete LunarLander setup for 2.2(4)")
    print(f"  Env ID: {cfg.env_id}")
    print(f"  Observation space: {discrete_env.observation_space}")
    print(f"  Action space: {discrete_env.action_space}")

    continuous_env.close()
    discrete_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="SAC setup for PA3 section 2.2 LunarLander.")
    parser.add_argument("--summary", action="store_true", help="Print environment details and exit.")
    parser.add_argument("--train", action="store_true", help="Train continuous-action SAC.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--random-steps", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--fixed-alpha", type=float, default=None)
    parser.add_argument("--hover-reward", type=float, default=None)
    parser.add_argument("--hover-switch-step", type=int, default=None)
    parser.add_argument("--switched-hover-reward", type=float, default=-100.0)
    parser.add_argument("--enable-wind", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = LunarLanderConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        random_steps=args.random_steps,
        replay_capacity=args.replay_capacity,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        fixed_alpha=args.fixed_alpha,
        enable_wind=args.enable_wind,
        hover_reward=args.hover_reward,
        hover_switch_step=args.hover_switch_step,
        switched_hover_reward=args.switched_hover_reward,
        log_dir=args.log_dir,
    )

    if args.summary or not args.train:
        print_env_summary(config)
    if args.train:
        metrics = train_continuous_sac(config)
        np.save(Path(config.log_dir) / f"{experiment_name(config)}_eval_results.npy", metrics, allow_pickle=True)
