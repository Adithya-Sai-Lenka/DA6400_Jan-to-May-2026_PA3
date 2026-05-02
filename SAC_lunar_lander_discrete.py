import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)


@dataclass
class DiscreteLunarConfig:
    env_id: str = "LunarLander-v3"
    algo: str = "discrete-sac"
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
    hidden_dim: int = 256
    grad_clip_norm: float | None = None
    target_entropy_ratio: float = 0.5
    min_alpha: float = 1e-4
    max_alpha: float = 1.0
    dqn_target_update_freq: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    enable_wind: bool = False
    log_dir: str = "logs"


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


class DiscreteReplayBuffer:
    def __init__(self, obs_dim, capacity, device):
        self.obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
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
        size = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, size, size=batch_size)
        return (
            torch.as_tensor(self.obses[idxs], device=self.device),
            torch.as_tensor(self.actions[idxs], device=self.device),
            torch.as_tensor(self.rewards[idxs], device=self.device),
            torch.as_tensor(self.next_obses[idxs], device=self.device),
            torch.as_tensor(self.not_dones[idxs], device=self.device),
        )


class MLP(nn.Module):
    def __init__(self, obs_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, obs):
        return self.net(obs)


class DiscreteDoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = MLP(obs_dim, action_dim, hidden_dim)
        self.q2 = MLP(obs_dim, action_dim, hidden_dim)

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)


class DiscreteSACAgent:
    def __init__(self, obs_dim, action_dim, device, cfg: DiscreteLunarConfig):
        self.device = device
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.min_log_alpha = np.log(cfg.min_alpha)
        self.max_log_alpha = np.log(cfg.max_alpha)
        self.target_entropy = cfg.target_entropy_ratio * np.log(action_dim)

        self.actor = MLP(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic = DiscreteDoubleQCritic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target = DiscreteDoubleQCritic(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def policy(self, obs):
        logits = self.actor(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy(obs)
            if deterministic:
                return int(torch.argmax(probs, dim=-1).item())
            return int(torch.distributions.Categorical(probs=probs).sample().item())

    def update(self, replay_buffer, batch_size):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_probs, next_log_probs = self.policy(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs)
            target_q = torch.min(target_q1, target_q2)
            target_v = (next_probs * (target_q - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            backup = reward + not_done * self.gamma * target_v

        current_q1, current_q2 = self.critic(obs)
        q1 = current_q1.gather(1, action)
        q2 = current_q2.gather(1, action)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        probs, log_probs = self.policy(obs)
        q1_pi, q2_pi = self.critic(obs)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (probs * (self.alpha * log_probs - q_pi)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


class DQNAgent:
    def __init__(self, obs_dim, action_dim, device, cfg: DiscreteLunarConfig):
        self.device = device
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.grad_clip_norm = cfg.grad_clip_norm
        self.q = MLP(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.q_target = MLP(obs_dim, action_dim, cfg.hidden_dim).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    def select_action(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(torch.argmax(self.q(obs), dim=-1).item())

    def update(self, replay_buffer, batch_size):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)
        q = self.q(obs).gather(1, action)
        with torch.no_grad():
            next_q = self.q_target(next_obs).max(dim=-1, keepdim=True).values
            target = reward + not_done * self.gamma * next_q
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip_norm)
        self.optimizer.step()

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())


def make_env(cfg):
    return gym.make(cfg.env_id, continuous=False, enable_wind=cfg.enable_wind)


def epsilon_at_step(step, cfg):
    frac = min(1.0, step / cfg.epsilon_decay_steps)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def evaluate_agent(agent, cfg, eval_seed):
    env = make_env(cfg)
    env.action_space.seed(eval_seed)
    returns = []
    for ep in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=eval_seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            if cfg.algo == "dqn":
                action = agent.select_action(obs, epsilon=0.0)
            else:
                action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


def experiment_name(cfg):
    return f"lunar_lander_discrete_{cfg.algo}_seed_{cfg.seed}"


def make_csv_logger(cfg):
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_name(cfg)}.csv"
    log_file = log_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "step",
        "eval_avg_return",
        "alpha",
        "epsilon",
        "seed",
        "algo",
        "total_steps",
        "random_steps",
        "replay_capacity",
        "batch_size",
        "eval_freq",
        "eval_episodes",
        "lr",
        "grad_clip_norm",
        "target_entropy_ratio",
        "min_alpha",
        "max_alpha",
    ]
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    return log_path, log_file, writer


def train(cfg):
    set_seed(cfg.seed)
    env = make_env(cfg)
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.algo == "dqn":
        agent = DQNAgent(obs_dim, action_dim, device, cfg)
    else:
        agent = DiscreteSACAgent(obs_dim, action_dim, device, cfg)

    replay_buffer = DiscreteReplayBuffer(obs_dim, cfg.replay_capacity, device)
    obs, _ = env.reset(seed=cfg.seed)
    metrics = []
    log_path, log_file, writer = make_csv_logger(cfg)
    print(
        f"Experiment: {experiment_name(cfg)} | seed={cfg.seed} | algo={cfg.algo} | "
        f"total_steps={cfg.total_steps} | log={log_path}"
    )

    try:
        for step in range(cfg.total_steps + 1):
            epsilon = epsilon_at_step(step, cfg) if cfg.algo == "dqn" else 0.0
            if step < cfg.random_steps:
                action = env.action_space.sample()
            elif cfg.algo == "dqn":
                action = agent.select_action(obs, epsilon=epsilon)
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs

            if step >= cfg.random_steps:
                agent.update(replay_buffer, cfg.batch_size)
                if cfg.algo == "dqn" and step % cfg.dqn_target_update_freq == 0:
                    agent.update_target()

            if terminated or truncated:
                obs, _ = env.reset()

            if step % cfg.eval_freq == 0:
                eval_return = evaluate_agent(agent, cfg, eval_seed=cfg.seed + 10_000)
                alpha = float(agent.alpha.cpu()) if cfg.algo == "discrete-sac" else ""
                metrics.append((step, eval_return, alpha, epsilon))
                writer.writerow(
                    {
                        "step": step,
                        "eval_avg_return": eval_return,
                        "alpha": alpha,
                        "epsilon": epsilon if cfg.algo == "dqn" else "",
                        "seed": cfg.seed,
                        "algo": cfg.algo,
                        "total_steps": cfg.total_steps,
                        "random_steps": cfg.random_steps,
                        "replay_capacity": cfg.replay_capacity,
                        "batch_size": cfg.batch_size,
                        "eval_freq": cfg.eval_freq,
                        "eval_episodes": cfg.eval_episodes,
                        "lr": cfg.lr,
                        "grad_clip_norm": "" if cfg.grad_clip_norm is None else cfg.grad_clip_norm,
                        "target_entropy_ratio": cfg.target_entropy_ratio,
                        "min_alpha": cfg.min_alpha,
                        "max_alpha": cfg.max_alpha,
                    }
                )
                log_file.flush()
                print(
                    f"Seed: {cfg.seed:3d} | Algo: {cfg.algo:12s} | Step: {step:7d} | "
                    f"Eval Avg Return: {eval_return:8.2f}"
                )
    finally:
        log_file.close()
        env.close()

    np.save(Path(cfg.log_dir) / f"{experiment_name(cfg)}_eval_results.npy", metrics, allow_pickle=True)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="PA3 2.2 Q4: discrete LunarLander SAC/DQN.")
    parser.add_argument("--algo", choices=["discrete-sac", "dqn", "both"], default="discrete-sac")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--random-steps", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dqn-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--target-entropy-ratio", type=float, default=0.5)
    parser.add_argument("--min-alpha", type=float, default=1e-4)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--enable-wind", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    algos = ["discrete-sac", "dqn"] if args.algo == "both" else [args.algo]
    for algo in algos:
        config = DiscreteLunarConfig(
            algo=algo,
            seed=args.seed,
            total_steps=args.total_steps,
            random_steps=args.random_steps,
            replay_capacity=args.replay_capacity,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            lr=args.dqn_lr if algo == "dqn" else args.lr,
            grad_clip_norm=args.grad_clip_norm if algo == "dqn" else None,
            target_entropy_ratio=args.target_entropy_ratio,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha,
            enable_wind=args.enable_wind,
            log_dir=args.log_dir,
        )
        train(config)
