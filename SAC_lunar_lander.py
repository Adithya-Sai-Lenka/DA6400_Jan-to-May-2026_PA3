import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from SAC_pendulum_automated_temp_tuning import ReplayBuffer, SACAgent, set_seed


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
    print(f"Writing evaluation log to: {log_path}")

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
                    f"Step: {step:7d} | Eval Avg Return: {eval_return:8.2f} | "
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
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
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
        replay_capacity=args.replay_capacity,
        eval_freq=args.eval_freq,
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
        np.save(f"{experiment_name(config)}_eval_results.npy", metrics, allow_pickle=True)
