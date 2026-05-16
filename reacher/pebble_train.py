# pebble_train.py

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Prefer EGL for headless rendering; users can still override externally.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac import SACAgent
from replay_buffer import ReplayBuffer
from reacher_env import ReacherEnv


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# REWARD MODEL
# ============================================================

class RewardModel(nn.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim=256):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                obs,
                action):

        x = torch.cat([obs, action], dim=-1)

        return self.net(x)


# ============================================================
# PREFERENCE BUFFER
# ============================================================

class PreferenceBuffer:

    def __init__(self,
                 capacity=100000):

        self.capacity = capacity

        self.data = []

    def add(self,
            seg1,
            seg2,
            label):

        if len(self.data) >= self.capacity:

            self.data.pop(0)

        self.data.append(
            (seg1, seg2, label)
        )

    def sample(self,
               batch_size):

        idxs = np.random.randint(
            0,
            len(self.data),
            size=batch_size
        )

        return [self.data[i] for i in idxs]

    def __len__(self):

        return len(self.data)


# ============================================================
# SEGMENT SAMPLING
# ============================================================

def sample_segment(trajectory,
                   segment_length=25):

    start = np.random.randint(
        0,
        len(trajectory) - segment_length
    )

    return trajectory[
        start:start + segment_length
    ]


# ============================================================
# SIMULATED TEACHER
# ============================================================

def teacher_preference(seg1,
                       seg2):

    r1 = sum([x[2] for x in seg1])

    r2 = sum([x[2] for x in seg2])

    return 0 if r1 >= r2 else 1


# ============================================================
# REWARD MODEL LOSS
# ============================================================

def preference_loss(reward_model,
                    batch,
                    device):

    logits = []

    labels = []

    for seg1, seg2, label in batch:

        r1 = 0
        r2 = 0

        for obs, action, reward, next_obs, done in seg1:

            obs = torch.FloatTensor(
                obs
            ).unsqueeze(0).to(device)

            action = torch.FloatTensor(
                action
            ).unsqueeze(0).to(device)

            r1 += reward_model(obs, action)

        for obs, action, reward, next_obs, done in seg2:

            obs = torch.FloatTensor(
                obs
            ).unsqueeze(0).to(device)

            action = torch.FloatTensor(
                action
            ).unsqueeze(0).to(device)

            r2 += reward_model(obs, action)

        logits.append(
            torch.cat([r1, r2], dim=1)
        )

        labels.append(label)

    logits = torch.cat(logits, dim=0)

    labels = torch.LongTensor(
        labels
    ).to(device)

    return F.cross_entropy(
        logits,
        labels
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(agent,
                    reward_type,
                    eval_episodes=20):

    env = ReacherEnv(
        reward_type=reward_type,
        eval_mode=True
    )

    returns = []

    for _ in range(eval_episodes):

        obs = env.reset()

        done = False

        ep_return = 0

        while not done:

            action = agent.select_action(
                obs,
                deterministic=True
            )

            obs, reward, done, info = env.step(action)

            ep_return += reward

            if reward_type == "rc":

                if info.get("timeout", False):

                    ep_return = -1020

                    done = True

        returns.append(ep_return)

    return np.mean(returns)


# ============================================================
# TRAIN
# ============================================================

def train_pebble(reward_type="ra",
                 seed=0):

    print(f"\nStarting PEBBLE Seed {seed} | Reward={reward_type}\n")

    set_seed(seed)

    device = "cpu"

    env = ReacherEnv(
        reward_type=reward_type,
        seed=seed,
        eval_mode=False
    )

    agent = SACAgent(
        env.obs_dim,
        env.action_dim,
        device=device
    )

    replay_buffer = ReplayBuffer(
        env.obs_dim,
        env.action_dim,
        device=device
    )

    reward_model = RewardModel(
        env.obs_dim,
        env.action_dim
    ).to(device)

    reward_optimizer = torch.optim.Adam(
        reward_model.parameters(),
        lr=3e-4
    )

    pref_buffer = PreferenceBuffer()

    total_steps = 500_000

    random_steps = 10_000

    eval_freq = 10_000

    query_freq = 5_000

    reward_updates = 200

    max_queries = 5000

    batch_size = 256

    segment_length = 25

    total_queries = 0

    trajectory = []

    obs = env.reset()

    logs = {
        "steps": [],
        "eval_return": []
    }

    os.makedirs("logs", exist_ok=True)

    for step in range(1, total_steps + 1):

        # ====================================================
        # ACTION
        # ====================================================

        if step < random_steps:

            action = np.random.uniform(
                low=-1,
                high=1,
                size=env.action_dim
            )

        else:

            action = agent.select_action(
                obs,
                deterministic=False
            )

        # ====================================================
        # ENV STEP
        # ====================================================

        next_obs, true_reward, done, info = env.step(action)

        trajectory.append(
            (
                obs,
                action,
                true_reward,
                next_obs,
                done
            )
        )

        # ====================================================
        # LEARNED REWARD
        # ====================================================

        with torch.no_grad():

            r_hat = reward_model(
                torch.FloatTensor(obs).unsqueeze(0),
                torch.FloatTensor(action).unsqueeze(0)
            ).item()

        buffer_done = done

        if reward_type == "rc":

            if info.get("timeout", False):

                buffer_done = True

        replay_buffer.add(
            obs,
            action,
            r_hat,
            next_obs,
            buffer_done
        )

        obs = next_obs

        # ====================================================
        # PREFERENCE QUERIES
        # ====================================================

        if (
            step % query_freq == 0 and
            len(trajectory) > 2 * segment_length and
            total_queries < max_queries
        ):

            for _ in range(20):

                seg1 = sample_segment(
                    trajectory,
                    segment_length
                )

                seg2 = sample_segment(
                    trajectory,
                    segment_length
                )

                label = teacher_preference(
                    seg1,
                    seg2
                )

                pref_buffer.add(
                    seg1,
                    seg2,
                    label
                )

                total_queries += 1

            print(
                f"Queries collected: {total_queries}"
            )

            # ================================================
            # TRAIN REWARD MODEL
            # ================================================

            if len(pref_buffer) >= 64:

                for _ in range(reward_updates):

                    batch = pref_buffer.sample(64)

                    loss = preference_loss(
                        reward_model,
                        batch,
                        device
                    )

                    reward_optimizer.zero_grad()

                    loss.backward()

                    reward_optimizer.step()

                print(
                    f"Reward model updated | Loss={loss.item():.4f}"
                )

        # ====================================================
        # SAC UPDATE
        # ====================================================

        if step >= random_steps:

            agent.update(
                replay_buffer,
                batch_size=batch_size
            )

        # ====================================================
        # RESET
        # ====================================================

        if done:

            obs = env.reset()

            trajectory = []

        # ====================================================
        # EVAL
        # ====================================================

        if step % 1000 == 0:
            progress = (
                100 * step / total_steps
            )

            print(
                f"[Seed {seed}] "
                f"Reward={reward_type} | "
                f"Step={step}/{total_steps} | "
                f"{progress:.1f}% | "
                f"Queries={total_queries} | "
                f"PrefBuffer={len(pref_buffer)}"
            )

        if step % eval_freq == 0:

            eval_return = evaluate_policy(
                agent,
                reward_type
            )

            logs["steps"].append(step)

            logs["eval_return"].append(
                eval_return
            )

            print("\n====================================")
            print(f"PEBBLE EVAL @ STEP {step}")
            print(f"Reward={reward_type}")
            print(f"Eval Return={eval_return:.2f}")
            print(f"Queries={total_queries}")
            print("====================================\n")

            np.save(
                f"logs/pebble_{reward_type}_seed_{seed}.npy",
                logs,
                allow_pickle=True
            )

            torch.save(
                agent.actor.state_dict(),
                f"logs/pebble_{reward_type}_seed_{seed}_actor.pt"
            )

    print(f"\nPEBBLE Training Complete | Seed={seed}\n")
