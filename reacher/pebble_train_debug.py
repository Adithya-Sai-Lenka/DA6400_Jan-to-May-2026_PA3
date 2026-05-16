# ============================================================
# pebble_train_debug.py
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "osmesa"

import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sac import SACAgent
from replay_buffer import ReplayBuffer
from reacher_env import ReacherEnv


# ============================================================
# CONFIG
# ============================================================

TOTAL_STEPS = 100_000

RANDOM_STEPS = 10_000

QUERY_FREQ = 5_000

QUERY_START = 20_000

EVAL_FREQ = 5_000

REWARD_UPDATES = 50

SEGMENT_LENGTH = 50

MAX_QUERIES = 5000

DEVICE = "cpu"


# ============================================================
# REPLAY SIZE HELPER
# ============================================================

def replay_size(replay_buffer):

    return (
        replay_buffer.capacity
        if replay_buffer.full
        else replay_buffer.idx
    )


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

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=256
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                obs_dim + action_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        obs,
        action
    ):

        x = torch.cat(
            [obs, action],
            dim=-1
        )

        return self.net(x)


# ============================================================
# PREFERENCE BUFFER
# ============================================================

class PreferenceBuffer:

    def __init__(
        self,
        capacity=100000
    ):

        self.capacity = capacity

        self.data = []

    def add(
        self,
        seg1,
        seg2,
        label
    ):

        if len(self.data) >= self.capacity:

            self.data.pop(0)

        self.data.append(
            (seg1, seg2, label)
        )

    def sample(
        self,
        batch_size
    ):

        idxs = np.random.randint(
            0,
            len(self.data),
            size=batch_size
        )

        return [
            self.data[i]
            for i in idxs
        ]

    def __len__(self):

        return len(self.data)


# ============================================================
# SEGMENT SAMPLING
# ============================================================

def sample_segment(
    trajectory,
    segment_length
):

    start = np.random.randint(
        0,
        len(trajectory) - segment_length
    )

    return trajectory[
        start:start + segment_length
    ]


# ============================================================
# TEACHER PREFERENCE
# ============================================================

def teacher_preference(seg1, seg2):

    r1 = sum([
        x[2]
        for x in seg1
    ])

    r2 = sum([
        x[2]
        for x in seg2
    ])

    if np.random.rand() < 0.001:
        print(f"Teacher Debug | r1={r1:.3f} | r2={r2:.3f}")
    # IMPORTANT:
    # Lower cumulative cost / less negative reward is better.

    return 0 if r1 <= r2 else 1

# ============================================================
# PREFERENCE LOSS
# ============================================================

def preference_loss(
    reward_model,
    batch,
    device
):

    logits = []

    labels = []

    for (
        seg1,
        seg2,
        label
    ) in batch:

        r1 = 0
        r2 = 0

        for (
            obs,
            action,
            reward,
            next_obs,
            done
        ) in seg1:

            obs = torch.FloatTensor(
                obs
            ).unsqueeze(0).to(device)

            action = torch.FloatTensor(
                action
            ).unsqueeze(0).to(device)

            r1 += reward_model(
                obs,
                action
            )

        for (
            obs,
            action,
            reward,
            next_obs,
            done
        ) in seg2:

            obs = torch.FloatTensor(
                obs
            ).unsqueeze(0).to(device)

            action = torch.FloatTensor(
                action
            ).unsqueeze(0).to(device)

            r2 += reward_model(
                obs,
                action
            )

        logits.append(
            torch.cat(
                [r1, r2],
                dim=1
            )
        )

        labels.append(label)

    logits = torch.cat(
        logits,
        dim=0
    )

    labels = torch.LongTensor(
        labels
    ).to(device)

    return F.cross_entropy(
        logits,
        labels
    )


# ============================================================
# REPLAY RELABEL
# ============================================================

def relabel_replay_buffer(
    replay_buffer,
    reward_model,
    device
):

    print(
        "Relabeling replay buffer..."
    )

    size = replay_size(
        replay_buffer
    )

    with torch.no_grad():

        for idx in range(size):

            obs_t = torch.FloatTensor(
                replay_buffer.obses[idx]
            ).unsqueeze(0).to(device)

            act_t = torch.FloatTensor(
                replay_buffer.actions[idx]
            ).unsqueeze(0).to(device)

            new_reward = reward_model(
                obs_t,
                act_t
            ).item()

            replay_buffer.rewards[idx] = (
                new_reward
            )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(
    agent,
    reward_type,
    eval_episodes=10
):

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

            obs, reward, done, info = env.step(
                action
            )

            ep_return += reward

            if reward_type == "rc":

                if info.get(
                    "timeout",
                    False
                ):

                    ep_return = -1020

                    done = True

        returns.append(ep_return)

    return np.mean(returns)


# ============================================================
# TRAIN
# ============================================================

def train_pebble(
    reward_type="ra",
    seed=0
):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    print(
        f"\nStarting "
        f"PEBBLE | "
        f"Reward={reward_type} | "
        f"Seed={seed}\n"
    )

    set_seed(seed)

    env = ReacherEnv(
        reward_type=reward_type,
        seed=seed,
        eval_mode=False
    )

    agent = SACAgent(
        env.obs_dim,
        env.action_dim,
        device=DEVICE
    )

    replay_buffer = ReplayBuffer(
        env.obs_dim,
        env.action_dim,
        device=DEVICE
    )

    reward_model = RewardModel(
        env.obs_dim,
        env.action_dim
    ).to(DEVICE)

    reward_optimizer = torch.optim.Adam(
        reward_model.parameters(),
        lr=3e-4
    )

    pref_buffer = PreferenceBuffer()

    trajectory_dataset = []

    logs = {

        "steps": [],
        "eval_return": [],
        "queries": [],
        "reward_loss": [],
        "reward_mean": [],
        "reward_std": []
    }

    total_queries = 0

    trajectory = []

    obs = env.reset()

    loss = torch.tensor(0.0)

    # ========================================================
    # MAIN LOOP
    # ========================================================

    for step in range(
        1,
        TOTAL_STEPS + 1
    ):

        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        if step < RANDOM_STEPS:

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

        # ----------------------------------------------------
        # ENV STEP
        # ----------------------------------------------------

        next_obs, true_reward, done, info = env.step(
            action
        )

        trajectory.append(
            (
                obs,
                action,
                true_reward,
                next_obs,
                done
            )
        )

        # ----------------------------------------------------
        # LEARNED REWARD
        # ----------------------------------------------------

        with torch.no_grad():

            r_hat = reward_model(
                torch.FloatTensor(obs).unsqueeze(0),
                torch.FloatTensor(action).unsqueeze(0)
            ).item()

        replay_buffer.add(
            obs,
            action,
            r_hat,
            next_obs,
            done
        )

        obs = next_obs

        # ----------------------------------------------------
        # PREFERENCE QUERIES
        # ----------------------------------------------------

        if (
            step >= QUERY_START and
            step % QUERY_FREQ == 0 and
            len(trajectory_dataset) >= 2 and
            total_queries < MAX_QUERIES
        ):

            for _ in range(20):

                traj1 = random.choice(
                    trajectory_dataset
                )

                traj2 = random.choice(
                    trajectory_dataset
                )

                seg1 = sample_segment(
                    traj1,
                    SEGMENT_LENGTH
                )

                seg2 = sample_segment(
                    traj2,
                    SEGMENT_LENGTH
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
                f"\nQueries collected: "
                f"{total_queries}"
            )

            # ------------------------------------------------
            # REWARD MODEL UPDATE
            # ------------------------------------------------

            if len(pref_buffer) >= 64:

                for _ in range(
                    REWARD_UPDATES
                ):

                    batch = pref_buffer.sample(
                        64
                    )

                    loss = preference_loss(
                        reward_model,
                        batch,
                        DEVICE
                    )

                    reward_optimizer.zero_grad()

                    loss.backward()

                    reward_optimizer.step()

                print(
                    f"Reward model updated | "
                    f"Loss={loss.item():.4f}"
                )

                # --------------------------------------------
                # CRITICAL FIX
                # --------------------------------------------

                relabel_replay_buffer(
                    replay_buffer,
                    reward_model,
                    DEVICE
                )

        # ----------------------------------------------------
        # SAC UPDATE
        # ----------------------------------------------------

        if step >= RANDOM_STEPS:

            agent.update(
                replay_buffer,
                batch_size=256
            )

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        if done:

            if len(trajectory) > SEGMENT_LENGTH:

                trajectory_dataset.append(
                    trajectory.copy()
                )

            obs = env.reset()

            trajectory = []

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        if step % 1000 == 0:

            print(
                f"[Seed {seed}] "
                f"Step={step}/{TOTAL_STEPS} | "
                f"Queries={total_queries} | "
                f"PrefBuffer={len(pref_buffer)}"
            )

        # ----------------------------------------------------
        # EVAL
        # ----------------------------------------------------

        if step % EVAL_FREQ == 0:

            eval_return = evaluate_policy(
                agent,
                reward_type
            )

            size = replay_size(
                replay_buffer
            )

            idxs = np.random.randint(
                0,
                size,
                size=min(2048, size)
            )

            obs_batch = torch.FloatTensor(
                replay_buffer.obses[idxs]
            )

            act_batch = torch.FloatTensor(
                replay_buffer.actions[idxs]
            )

            with torch.no_grad():

                pred_rewards = reward_model(
                    obs_batch,
                    act_batch
                ).cpu().numpy()

            reward_mean = pred_rewards.mean()

            reward_std = pred_rewards.std()

            logs["steps"].append(step)

            logs["eval_return"].append(
                eval_return
            )

            logs["queries"].append(
                total_queries
            )

            logs["reward_loss"].append(
                float(loss.item())
            )

            logs["reward_mean"].append(
                reward_mean
            )

            logs["reward_std"].append(
                reward_std
            )

            print("\n================================")

            print(
                f"PEBBLE EVAL @ {step}"
            )

            print(
                f"Eval Return = "
                f"{eval_return:.2f}"
            )

            print(
                f"Queries = "
                f"{total_queries}"
            )

            print(
                f"Reward Loss = "
                f"{loss.item():.4f}"
            )

            print(
                f"Reward Mean = "
                f"{reward_mean:.6f}"
            )

            print(
                f"Reward Std = "
                f"{reward_std:.6f}"
            )

            print("================================\n")

            # ------------------------------------------------
            # SAVE EVERYTHING
            # ------------------------------------------------

            np.save(
                f"logs/pebble_{reward_type}_seed_{seed}.npy",
                logs,
                allow_pickle=True
            )

            torch.save(
                agent.actor.state_dict(),
                f"logs/pebble_{reward_type}_seed_{seed}_actor.pt"
            )

            torch.save(
                reward_model.state_dict(),
                f"logs/pebble_{reward_type}_seed_{seed}_reward_model.pt"
            )

            with open(
                f"logs/{reward_type}_seed_{seed}_pref.pkl",
                "wb"
            ) as f:

                pickle.dump(
                    pref_buffer.data,
                    f
                )

            with open(
                f"logs/{reward_type}_seed_{seed}_replay.pkl",
                "wb"
            ) as f:

                pickle.dump(
                    replay_buffer,
                    f
                )

    print(
        f"\nFinished "
        f"PEBBLE | "
        f"Reward={reward_type} | "
        f"Seed={seed}\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_pebble(
        reward_type="ra",
        seed=0
    )