# ============================================================
# inspect_pebble_training.py
# ============================================================

"""
Run this separately while training is ongoing.

Purpose:
- inspect reward model behavior
- inspect replay relabeling
- inspect reward prediction scale
- inspect preference diversity

Does NOT interrupt training.
"""

import os

os.environ["MUJOCO_GL"] = "osmesa"

import argparse
import pickle
import numpy as np
import torch

from reward_model import RewardModel
from replay_buffer import ReplayBuffer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    required=True
)

parser.add_argument(
    "--reward",
    type=str,
    required=True
)

args = parser.parse_args()


reward_type = args.reward
seed = args.seed


# ============================================================
# paths
# ============================================================

reward_model_path = (
    f"pebble_{reward_type}_seed_{seed}_reward_model.pt"
)

replay_buffer_path = (
    f"logs/{reward_type}_seed_{seed}_replay.pkl"
)

pref_buffer_path = (
    f"logs/{reward_type}_seed_{seed}_pref.pkl"
)


# ============================================================
# load reward model
# ============================================================

device = "cpu"

from reacher_env import ReacherEnv

env = ReacherEnv(
    reward_type=reward_type,
    seed=seed,
    eval_mode=True
)

reward_model = RewardModel(
    env.obs_dim,
    env.action_dim
).to(device)

reward_model.load_state_dict(
    torch.load(
        reward_model_path,
        map_location=device
    )
)

reward_model.eval()


print("\nLoaded reward model.")


# ============================================================
# load replay buffer
# ============================================================

with open(replay_buffer_path, "rb") as f:

    replay_buffer = pickle.load(f)

print("Loaded replay buffer.")


# ============================================================
# reward prediction stats
# ============================================================

idxs = np.random.randint(
    0,
    replay_buffer.size,
    size=2048
)

obs = torch.FloatTensor(
    replay_buffer.obs[idxs]
).to(device)

actions = torch.FloatTensor(
    replay_buffer.actions[idxs]
).to(device)

with torch.no_grad():

    pred_rewards = reward_model(
        obs,
        actions
    ).cpu().numpy()


print("\n===== REWARD PREDICTION STATS =====")

print(
    f"Mean : {pred_rewards.mean():.6f}"
)

print(
    f"Std  : {pred_rewards.std():.6f}"
)

print(
    f"Min  : {pred_rewards.min():.6f}"
)

print(
    f"Max  : {pred_rewards.max():.6f}"
)


# ============================================================
# replay relabel verification
# ============================================================

idx = np.random.randint(
    0,
    replay_buffer.size
)

obs = replay_buffer.obs[idx]
action = replay_buffer.actions[idx]

stored_reward = replay_buffer.rewards[idx]

obs_t = torch.FloatTensor(
    obs
).unsqueeze(0).to(device)

action_t = torch.FloatTensor(
    action
).unsqueeze(0).to(device)

with torch.no_grad():

    predicted_reward = reward_model(
        obs_t,
        action_t
    ).item()


print("\n===== RELABEL CHECK =====")

print(
    f"Stored replay reward : "
    f"{stored_reward}"
)

print(
    f"Reward model output  : "
    f"{predicted_reward}"
)

print(
    f"Absolute difference  : "
    f"{abs(stored_reward - predicted_reward):.6f}"
)


# ============================================================
# preference diversity
# ============================================================

with open(pref_buffer_path, "rb") as f:

    pref_buffer = pickle.load(f)

segment_returns = []

for item in pref_buffer:

    r1 = np.sum(item["r1"])

    r2 = np.sum(item["r2"])

    segment_returns.append(r1)
    segment_returns.append(r2)

segment_returns = np.array(segment_returns)

print("\n===== PREFERENCE DIVERSITY =====")

print(
    f"Mean return : "
    f"{segment_returns.mean():.6f}"
)

print(
    f"Std return  : "
    f"{segment_returns.std():.6f}"
)

print(
    f"Min return  : "
    f"{segment_returns.min():.6f}"
)

print(
    f"Max return  : "
    f"{segment_returns.max():.6f}"
)

print(
    f"Unique returns : "
    f"{len(np.unique(segment_returns))}"
)

print("\nDone.")