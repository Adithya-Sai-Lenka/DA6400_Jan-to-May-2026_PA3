# ============================================================
# FINAL train.py
# ============================================================

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "osmesa"

import random
import numpy as np
import torch

torch.set_num_threads(1)

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

            # ================================================
            # Rc evaluation simplification
            # ================================================

            if reward_type == "rc":

                if info.get("timeout", False):

                    ep_return = -1020

                    done = True

        returns.append(ep_return)

    return np.mean(returns)


# ============================================================
# TRAIN
# ============================================================

def train(reward_type="rb",
          seed=0):

    print(f"\nStarting Seed {seed} | Reward={reward_type}\n")

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

    # ========================================================
    # HYPERPARAMETERS
    # ========================================================

    total_steps = 500_000

    random_steps = 10_000

    eval_freq = 10_000

    batch_size = 256

    obs = env.reset()

    # ========================================================
    # LOGS
    # ========================================================

    logs = {

        "reward_type": reward_type,

        "seed": seed,

        "steps": [],

        "eval_ra": [],
        "eval_rb": [],
        "eval_rc": [],

        "train_episode_returns": [],
        "train_episode_lengths": []
    }

    os.makedirs("logs", exist_ok=True)

    # ========================================================
    # INITIAL EVAL @ STEP 0
    # ========================================================

    print("\n====================================")
    print(f"[Seed {seed}] INITIAL EVAL @ STEP 0")
    print("====================================")

    eval_ra = evaluate_policy(
        agent,
        "ra"
    )

    eval_rb = evaluate_policy(
        agent,
        "rb"
    )

    eval_rc = evaluate_policy(
        agent,
        "rc"
    )

    logs["steps"].append(0)

    logs["eval_ra"].append(eval_ra)

    logs["eval_rb"].append(eval_rb)

    logs["eval_rc"].append(eval_rc)

    print(f"[Seed {seed}] Eval Ra: {eval_ra:.2f}")
    print(f"[Seed {seed}] Eval Rb: {eval_rb:.2f}")
    print(f"[Seed {seed}] Eval Rc: {eval_rc:.2f}")

    print("====================================\n")

    np.save(
        f"logs/{reward_type}_seed_{seed}.npy",
        logs,
        allow_pickle=True
    )

    # ========================================================
    # EPISODE BOOKKEEPING
    # ========================================================

    episode_return = 0

    episode_length = 0

    # ========================================================
    # MAIN LOOP
    # ========================================================

    for step in range(1, total_steps + 1):

        # ====================================================
        # ACTION SELECTION
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

        next_obs, reward, done, info = env.step(action)

        # ====================================================
        # IMPORTANT:
        # Rc timeout resets should terminate Bellman backup
        # but NOT episode bookkeeping.
        # ====================================================

        buffer_done = done

        if reward_type == "rc":

            if info.get("timeout", False):

                buffer_done = True

        replay_buffer.add(
            obs,
            action,
            reward,
            next_obs,
            buffer_done
        )

        obs = next_obs

        episode_return += reward

        # ====================================================
        # Rc effective episode length
        # ====================================================

        if reward_type == "rc":

            if info.get("timeout", False):

                # 1000 env steps + 20 reset cost
                episode_length += 1020

            else:

                episode_length += 1

        else:

            episode_length += 1

        # ====================================================
        # SAC UPDATE
        # ====================================================

        if step >= random_steps:

            agent.update(
                replay_buffer,
                batch_size=batch_size
            )

        # ====================================================
        # EPISODE TERMINATION
        # ====================================================

        if done:

            logs["train_episode_returns"].append(
                episode_return
            )

            logs["train_episode_lengths"].append(
                episode_length
            )

            print(
                f"[Seed {seed}] "
                f"Reward={reward_type} | "
                f"Step={step} | "
                f"Return={episode_return:.2f} | "
                f"Length={episode_length}"
            )

            obs = env.reset()

            episode_return = 0

            episode_length = 0

        # ====================================================
        # EVALUATION
        # ====================================================

        if step % eval_freq == 0:

            eval_ra = evaluate_policy(
                agent,
                "ra"
            )

            eval_rb = evaluate_policy(
                agent,
                "rb"
            )

            eval_rc = evaluate_policy(
                agent,
                "rc"
            )

            logs["steps"].append(step)

            logs["eval_ra"].append(eval_ra)

            logs["eval_rb"].append(eval_rb)

            logs["eval_rc"].append(eval_rc)

            print("\n====================================")
            print(f"[Seed {seed}] EVAL @ STEP {step}")
            print(f"[Seed {seed}] Eval Ra: {eval_ra:.2f}")
            print(f"[Seed {seed}] Eval Rb: {eval_rb:.2f}")
            print(f"[Seed {seed}] Eval Rc: {eval_rc:.2f}")
            print(f"[Seed {seed}] Alpha: {agent.alpha.item():.4f}")
            print("====================================\n")

            # =================================================
            # SAVE LOGS
            # =================================================

            np.save(
                f"logs/{reward_type}_seed_{seed}.npy",
                logs,
                allow_pickle=True
            )

            # =================================================
            # SAVE POLICY
            # =================================================

            torch.save(
                agent.actor.state_dict(),
                f"logs/{reward_type}_seed_{seed}_actor.pt"
            )

    print(f"\n[Seed {seed}] Training Complete.\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train(
        reward_type="rb",
        seed=0
    )