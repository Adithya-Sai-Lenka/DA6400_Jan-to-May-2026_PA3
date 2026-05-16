# evaluate_behavior.py
# This script evaluates the behavior of the trained policies for RA, RB, and RC.
# It runs multiple episodes for each seed and reward type, and computes statistics on
# the steps required to reach the target and the time spent inside the target region.

import os
os.environ["MUJOCO_GL"] = "osmesa"
import concurrent.futures
import multiprocessing as mp
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from sac import SACAgent
from reacher_env import ReacherEnv


NUM_EPISODES = 500
EPISODE_LENGTH = 5000

NUM_WORKERS = 7

TARGET_THRESHOLDS = {
    "ra": 0.03,
    "rb": 0.03,
    "rc": 0.05
}


# distance computation for r_a


def get_distance(env):

    fingertip = env.env.physics.named.data.geom_xpos[
        "finger",
        :2
    ]

    target = env.env.physics.named.data.geom_xpos[
        "target",
        :2
    ]

    return np.linalg.norm(fingertip - target)

# worker function

def evaluate_seed(args):

    reward_type, seed = args

    env = ReacherEnv(
        reward_type=reward_type,
        seed=seed,
        eval_mode=True
    )

    agent = SACAgent(
        env.obs_dim,
        env.action_dim,
        device="cpu"
    )

    checkpoint_path = (
        f"logs/{reward_type}_seed_{seed}_actor.pt"
    )

    agent.actor.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu"
        )
    )

    agent.actor.eval()

    threshold = TARGET_THRESHOLDS[reward_type]

    steps_to_goal_all = []
    steps_in_target_all = []

    for ep in range(NUM_EPISODES):

        obs = env.reset()

        reached = False

        steps_to_goal = EPISODE_LENGTH
        steps_in_target = 0

        for step in range(EPISODE_LENGTH):

            action = agent.select_action(
                obs,
                deterministic=True
            )

            obs, reward, done, info = env.step(action)

            dist = get_distance(env)

            inside_target = dist < threshold

            if inside_target:

                steps_in_target += 1

                if not reached:

                    reached = True

                    steps_to_goal = step + 1

        steps_to_goal_all.append(
            steps_to_goal
        )

        steps_in_target_all.append(
            steps_in_target
        )

    return {
        "reward_type": reward_type,
        "seed": seed,

        "goal_steps": np.array(
            steps_to_goal_all
        ),

        "target_steps": np.array(
            steps_in_target_all
        )
    }

# main

if __name__ == "__main__":

    try:

        mp.set_start_method(
            "spawn",
            force=True
        )

    except RuntimeError:

        pass

    tasks = []

    for reward_type in ["ra", "rb", "rc"]:

        for seed in range(15):

            tasks.append(
                (reward_type, seed)
            )

    print(f"\nTotal jobs: {len(tasks)}")

    print(f"Workers: {NUM_WORKERS}")

    results = {
        "ra": {
            "goal": [],
            "target": []
        },
        "rb": {
            "goal": [],
            "target": []
        },
        "rc": {
            "goal": [],
            "target": []
        }
    }

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                evaluate_seed,
                task
            )
            for task in tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating Policies"
        ):

            out = future.result()

            rt = out["reward_type"]

            results[rt]["goal"].append(
                out["goal_steps"]
            )

            results[rt]["target"].append(
                out["target_steps"]
            )

    # aggregate results

    final_stats = {}

    for rt in ["ra", "rb", "rc"]:

        goal = np.concatenate(
            results[rt]["goal"]
        )

        target = np.concatenate(
            results[rt]["target"]
        )

        final_stats[rt] = {

            "goal_mean": goal.mean(),
            "goal_std": goal.std(),

            "target_mean": target.mean(),
            "target_std": target.std()
        }

    print("FINAL RESULTS")

    for rt in ["ra", "rb", "rc"]:

        stats = final_stats[rt]

        print(f"\n{rt.upper()}")

        print(
            f"Steps to Goal: "
            f"{stats['goal_mean']:.2f} "
            f"+/- "
            f"{stats['goal_std']:.2f}"
        )

        print(
            f"Steps in Target: "
            f"{stats['target_mean']:.2f} "
            f"+/- "
            f"{stats['target_std']:.2f}"
        )

    labels = ["RA", "RB", "RC"]

    goal_means = [
        final_stats["ra"]["goal_mean"],
        final_stats["rb"]["goal_mean"],
        final_stats["rc"]["goal_mean"]
    ]

    goal_stds = [
        final_stats["ra"]["goal_std"],
        final_stats["rb"]["goal_std"],
        final_stats["rc"]["goal_std"]
    ]

    target_means = [
        final_stats["ra"]["target_mean"],
        final_stats["rb"]["target_mean"],
        final_stats["rc"]["target_mean"]
    ]

    target_stds = [
        final_stats["ra"]["target_std"],
        final_stats["rb"]["target_std"],
        final_stats["rc"]["target_std"]
    ]

    plt.figure(figsize=(7,5))

    plt.bar(
        labels,
        goal_means,
        yerr=goal_stds,
        capsize=5
    )

    plt.ylabel("Steps to Goal")

    plt.title(
        "Steps Required to Reach Target"
    )

    plt.tight_layout()

    plt.savefig(
        "logs/steps_to_goal_bar.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(7,5))

    plt.bar(
        labels,
        target_means,
        yerr=target_stds,
        capsize=5
    )

    plt.ylabel("Steps in Target")

    plt.title(
        "Time Spent Inside Target Region"
    )

    plt.tight_layout()

    plt.savefig(
        "logs/steps_in_target_bar.png",
        dpi=300
    )

    plt.close()

    print("\nSaved plots to logs/")