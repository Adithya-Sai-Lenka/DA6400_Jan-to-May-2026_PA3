# ============================================================
# run_parallel.py
# ============================================================

import concurrent.futures
import multiprocessing as mp

from train import train


def run_task(args):

    reward_type, seed = args

    train(
        reward_type=reward_type,
        seed=seed
    )


if __name__ == "__main__":

    try:
        mp.set_start_method(
            "spawn",
            force=True
        )

    except RuntimeError:

        pass

    reward_type = "rc"

    num_seeds = 15

    tasks = [
        (reward_type, seed)
        for seed in range(num_seeds)
    ]

    max_workers = 4

    print(f"Running {len(tasks)} seeds")
    print(f"Using {max_workers} workers")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(run_task, task)
            for task in tasks
        ]

        for future in concurrent.futures.as_completed(futures):

            try:

                future.result()

                print("Task completed")

            except Exception as e:

                print("Task failed:", e)