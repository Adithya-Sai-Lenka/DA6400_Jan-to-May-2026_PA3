# run_parallel_pebble.py

import concurrent.futures
import multiprocessing as mp
import os

from pebble_train import train_pebble


def run_task(args):

    reward_type, seed = args

    train_pebble(
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

    reward_type = "ra"

    num_seeds = 15

    max_workers = os.cpu_count() - 2

    tasks = [
        (reward_type, seed)
        for seed in range(num_seeds)
    ]

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