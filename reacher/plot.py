import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_reward",
    type=str,
    required=True,
    choices=["a", "b", "c"]
)

parser.add_argument(
    "--eval_reward",
    type=str,
    required=True,
    choices=["a", "b", "c"]
)

args = parser.parse_args()

train_reward = f"r{args.train_reward}"

eval_key = f"eval_r{args.eval_reward}"


all_logs = []

seed = 0

while True:

    try:

        logs = np.load(
            f"logs/{train_reward}_seed_{seed}.npy",
            allow_pickle=True
        ).item()

        all_logs.append(logs)

        seed += 1

    except FileNotFoundError:

        break


if len(all_logs) == 0:

    raise ValueError(
        f"No logs found for {train_reward}"
    )


steps = np.array(all_logs[0]["steps"])

evals = np.array([
    logs[eval_key]
    for logs in all_logs
])


mean = evals.mean(axis=0)

std = evals.std(axis=0)


plt.figure(figsize=(10,6))

plt.plot(
    steps,
    mean,
    label=(
        f"SAC-{train_reward.upper()} "
        f"evaluated on R_{args.eval_reward}"
    )
)

plt.fill_between(
    steps,
    mean - std,
    mean + std,
    alpha=0.2
)

plt.xlabel("Environment Timesteps")

plt.ylabel(
    "Average Return ± Std Dev (across seeds)"
)

plt.title(
    f"SAC-{train_reward.upper()} "
    f"evaluated on R_{args.eval_reward}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


save_path = (
    f"logs/"
    f"{train_reward}_eval_"
    f"r{args.eval_reward}.png"
)

plt.savefig(save_path)

print(f"\nSaved plot to: {save_path}\n")

plt.show()