import numpy as np


seed_stats = []


for seed in range(15):

    x = np.load(
        f"logs/rc_seed_{seed}.npy",
        allow_pickle=True
    ).item()

    lengths = np.array(
        x["train_episode_lengths"]
    )

    last_50 = lengths[-50:]

    mean_len = last_50.mean()

    std_len = last_50.std()

    seed_stats.append(
        (
            seed,
            mean_len,
            std_len
        )
    )


seed_stats = sorted(
    seed_stats,
    key=lambda x: x[1]
)


print("RC SEED DIFFICULTY (sorted by mean episode length)")

for seed, mean_len, std_len in seed_stats:

    print(
        f"Seed {seed:2d} | "
        f"Mean={mean_len:.2f} | "
        f"Std={std_len:.2f}"
    )


print("Easiest seed")

print(seed_stats[0])

print("Hardest seed")

print(seed_stats[-1])