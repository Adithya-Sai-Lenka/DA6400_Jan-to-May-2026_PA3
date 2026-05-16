import numpy as np
import matplotlib.pyplot as plt
import glob

files = sorted(
    glob.glob("logs/pebble_ra_seed_*.npy")
)

plt.figure(figsize=(8,5))

for f in files:

    data = np.load(
        f,
        allow_pickle=True
    ).item()

    steps = data["steps"]

    returns = data["eval_return"]

    seed = f.split("_seed_")[1].split(".")[0]

    plt.plot(
        steps,
        returns,
        label=f"seed {seed}"
    )

plt.xlabel("Environment Steps")

plt.ylabel("Evaluation Return")

plt.title("PEBBLE-RA Individual Seed Curves")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "logs/pebble_ra_individual_seeds.png",
    dpi=300
)

plt.show()