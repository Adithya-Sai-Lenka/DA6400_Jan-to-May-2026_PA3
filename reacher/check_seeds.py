import numpy as np
import glob

reward_type = "rc"   # change: ra / rb / rc
eval_key = "eval_rc" # change accordingly

files = sorted(glob.glob(f"logs/{reward_type}_seed_*.npy"))

worst_seed = None
worst_value = float("inf")

for file in files:

    logs = np.load(file, allow_pickle=True).item()

    seed = logs["seed"]

    final_eval = logs[eval_key][-1]

    print(f"Seed {seed}: final {eval_key} = {final_eval:.2f}")

    if final_eval < worst_value:

        worst_value = final_eval
        worst_seed = seed

print("\nWorst seed:")
print(f"Seed {worst_seed} with {eval_key} = {worst_value:.2f}")