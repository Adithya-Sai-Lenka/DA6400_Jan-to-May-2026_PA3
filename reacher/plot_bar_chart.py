# ============================================================
# proper_behavior_plots.py
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs(
    "plots",
    exist_ok=True
)

# ============================================================
# results
# ============================================================

goal_means = [
    57.95,
    45.53,
    136.65
]

goal_stds = [
    178.68,
    118.94,
    374.86
]

target_means = [
    4612.70,
    4697.48,
    3919.20
]

target_stds = [
    478.60,
    370.59,
    1009.20
]

labels = ["RA", "RB", "RC"]

N = 500


# ============================================================
# 95% confidence intervals
# ============================================================

goal_ci = [
    1.96 * s / np.sqrt(N)
    for s in goal_stds
]

target_ci = [
    1.96 * s / np.sqrt(N)
    for s in target_stds
]


# ============================================================
# plot: steps to goal
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(
    labels,
    goal_means,
    yerr=goal_ci,
    capsize=8
)

plt.xlabel(
    "Reward Formulation",
    fontsize=13
)

plt.ylabel(
    "Steps to Goal",
    fontsize=13
)

plt.title(
    "Average Steps to Reach Target",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "plots/steps_to_goal_bar.png",
    dpi=300
)

plt.close()


# ============================================================
# plot: steps in target
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(
    labels,
    target_means,
    yerr=target_ci,
    capsize=8
)

plt.xlabel(
    "Reward Formulation",
    fontsize=13
)

plt.ylabel(
    "Steps in Target",
    fontsize=13
)

plt.title(
    "Average Time Spent Inside Target Region",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "plots/steps_in_target_bar.png",
    dpi=300
)

plt.close()


print("Saved plots:")
print("plots/steps_to_goal_bar.png")
print("plots/steps_in_target_bar.png")