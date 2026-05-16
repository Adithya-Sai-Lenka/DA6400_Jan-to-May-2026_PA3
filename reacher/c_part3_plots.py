import os
import numpy as np
import matplotlib.pyplot as plt


os.makedirs(
    "plots",
    exist_ok=True
)


reward_types = [
    "ra",
    "rb",
    "rc"
]


eval_keys = {
    "ra": "eval_ra",
    "rb": "eval_rb",
    "rc": "eval_rc"
}


plot_titles = {
    "ra": r"Evaluation using $R_a$",
    "rb": r"Evaluation using $R_b$",
    "rc": r"Evaluation using $R_c$"
}


save_names = {
    "ra": "plots/all_sac_eval_ra.png",
    "rb": "plots/all_sac_eval_rb.png",
    "rc": "plots/all_sac_eval_rc.png"
}


labels = {
    "ra": r"SAC-$R_a$",
    "rb": r"SAC-$R_b$",
    "rc": r"SAC-$R_c$"
}


# ============================================================
# CREATE 3 FIGURES
# ============================================================

for eval_reward in reward_types:

    plt.figure(
        figsize=(8, 5)
    )

    # --------------------------------------------------------
    # Plot SAC-Ra, SAC-Rb, SAC-Rc
    # evaluated on current eval_reward
    # --------------------------------------------------------

    for train_reward in reward_types:

        all_curves = []

        for seed in range(15):

            data = np.load(
                f"logs/{train_reward}_seed_{seed}.npy",
                allow_pickle=True
            ).item()

            steps = np.array(
                data["steps"]
            )

            curve = np.array(
                data[eval_keys[eval_reward]]
            )

            all_curves.append(
                curve
            )

        all_curves = np.array(
            all_curves
        )

        mean_curve = all_curves.mean(
            axis=0
        )

        std_curve = all_curves.std(
            axis=0
        )

        plt.plot(
            steps,
            mean_curve,
            label=labels[train_reward],
            linewidth=2
        )

        plt.fill_between(
            steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    plt.xlabel(
        "Environment Timesteps",
        fontsize=12
    )

    plt.ylabel(
        "Average Return",
        fontsize=12
    )

    plt.title(
        plot_titles[eval_reward],
        fontsize=14
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        save_names[eval_reward],
        dpi=300
    )

    print(
        f"Saved: {save_names[eval_reward]}"
    )

    plt.close()