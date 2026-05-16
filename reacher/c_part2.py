import numpy as np


def final_mean_std(x):

    x = np.array(x)

    # use final 10 eval points
    x = x[-10:]

    return x.mean(), x.std()


reward_types = ["ra", "rb", "rc"]


for train_reward in reward_types:

    all_ra = []
    all_rb = []
    all_rc = []

    for seed in range(15):

        data = np.load(
            f"logs/{train_reward}_seed_{seed}.npy",
            allow_pickle=True
        ).item()

        all_ra.extend(
            data["eval_ra"][-30:]
        )

        all_rb.extend(
            data["eval_rb"][-30:]
        )

        all_rc.extend(
            data["eval_rc"][-30:]
        )

    mean_ra, std_ra = (
        np.mean(all_ra),
        np.std(all_ra)
    )

    mean_rb, std_rb = (
        np.mean(all_rb),
        np.std(all_rb)
    )

    mean_rc, std_rc = (
        np.mean(all_rc),
        np.std(all_rc)
    )

    print("\n================================")

    print(
        f"SAC-{train_reward.upper()}"
    )

    print(
        f"Eval on RA: "
        f"{mean_ra:.2f} +/- {std_ra:.2f}"
    )

    print(
        f"Eval on RB: "
        f"{mean_rb:.2f} +/- {std_rb:.2f}"
    )

    print(
        f"Eval on RC: "
        f"{mean_rc:.2f} +/- {std_rc:.2f}"
    )

    print("================================")