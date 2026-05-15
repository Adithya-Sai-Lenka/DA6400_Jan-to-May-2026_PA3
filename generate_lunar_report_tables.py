import argparse
import csv
from pathlib import Path

from plot_lunar_multiseed import EXPERIMENTS, latest_log_dir, load_experiment


def write_reward_curve_table(log_dir, output_dir):
    path = output_dir / "lunar_lander_average_rewards_by_step.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "label",
                "step",
                "mean_eval_return",
                "ci95",
                "ci95_lower",
                "ci95_upper",
                "num_seeds",
            ],
        )
        writer.writeheader()

        for experiment, spec in EXPERIMENTS.items():
            data = load_experiment(log_dir, experiment, "eval_avg_return")
            if data is None:
                continue

            for step, mean, ci95 in zip(data["steps"], data["mean"], data["ci"]):
                writer.writerow(
                    {
                        "experiment": experiment,
                        "label": spec["label"],
                        "step": int(step),
                        "mean_eval_return": f"{mean:.6f}",
                        "ci95": f"{ci95:.6f}",
                        "ci95_lower": f"{mean - ci95:.6f}",
                        "ci95_upper": f"{mean + ci95:.6f}",
                        "num_seeds": data["n"],
                    }
                )
    return path


def write_final_reward_table(log_dir, output_dir):
    path = output_dir / "lunar_lander_final_average_rewards.csv"
    rows = []

    for experiment, spec in EXPERIMENTS.items():
        data = load_experiment(log_dir, experiment, "eval_avg_return")
        if data is None:
            continue
        final_index = -1
        step = int(data["steps"][final_index])
        mean = float(data["mean"][final_index])
        ci95 = float(data["ci"][final_index])
        rows.append(
            {
                "experiment": experiment,
                "label": spec["label"],
                "final_step": step,
                "mean_eval_return": f"{mean:.6f}",
                "ci95": f"{ci95:.6f}",
                "ci95_lower": f"{mean - ci95:.6f}",
                "ci95_upper": f"{mean + ci95:.6f}",
                "num_seeds": data["n"],
            }
        )

    rows.sort(key=lambda row: float(row["mean_eval_return"]), reverse=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "label",
                "final_step",
                "mean_eval_return",
                "ci95",
                "ci95_lower",
                "ci95_upper",
                "num_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_markdown_table(csv_path, output_dir):
    path = output_dir / "lunar_lander_final_average_rewards.md"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        "| Experiment | Final step | Mean eval return | 95% CI | Seeds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {final_step} | {mean_eval_return} | "
            "[{ci95_lower}, {ci95_upper}] | {num_seeds} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-ready LunarLander average reward tables."
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=Path("logs_multiseed"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "lunar_lander")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir if args.log_dir is not None else latest_log_dir(args.logs_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_path = write_reward_curve_table(log_dir, output_dir)
    final_path = write_final_reward_table(log_dir, output_dir)
    markdown_path = write_markdown_table(final_path, output_dir)

    print(f"Reading logs from: {log_dir}")
    print(f"Saved {curve_path}")
    print(f"Saved {final_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
