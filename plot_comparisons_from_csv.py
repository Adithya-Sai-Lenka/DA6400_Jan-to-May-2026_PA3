import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAIN_COLORS = {
    "Continuous SAC": "#1f77b4",
    "Hover reward, auto alpha": "#2ca02c",
    "Hover reward, fixed alpha=0.01": "#d62728",
    "Discrete SAC": "#9467bd",
    "DQN": "#ff7f0e",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    return float(row[key])


def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_bar_comparison(rows, output_path, title, label_key, x_label):
    labels = [row[label_key] for row in rows]
    means = np.array([as_float(row, "mean_eval_return") for row in rows], dtype=float)
    lower = np.array([as_float(row, "ci95_lower") for row in rows], dtype=float)
    upper = np.array([as_float(row, "ci95_upper") for row in rows], dtype=float)
    yerr = np.vstack([means - lower, upper - means])
    colors = [MAIN_COLORS.get(label, "#4c78a8") for label in labels]

    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=yerr, capsize=5, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_xlabel(x_label)
    ax.set_title(title)
    style_axis(ax, "Final average evaluation return")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_curve_comparison(rows, output_path, title, group_key, label_name):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)

    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    for group, group_rows in grouped.items():
        group_rows = sorted(group_rows, key=lambda row: as_float(row, "step"))
        steps = np.array([as_float(row, "step") for row in group_rows], dtype=float)
        means = np.array([as_float(row, "mean_eval_return") for row in group_rows], dtype=float)
        lower = np.array([as_float(row, "ci95_lower") for row in group_rows], dtype=float)
        upper = np.array([as_float(row, "ci95_upper") for row in group_rows], dtype=float)
        label = group_rows[0].get("label", group)
        if group_key == "buffer_size":
            label = f"Buffer {int(float(group)):,}"
        color = MAIN_COLORS.get(label)
        ax.plot(steps, means, label=label, color=color, linewidth=2.0)
        ax.fill_between(steps, lower, upper, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel("Environment steps")
    ax.set_title(title)
    ax.legend(frameon=False)
    style_axis(ax, "Average evaluation return")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_main_lunar(args, outputs):
    final_csv = args.main_final_csv
    curve_csv = args.main_curve_csv
    if final_csv.exists():
        rows = read_rows(final_csv)
        outputs.append(
            save_bar_comparison(
                rows,
                args.output_dir / "main_final_reward_comparison.png",
                "LunarLander Final Performance Comparison",
                "label",
                "Experiment",
            )
        )
    if curve_csv.exists():
        rows = read_rows(curve_csv)
        outputs.append(
            save_curve_comparison(
                rows,
                args.output_dir / "main_learning_curve_comparison.png",
                "LunarLander Learning Curves from CSV",
                "experiment",
                "Experiment",
            )
        )


def plot_buffer_lunar(args, outputs):
    final_csv = args.buffer_final_csv
    curve_csv = args.buffer_curve_csv
    if final_csv.exists():
        rows = read_rows(final_csv)
        outputs.append(
            save_bar_comparison(
                rows,
                args.output_dir / "buffer_size_final_reward_comparison.png",
                "Replay Buffer Size Final Performance Comparison",
                "buffer_size",
                "Replay buffer size",
            )
        )
    if curve_csv.exists():
        rows = read_rows(curve_csv)
        outputs.append(
            save_curve_comparison(
                rows,
                args.output_dir / "buffer_size_learning_curve_comparison.png",
                "Replay Buffer Size Learning Curves from CSV",
                "buffer_size",
                "Replay buffer size",
            )
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create comparison plots from generated LunarLander CSV summary files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "csv_comparisons")
    parser.add_argument(
        "--main-final-csv",
        type=Path,
        default=Path("plots") / "lunar_lander" / "lunar_lander_final_average_rewards.csv",
    )
    parser.add_argument(
        "--main-curve-csv",
        type=Path,
        default=Path("plots") / "lunar_lander" / "lunar_lander_average_rewards_by_step.csv",
    )
    parser.add_argument(
        "--buffer-final-csv",
        type=Path,
        default=Path("plots") / "lunar_lander_buffer_size" / "buffer_size_final_average_rewards.csv",
    )
    parser.add_argument(
        "--buffer-curve-csv",
        type=Path,
        default=Path("plots") / "lunar_lander_buffer_size" / "buffer_size_average_rewards_by_step.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    plot_main_lunar(args, outputs)
    plot_buffer_lunar(args, outputs)

    if not outputs:
        print("No expected CSV files were found. Generate the CSV summaries first.")
        return

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
