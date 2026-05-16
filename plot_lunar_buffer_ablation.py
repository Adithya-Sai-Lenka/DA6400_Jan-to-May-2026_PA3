import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import t
except ImportError:  # pragma: no cover
    t = None


BUFFER_DIR_RE = re.compile(r"buffer_(\d+)$")


def latest_buffer_run(root):
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No buffer ablation runs found under {root}")

    def score(path):
        csv_count = len(list(path.glob("buffer_*/*.csv")))
        return (csv_count, path.stat().st_mtime)

    return max(candidates, key=score)


def read_metric_by_step(path, metric):
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_value = row.get(metric, "")
            if raw_value == "":
                continue
            values[int(float(row["step"]))] = float(raw_value)
    return values


def confidence_interval(values):
    n = values.shape[0]
    if n <= 1:
        return np.zeros(values.shape[1], dtype=float)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(n)
    multiplier = t.ppf(0.975, df=n - 1) if t is not None else 1.96
    return multiplier * sem


def find_buffer_dirs(log_dir):
    buffer_dirs = []
    for path in log_dir.iterdir():
        if not path.is_dir():
            continue
        match = BUFFER_DIR_RE.match(path.name)
        if match:
            buffer_dirs.append((int(match.group(1)), path))
    return sorted(buffer_dirs)


def load_buffer_data(buffer_dir, metric):
    seed_runs = []
    for path in sorted(buffer_dir.glob("sac_lunar_lander_hover_switch_auto_alpha_seed_*.csv")):
        run = read_metric_by_step(path, metric)
        if run:
            seed_runs.append(run)

    if not seed_runs:
        return None

    common_steps = sorted(set.intersection(*(set(run) for run in seed_runs)))
    if not common_steps:
        return None

    y = np.array([[run[step] for step in common_steps] for run in seed_runs], dtype=float)
    return {
        "steps": np.array(common_steps, dtype=int),
        "values": y,
        "mean": np.mean(y, axis=0),
        "ci": confidence_interval(y),
        "n": y.shape[0],
    }


def style_axis(ax, ylabel):
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_return_curves(log_dir, output_dir):
    colors = {
        50_000: "#1f77b4",
        70_000: "#2ca02c",
        120_000: "#ff7f0e",
        150_000: "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    plotted = False

    for buffer_size, buffer_dir in find_buffer_dirs(log_dir):
        data = load_buffer_data(buffer_dir, "eval_avg_return")
        if data is None:
            print(f"Skipping buffer={buffer_size}: no usable eval_avg_return values found")
            continue

        color = colors.get(buffer_size)
        label = f"Buffer {buffer_size:,} (n={data['n']})"
        lower = data["mean"] - data["ci"]
        upper = data["mean"] + data["ci"]
        ax.plot(data["steps"], data["mean"], label=label, color=color, linewidth=2.0)
        ax.fill_between(data["steps"], lower, upper, color=color, alpha=0.18, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.axvline(250_000, color="#444444", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(250_000, ax.get_ylim()[0], " reward switch", va="bottom", ha="left", color="#444444")
    ax.set_title("Replay Buffer Size Ablation in Changing Hover Environment")
    style_axis(ax, "Average evaluation return")
    ax.legend(frameon=False)
    path = output_dir / "buffer_size_return_ci.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_curve_csv(log_dir, output_dir):
    path = output_dir / "buffer_size_average_rewards_by_step.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "buffer_size",
                "step",
                "mean_eval_return",
                "ci95",
                "ci95_lower",
                "ci95_upper",
                "num_seeds",
            ],
        )
        writer.writeheader()

        for buffer_size, buffer_dir in find_buffer_dirs(log_dir):
            data = load_buffer_data(buffer_dir, "eval_avg_return")
            if data is None:
                continue
            for step, mean, ci95 in zip(data["steps"], data["mean"], data["ci"]):
                writer.writerow(
                    {
                        "buffer_size": buffer_size,
                        "step": int(step),
                        "mean_eval_return": f"{mean:.6f}",
                        "ci95": f"{ci95:.6f}",
                        "ci95_lower": f"{mean - ci95:.6f}",
                        "ci95_upper": f"{mean + ci95:.6f}",
                        "num_seeds": data["n"],
                    }
                )
    return path


def write_final_csv_and_markdown(log_dir, output_dir):
    csv_path = output_dir / "buffer_size_final_average_rewards.csv"
    markdown_path = output_dir / "buffer_size_final_average_rewards.md"
    rows = []

    for buffer_size, buffer_dir in find_buffer_dirs(log_dir):
        data = load_buffer_data(buffer_dir, "eval_avg_return")
        if data is None:
            continue
        step = int(data["steps"][-1])
        mean = float(data["mean"][-1])
        ci95 = float(data["ci"][-1])
        rows.append(
            {
                "buffer_size": buffer_size,
                "final_step": step,
                "mean_eval_return": f"{mean:.6f}",
                "ci95": f"{ci95:.6f}",
                "ci95_lower": f"{mean - ci95:.6f}",
                "ci95_upper": f"{mean + ci95:.6f}",
                "num_seeds": data["n"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "buffer_size",
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

    lines = [
        "| Replay buffer size | Final step | Mean eval return | 95% CI | Seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {buffer_size} | {final_step} | {mean_eval_return} | "
            "[{ci95_lower}, {ci95_upper}] | {num_seeds} |".format(**row)
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LunarLander replay buffer-size ablation with 95% confidence intervals."
    )
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=Path("logs_buffer_ablation"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "lunar_lander_buffer_size")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir if args.log_dir is not None else latest_buffer_run(args.logs_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading logs from: {log_dir}")
    print(f"Writing buffer-size outputs to: {output_dir}")

    plot_path = plot_return_curves(log_dir, output_dir)
    curve_csv = write_curve_csv(log_dir, output_dir)
    final_csv, final_md = write_final_csv_and_markdown(log_dir, output_dir)

    if plot_path is not None:
        print(f"Saved {plot_path}")
    print(f"Saved {curve_csv}")
    print(f"Saved {final_csv}")
    print(f"Saved {final_md}")


if __name__ == "__main__":
    main()
