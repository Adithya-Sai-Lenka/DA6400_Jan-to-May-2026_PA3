import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import t
except ImportError:  # pragma: no cover - optional dependency fallback
    t = None


EXPERIMENTS = {
    "continuous-default": {
        "label": "Continuous SAC",
        "pattern": "sac_lunar_lander_default_auto_alpha_seed_*.csv",
        "color": "#1f77b4",
    },
    "hover-auto": {
        "label": "Hover reward, auto alpha",
        "pattern": "sac_lunar_lander_hover_switch_auto_alpha_seed_*.csv",
        "color": "#2ca02c",
    },
    "hover-fixed": {
        "label": "Hover reward, fixed alpha=0.01",
        "pattern": "sac_lunar_lander_hover_switch_fixed_alpha_0.01_seed_*.csv",
        "color": "#d62728",
    },
    "discrete-sac": {
        "label": "Discrete SAC",
        "pattern": "lunar_lander_discrete_discrete-sac_seed_*.csv",
        "color": "#9467bd",
    },
    "dqn": {
        "label": "DQN",
        "pattern": "lunar_lander_discrete_dqn_seed_*.csv",
        "color": "#ff7f0e",
    },
}


def latest_log_dir(root):
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")

    def score(path):
        experiment_counts = [len(list(path.glob(spec["pattern"]))) for spec in EXPERIMENTS.values()]
        return (sum(experiment_counts), path.stat().st_mtime)

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


def load_experiment(log_dir, experiment, metric):
    spec = EXPERIMENTS[experiment]
    seed_runs = []
    for path in sorted(log_dir.glob(spec["pattern"])):
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


def confidence_interval(values):
    n = values.shape[0]
    if n <= 1:
        return np.zeros(values.shape[1], dtype=float)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(n)
    multiplier = t.ppf(0.975, df=n - 1) if t is not None else 1.96
    return multiplier * sem


def style_axis(ax, ylabel):
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric(log_dir, output_dir, experiments, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    plotted = False

    for experiment in experiments:
        data = load_experiment(log_dir, experiment, metric)
        if data is None:
            print(f"Skipping {experiment}: no usable {metric} values found")
            continue

        spec = EXPERIMENTS[experiment]
        label = f"{spec['label']} (n={data['n']})"
        lower = data["mean"] - data["ci"]
        upper = data["mean"] + data["ci"]
        ax.plot(data["steps"], data["mean"], label=label, color=spec["color"], linewidth=2.0)
        ax.fill_between(data["steps"], lower, upper, color=spec["color"], alpha=0.18, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(title)
    style_axis(ax, ylabel)
    ax.legend(frameon=False)
    path = output_dir / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_summary(log_dir, output_dir):
    summary_path = output_dir / "lunar_lander_multiseed_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "metric",
                "step",
                "mean",
                "ci95",
                "num_seeds",
            ],
        )
        writer.writeheader()
        for experiment in EXPERIMENTS:
            for metric in ["eval_avg_return", "alpha", "epsilon"]:
                data = load_experiment(log_dir, experiment, metric)
                if data is None:
                    continue
                for step, mean, ci95 in zip(data["steps"], data["mean"], data["ci"]):
                    writer.writerow(
                        {
                            "experiment": experiment,
                            "metric": metric,
                            "step": int(step),
                            "mean": mean,
                            "ci95": ci95,
                            "num_seeds": data["n"],
                        }
                    )
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PA3 LunarLander multiseed learning curves with 95% confidence intervals."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing per-seed CSV logs. Default picks the fullest logs_multiseed run.",
    )
    parser.add_argument("--logs-root", type=Path, default=Path("logs_multiseed"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "lunar_lander")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir if args.log_dir is not None else latest_log_dir(args.logs_root)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading logs from: {log_dir}")
    print(f"Writing plots to: {output_dir}")

    outputs = [
        plot_metric(
            log_dir,
            output_dir,
            ["continuous-default"],
            "eval_avg_return",
            "Average evaluation return",
            "Continuous LunarLander SAC",
            "01_continuous_sac_return_ci.png",
        ),
        plot_metric(
            log_dir,
            output_dir,
            ["continuous-default", "hover-auto", "hover-fixed"],
            "eval_avg_return",
            "Average evaluation return",
            "Hover Reward Ablation",
            "02_hover_reward_ablation_return_ci.png",
        ),
        plot_metric(
            log_dir,
            output_dir,
            ["discrete-sac", "dqn"],
            "eval_avg_return",
            "Average evaluation return",
            "Discrete LunarLander: SAC vs DQN",
            "03_discrete_sac_vs_dqn_return_ci.png",
        ),
        plot_metric(
            log_dir,
            output_dir,
            ["continuous-default", "hover-auto", "discrete-sac"],
            "alpha",
            "Entropy temperature alpha",
            "Automatic Entropy Temperature",
            "04_alpha_temperature_ci.png",
        ),
        plot_metric(
            log_dir,
            output_dir,
            ["continuous-default", "hover-auto", "hover-fixed", "discrete-sac", "dqn"],
            "eval_avg_return",
            "Average evaluation return",
            "All LunarLander Experiments",
            "05_all_experiments_return_ci.png",
        ),
    ]
    summary_path = write_summary(log_dir, output_dir)

    for path in outputs:
        if path is not None:
            print(f"Saved {path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
