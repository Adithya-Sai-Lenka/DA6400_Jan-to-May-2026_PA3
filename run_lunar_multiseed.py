import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


EXPERIMENTS = {
    "continuous-default": {
        "script": "SAC_lunar_lander.py",
        "args": ["--train"],
    },
    "hover-auto": {
        "script": "SAC_lunar_lander.py",
        "args": ["--train", "--hover-reward", "200", "--switched-hover-reward", "-100"],
    },
    "hover-fixed": {
        "script": "SAC_lunar_lander.py",
        "args": [
            "--train",
            "--hover-reward",
            "200",
            "--switched-hover-reward",
            "-100",
            "--fixed-alpha",
            "0.01",
        ],
    },
    "discrete-sac": {
        "script": "SAC_lunar_lander_discrete.py",
        "args": ["--algo", "discrete-sac"],
    },
    "dqn": {
        "script": "SAC_lunar_lander_discrete.py",
        "args": ["--algo", "dqn"],
    },
}


def default_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs_multiseed") / f"run_{stamp}"


def build_command(experiment, seed, args, output_dir):
    spec = EXPERIMENTS[experiment]
    command = [
        sys.executable,
        spec["script"],
        *spec["args"],
        "--seed",
        str(seed),
        "--total-steps",
        str(args.total_steps),
        "--random-steps",
        str(args.random_steps),
        "--replay-capacity",
        str(args.replay_capacity),
        "--eval-freq",
        str(args.eval_freq),
        "--log-dir",
        str(output_dir),
    ]

    if experiment in {"hover-auto", "hover-fixed"}:
        command.extend(["--hover-switch-step", str(args.hover_switch_step)])

    if experiment in {"discrete-sac", "dqn"}:
        command.extend(["--eval-episodes", str(args.eval_episodes)])
        if experiment == "discrete-sac":
            command.extend(["--target-entropy-ratio", str(args.target_entropy_ratio)])
            command.extend(["--min-alpha", str(args.min_alpha)])
            command.extend(["--max-alpha", str(args.max_alpha)])
        if experiment == "dqn":
            command.extend(["--dqn-lr", str(args.dqn_lr)])
            command.extend(["--grad-clip-norm", str(args.grad_clip_norm)])
    else:
        command.extend(["--eval-episodes", str(args.eval_episodes)])

    if args.enable_wind:
        command.append("--enable-wind")

    return command


def write_manifest_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "seed",
                "status",
                "returncode",
                "command",
                "started_order",
                "started_at",
                "finished_at",
                "duration_sec",
            ],
        )
        writer.writeheader()


def append_manifest(path, row):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "seed",
                "status",
                "returncode",
                "command",
                "started_order",
                "started_at",
                "finished_at",
                "duration_sec",
            ],
        )
        writer.writerow(row)


def stream_run(run_index, total_runs, experiment, seed, command, manifest_path, manifest_lock):
    command_text = " ".join(command)
    prefix = f"[{run_index:03d}/{total_runs:03d} {experiment} seed={seed}]"
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()

    with manifest_lock:
        append_manifest(
            manifest_path,
            {
                "experiment": experiment,
                "seed": seed,
                "status": "started",
                "returncode": "",
                "command": command_text,
                "started_order": run_index,
                "started_at": started_at,
                "finished_at": "",
                "duration_sec": "",
            },
        )

    print(f"{prefix} START")
    print(f"{prefix} CMD {command_text}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(f"{prefix} {line.rstrip()}")

    returncode = process.wait()
    finished_at = datetime.now().isoformat(timespec="seconds")
    duration_sec = time.perf_counter() - start_time
    status = "completed" if returncode == 0 else "failed"

    with manifest_lock:
        append_manifest(
            manifest_path,
            {
                "experiment": experiment,
                "seed": seed,
                "status": status,
                "returncode": returncode,
                "command": command_text,
                "started_order": run_index,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": f"{duration_sec:.1f}",
            },
        )

    print(f"{prefix} {status.upper()} returncode={returncode} duration={duration_sec:.1f}s")
    return {
        "experiment": experiment,
        "seed": seed,
        "returncode": returncode,
        "duration_sec": duration_sec,
    }


def run_all(args):
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    write_manifest_header(manifest_path)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    total_runs = len(args.experiments) * len(seeds)
    max_workers = min(args.max_workers, total_runs)

    print(f"Writing all CSV logs to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Experiments: {', '.join(args.experiments)}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Total runs: {total_runs}")
    print(f"Parallel workers: {max_workers}")

    jobs = []
    run_index = 0

    for experiment in args.experiments:
        for seed in seeds:
            run_index += 1
            command = build_command(experiment, seed, args, output_dir)
            jobs.append((run_index, experiment, seed, command))

    manifest_lock = threading.Lock()
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                stream_run,
                run_index,
                total_runs,
                experiment,
                seed,
                command,
                manifest_path,
                manifest_lock,
            )
            for run_index, experiment, seed, command in jobs
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["returncode"] != 0:
                failures.append(result)

    if failures:
        print(f"\n{len(failures)} run(s) failed. CSV logs and manifest are in: {output_dir}")
        for failure in failures:
            print(
                f"FAILED {failure['experiment']} seed={failure['seed']} "
                f"returncode={failure['returncode']}"
            )
        raise SystemExit(1)

    print(f"\nAll runs completed. CSV logs and manifest are in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PA3 LunarLander experiments for multiple seeds and save CSV logs."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(EXPERIMENTS),
        help="Experiments to run. Default runs all LunarLander 2.2 experiments.",
    )
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of experiment subprocesses to run at once.",
    )
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--hover-switch-step", type=int, default=None)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--random-steps", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--enable-wind", action="store_true")
    parser.add_argument("--target-entropy-ratio", type=float, default=0.5)
    parser.add_argument("--min-alpha", type=float, default=1e-4)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--dqn-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    args = parser.parse_args()

    if args.hover_switch_step is None:
        args.hover_switch_step = args.total_steps // 2
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")

    return args


if __name__ == "__main__":
    run_all(parse_args())
