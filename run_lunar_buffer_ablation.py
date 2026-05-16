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


BUFFER_SIZES = [50_000, 70_000, 120_000, 150_000]


def default_output_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs_buffer_ablation") / f"run_{stamp}"


def build_command(seed, buffer_size, args, output_dir):
    return [
        sys.executable,
        "SAC_lunar_lander.py",
        "--train",
        "--seed",
        str(seed),
        "--total-steps",
        str(args.total_steps),
        "--random-steps",
        str(args.random_steps),
        "--replay-capacity",
        str(buffer_size),
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.eval_episodes),
        "--hover-reward",
        str(args.hover_reward),
        "--hover-switch-step",
        str(args.hover_switch_step),
        "--switched-hover-reward",
        str(args.switched_hover_reward),
        "--log-dir",
        str(output_dir / f"buffer_{buffer_size}"),
    ] + (["--enable-wind"] if args.enable_wind else [])


def manifest_fields():
    return [
        "buffer_size",
        "seed",
        "status",
        "returncode",
        "command",
        "started_order",
        "started_at",
        "finished_at",
        "duration_sec",
    ]


def write_manifest_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writeheader()


def append_manifest(path, row):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields())
        writer.writerow(row)


def stream_run(run_index, total_runs, seed, buffer_size, command, manifest_path, manifest_lock):
    command_text = " ".join(command)
    prefix = f"[{run_index:03d}/{total_runs:03d} buffer={buffer_size} seed={seed}]"
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()

    with manifest_lock:
        append_manifest(
            manifest_path,
            {
                "buffer_size": buffer_size,
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
                "buffer_size": buffer_size,
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
    return {"buffer_size": buffer_size, "seed": seed, "returncode": returncode}


def run_all(args):
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    write_manifest_header(manifest_path)

    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    total_runs = len(args.buffer_sizes) * len(seeds)
    max_workers = min(args.max_workers, total_runs)

    print(f"Writing buffer ablation logs to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Buffer sizes: {', '.join(str(size) for size in args.buffer_sizes)}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Hover reward changes from {args.hover_reward} to {args.switched_hover_reward} at step {args.hover_switch_step}")
    print(f"Total runs: {total_runs}")
    print(f"Parallel workers: {max_workers}")

    jobs = []
    run_index = 0
    for buffer_size in args.buffer_sizes:
        for seed in seeds:
            run_index += 1
            command = build_command(seed, buffer_size, args, output_dir)
            jobs.append((run_index, seed, buffer_size, command))

    failures = []
    manifest_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                stream_run,
                run_index,
                total_runs,
                seed,
                buffer_size,
                command,
                manifest_path,
                manifest_lock,
            )
            for run_index, seed, buffer_size, command in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["returncode"] != 0:
                failures.append(result)

    if failures:
        print(f"\n{len(failures)} run(s) failed. Logs and manifest are in: {output_dir}")
        for failure in failures:
            print(
                f"FAILED buffer={failure['buffer_size']} seed={failure['seed']} "
                f"returncode={failure['returncode']}"
            )
        raise SystemExit(1)

    print(f"\nAll buffer ablation runs completed. Logs and manifest are in: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run continuous SAC LunarLander hover-switch buffer-size ablation."
    )
    parser.add_argument("--buffer-sizes", nargs="+", type=int, default=BUFFER_SIZES)
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--random-steps", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--hover-reward", type=float, default=200.0)
    parser.add_argument("--hover-switch-step", type=int, default=None)
    parser.add_argument("--switched-hover-reward", type=float, default=-100.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--enable-wind", action="store_true")
    args = parser.parse_args()

    if args.hover_switch_step is None:
        args.hover_switch_step = args.total_steps // 2
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if any(size < 1 for size in args.buffer_sizes):
        parser.error("--buffer-sizes must all be positive")

    return args


if __name__ == "__main__":
    run_all(parse_args())
