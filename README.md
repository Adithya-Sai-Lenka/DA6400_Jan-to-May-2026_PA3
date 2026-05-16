# RL_PA3

## Setup

```bash
conda env create -f RL_PA3.yml
conda activate RL_PA3
```

## Pendulum Experiments

To generate plots for section 2.1, run the following scripts sequentially:

```bash
python SAC_pendulum_automated_temp_tuning.py
python SAC_pendulum_manual_temp_tuning.py
python SAC_pendulum_scaled_rewards.py
python plot_pendulum.py
```

## Lunar Lander

The Lunar Lander code supports both running a single training job directly and running the full multi-seed experiment suite used for the report.

To run all section 2.2 Lunar Lander experiments across multiple seeds, use:

```bash
python run_lunar_multiseed.py
```

This launches the following experiments and writes CSV logs to a timestamped directory under `logs_multiseed/`:

- Continuous SAC
- Continuous SAC with hover reward switch and automatic alpha tuning
- Continuous SAC with hover reward switch and fixed alpha
- Discrete SAC
- DQN

If you want to run only a subset of experiments, you can pass them explicitly. For example:

```bash
python run_lunar_multiseed.py --experiments continuous-default hover-auto discrete-sac
```

After the runs finish, generate the Lunar Lander plots and summary CSV with:

```bash
python plot_lunar_multiseed.py
```

By default, the plotting script reads the fullest run inside `logs_multiseed/` and saves outputs to `plots/lunar_lander/`.

For the replay-buffer-size ablation in the changing hover-reward setting, run:

```bash
python run_lunar_buffer_ablation.py
```

This writes per-buffer logs to a timestamped directory under `logs_buffer_ablation/`.

To generate the corresponding ablation plot and summary tables, run:

```bash
python plot_lunar_buffer_ablation.py
```

This saves the plot and CSV/Markdown summaries to `plots/lunar_lander_buffer_size/`.

If you want to train a single configuration manually instead of using the wrappers, the main entry points are:

```bash
python SAC_lunar_lander.py --train
python SAC_lunar_lander_discrete.py --algo discrete-sac
python SAC_lunar_lander_discrete.py --algo dqn
```

`SAC_lunar_lander.py` supports:

```bash
python SAC_lunar_lander.py \
  --summary \
  --train \
  --seed 1 \
  --total-steps 500000 \
  --random-steps 10000 \
  --replay-capacity 100000 \
  --eval-freq 5000 \
  --eval-episodes 10 \
  --fixed-alpha 0.01 \
  --hover-reward 200 \
  --hover-switch-step 250000 \
  --switched-hover-reward -100 \
  --enable-wind \
  --log-dir logs
```

Notes:

- `--summary` prints the environment summary and exits.
- `--train` is required for training.
- Omit `--fixed-alpha` to use automatic entropy tuning.
- Omit `--hover-reward` and `--hover-switch-step` for the default continuous SAC setting.

`SAC_lunar_lander_discrete.py` supports:

```bash
python SAC_lunar_lander_discrete.py \
  --algo discrete-sac \
  --seed 1 \
  --total-steps 500000 \
  --random-steps 10000 \
  --replay-capacity 100000 \
  --eval-freq 5000 \
  --eval-episodes 10 \
  --lr 3e-4 \
  --dqn-lr 1e-4 \
  --grad-clip-norm 10.0 \
  --target-entropy-ratio 0.5 \
  --min-alpha 1e-4 \
  --max-alpha 1.0 \
  --enable-wind \
  --log-dir logs
```

Notes:

- `--algo` can be `discrete-sac`, `dqn`, or `both`.
- `--lr`, `--target-entropy-ratio`, `--min-alpha`, and `--max-alpha` are used by discrete SAC.
- `--dqn-lr` and `--grad-clip-norm` are used by DQN.


## Reacher

The Reacher implementation supports SAC training using:
- `R_a` — dense shaping reward,
- `R_b` — occupancy reward,
- `R_c` — sparse reset-style reward.
Main files:
- `reacher_env.py` — environment and reward formulations,
- `sac.py` — SAC implementation,
- `replay_buffer.py` — replay buffer,
- `train.py` — SAC training,
- `run_parallel.py` — parallel multi-seed training,
- `plot.py` — training curve generation,
- `evaluate_behaviour.py` — behavioral evaluation,
- `seed_wise_variation_episode_length.py` — `R_c` seed difficulty analysis,
- `render.py` — policy rendering.

Move into the reacher directory:
```bash
cd reacher
```
Single-seed training:
```bash
python train.py
```
Modify inside train.py for required reward for single seed training.
```bash
train(
    reward_type="rb",
    seed=0
)
```
Parallel multi-seed training:
Modify inside `run_parallel.py` for desired SAC-$R_i$ training where $i \in \{a,b,c\}$:
```bash
reward_type = "ra"
```
Run:
```bash
python run_parallel.py
```
Generate cross-evaluation plots:
```bash
python plot.py --train_reward a --eval_reward b
```
Arguments:

--train_reward: a, b, c
--eval_reward: a, b, c

Behavioral evaluation:
```bash
python evaluate_behaviour.py
```
This generates:

```text
logs/steps_to_goal_bar.png
logs/steps_in_target_bar.png
```
Analyze R_c exploration difficulty across seeds:
```bash
python seed_wise_variation_episode_length.py
```
Render trained policies:
```bash
python render.py --reward_type rc --seed 0
```
Supported Arguments:

--reward_type: ra, rb, rc
--seed: training seed
--steps: rollout length (default: 1000)

## Bonus

### Bonus Pendulum Experiments

To run the PEBBLE pendulum experiments, execute:

```bash
python PEBBLE_pendulum.py
```

After the experiments complete, generate visualizations with:

```bash
python plot_pendulum_PEBBLE.py
```