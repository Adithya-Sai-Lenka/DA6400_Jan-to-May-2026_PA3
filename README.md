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



## Reacher



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

### Bonus Reacher Experiments