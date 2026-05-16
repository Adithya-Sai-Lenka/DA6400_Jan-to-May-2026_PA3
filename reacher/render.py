# render.py
'''
```

Run example:

```bash
python render_rc.py --reward_type rc --seed 0
```

This will create:

```text
logs/rc_seed_0_render.mp4
```

You can SCP/download the MP4 from the HPC and play locally.
'''
import os

os.environ["MUJOCO_GL"] = "osmesa"

import cv2
import argparse
import numpy as np
import torch

from sac import SACAgent
from reacher_env import ReacherEnv


parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    required=True
)

parser.add_argument(
    "--steps",
    type=int,
    default=1000
)
parser.add_argument(
    "--reward_type",
    type=str,
    required=True,
    choices=["ra", "rb", "rc"]
)
args = parser.parse_args()
reward_type = args.reward_type
args = parser.parse_args()
# reward_type = "rc" # Change as required (ra/rb/rc)
seed = args.seed


env = ReacherEnv(
    reward_type=reward_type,
    seed=seed,
    eval_mode=True
)


agent = SACAgent(
    env.obs_dim,
    env.action_dim,
    device="cpu"
)


checkpoint_path = (
    f"logs/{reward_type}_seed_{seed}_actor.pt"
)


agent.actor.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location="cpu"
    )
)

agent.actor.eval()


obs = env.reset()


frames = []


episode_return = 0


for step in range(args.steps):

    frame = env.env.physics.render(
        height=480,
        width=480,
        camera_id=0
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    frames.append(frame)

    action = agent.select_action(
        obs,
        deterministic=True
    )

    obs, reward, done, info = env.step(action)

    episode_return += reward

    if done and "success_step" not in locals():

        success_step = step

        print(f"Success triggered at step {step}")

    # Continue rendering 150 more steps after success
    if "success_step" in locals():

        if step >= success_step + 150:

            print(f"Stopping render at step {step}")

            break


save_path = f"logs/{reward_type}_seed_{seed}_render.mp4"


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    save_path,
    fourcc,
    30,
    (480, 480)
)


for frame in frames:

    video.write(frame)


video.release()


print(f"Saved video to: {save_path}")
print(f"Episode return: {episode_return}")