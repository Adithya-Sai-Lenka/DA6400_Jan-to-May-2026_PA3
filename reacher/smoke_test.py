from reacher_env import ReacherEnv
import numpy as np
env = ReacherEnv("rb")

obs = env.reset()

for _ in range(1000):

    action = np.random.uniform(-1,1,size=2)

    _, reward, _, _ = env.step(action)

    if reward > 0:
        print("TARGET REACHED")