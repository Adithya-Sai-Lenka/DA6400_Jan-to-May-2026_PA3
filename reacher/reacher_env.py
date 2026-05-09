# ============================================================
# reacher_env.py
# ============================================================

import numpy as np

from dm_control import suite


def flatten_obs(obs_dict):

    return np.concatenate([
        np.array(v).ravel()
        for v in obs_dict.values()
    ]).astype(np.float32)


class ReacherEnv:

    def __init__(self,
                 reward_type="rb",
                 seed=0,
                 max_episode_steps=1000,
                 eval_mode=False):

        self.reward_type = reward_type

        self.eval_mode = eval_mode

        self.env = suite.load(
            domain_name="reacher",
            task_name="easy",
            task_kwargs={"random": seed}
        )

        self.max_episode_steps = max_episode_steps

        self.current_step = 0

        self.timeout_count = 0

        ts = self.env.reset()

        obs = flatten_obs(ts.observation)

        self.obs_dim = obs.shape[0]

        action_spec = self.env.action_spec()

        self.action_dim = action_spec.shape[0]

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.current_step = 0

        self.timeout_count = 0

        ts = self.env.reset()

        obs = flatten_obs(ts.observation)

        return obs

    # ========================================================
    # DISTANCE
    # ========================================================

    def get_distance(self):

        vec = self.env.physics.finger_to_target()

        return np.linalg.norm(vec)

    # ========================================================
    # VELOCITY
    # ========================================================

    def get_velocity_mag(self):

        vel = self.env.physics.velocity()

        return np.linalg.norm(vel)

    # ========================================================
    # SUCCESS
    # ========================================================

    def success(self):

        return (
            self.get_distance() < 0.05
            and
            self.get_velocity_mag() < 0.2
        )

    # ========================================================
    # REWARD
    # ========================================================

    def compute_reward(self, action):

        dist = self.get_distance()

        # ----------------------------------------------------
        # Ra
        # ----------------------------------------------------

        if self.reward_type == "ra":

            if self.get_distance() < 0.03:

                return 1.0

            action_penalty = 0.01 * np.square(action).sum()

            return -dist - action_penalty

        # ----------------------------------------------------
        # Rb
        # ----------------------------------------------------

        elif self.reward_type == "rb":

            return 1.0 if self.get_distance() < 0.03 else 0.0

        # ----------------------------------------------------
        # Rc
        # ----------------------------------------------------

        elif self.reward_type == "rc":

            return -1.0

        else:

            raise ValueError("Invalid reward type")

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        self.current_step += 1

        ts = self.env.step(action)

        obs = flatten_obs(ts.observation)

        reward = self.compute_reward(action)

        info = {}

        # ====================================================
        # Ra / Rb
        # ====================================================

        if self.reward_type in ["ra", "rb"]:

            done = self.current_step >= self.max_episode_steps

            return obs, reward, done, info

        # ====================================================
        # Rc
        # ====================================================

        elif self.reward_type == "rc":

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if self.success():

                done = True

                info["effective_length"] = (
                    self.current_step +
                    20 * self.timeout_count
                )

                return obs, reward, done, info

            # ------------------------------------------------
            # TIMEOUT
            # ------------------------------------------------

            if self.current_step >= self.max_episode_steps:

                reward -= 20

                info["timeout"] = True

                # ====================================================
                # EVAL MODE
                # ====================================================

                if self.eval_mode:

                    done = True

                    return obs, reward, done, info

                # ====================================================
                # TRAIN MODE
                # ====================================================

                else:

                    self.timeout_count += 1

                    physics = self.env.physics

                    # ------------------------------------------------
                    # Preserve target position
                    # ------------------------------------------------

                    

                    # ------------------------------------------------
                    # Randomize ONLY arm joint state
                    # ------------------------------------------------

                    with physics.reset_context():

                        physics.named.data.qpos[:] = np.random.uniform(
                            low=-0.5,
                            high=0.5,
                            size=physics.named.data.qpos.shape
                        )

                        physics.named.data.qvel[:] = 0

                    # ------------------------------------------------
                    # Restore target position
                    # ------------------------------------------------

                    

                    obs = flatten_obs(
                        self.env.task.get_observation(physics)
                    )

                    self.current_step = 0

                    done = False

                    return obs, reward, done, info

            done = False

            return obs, reward, done, info