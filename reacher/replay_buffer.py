# ============================================================
# replay_buffer.py
# ============================================================

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self,
                 obs_dim,
                 action_dim,
                 capacity=1_000_000,
                 device="cpu"):

        self.capacity = capacity

        self.device = device

        self.obses = np.empty(
            (capacity, obs_dim),
            dtype=np.float32
        )

        self.next_obses = np.empty(
            (capacity, obs_dim),
            dtype=np.float32
        )

        self.actions = np.empty(
            (capacity, action_dim),
            dtype=np.float32
        )

        self.rewards = np.empty(
            (capacity, 1),
            dtype=np.float32
        )

        self.not_dones = np.empty(
            (capacity, 1),
            dtype=np.float32
        )

        self.idx = 0

        self.full = False

    def add(self,
            obs,
            action,
            reward,
            next_obs,
            done):

        self.obses[self.idx] = obs

        self.actions[self.idx] = action

        self.rewards[self.idx] = reward

        self.next_obses[self.idx] = next_obs

        self.not_dones[self.idx] = 1.0 - done

        self.idx = (self.idx + 1) % self.capacity

        self.full = self.full or self.idx == 0

    def sample(self, batch_size):

        max_idx = self.capacity if self.full else self.idx

        idxs = np.random.randint(
            0,
            max_idx,
            size=batch_size
        )

        return (
            torch.FloatTensor(self.obses[idxs]).to(self.device),
            torch.FloatTensor(self.actions[idxs]).to(self.device),
            torch.FloatTensor(self.rewards[idxs]).to(self.device),
            torch.FloatTensor(self.next_obses[idxs]).to(self.device),
            torch.FloatTensor(self.not_dones[idxs]).to(self.device)
        )