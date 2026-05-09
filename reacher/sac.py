# ============================================================
# sac.py
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal


def weight_init(m):

    if isinstance(m, nn.Linear):

        nn.init.orthogonal_(m.weight)

        nn.init.constant_(m.bias, 0)


class DoubleQCritic(nn.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim=256):

        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weight_init)

    def forward(self, obs, action):

        sa = torch.cat([obs, action], dim=-1)

        return self.q1(sa), self.q2(sa)


class SquashedGaussianActor(nn.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim=256,
                 log_std_bounds=(-20, 2)):

        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim, action_dim)

        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.apply(weight_init)

    def forward(self,
                obs,
                deterministic=False):

        x = self.net(obs)

        mu = self.mu(x)

        if deterministic:

            return torch.tanh(mu), None

        log_std = self.log_std(x)

        log_std = torch.clamp(
            log_std,
            self.log_std_bounds[0],
            self.log_std_bounds[1]
        )

        std = log_std.exp()

        dist = Normal(mu, std)

        x_t = dist.rsample()

        y_t = torch.tanh(x_t)

        action = y_t

        log_prob = dist.log_prob(x_t)

        log_prob -= torch.log(
            1 - y_t.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob


class SACAgent:

    def __init__(self,
                 obs_dim,
                 action_dim,
                 device="cpu",
                 gamma=0.99,
                 tau=0.005,
                 lr=3e-4):

        self.device = device

        self.gamma = gamma

        self.tau = tau

        self.actor = SquashedGaussianActor(
            obs_dim,
            action_dim
        ).to(device)

        self.critic = DoubleQCritic(
            obs_dim,
            action_dim
        ).to(device)

        self.critic_target = DoubleQCritic(
            obs_dim,
            action_dim
        ).to(device)

        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=lr
        )

        self.target_entropy = -action_dim

        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=device
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=lr
        )

    @property
    def alpha(self):

        return self.log_alpha.exp()

    def select_action(self,
                      obs,
                      deterministic=False):

        obs = torch.FloatTensor(
            obs
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            action, _ = self.actor(
                obs,
                deterministic=deterministic
            )

        return action.cpu().numpy()[0]

    def update(self,
               replay_buffer,
               batch_size=256):

        obs, action, reward, next_obs, not_done = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():

            next_action, next_log_prob = \
                self.actor(next_obs)

            target_q1, target_q2 = \
                self.critic_target(
                    next_obs,
                    next_action
                )

            target_v = torch.min(
                target_q1,
                target_q2
            ) - self.alpha.detach() * next_log_prob

            target_q = reward + \
                       self.gamma * not_done * target_v

        current_q1, current_q2 = \
            self.critic(obs, action)

        critic_loss = \
            F.mse_loss(current_q1, target_q) + \
            F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        pi, log_prob = self.actor(obs)

        q1_pi, q2_pi = self.critic(obs, pi)

        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (
            self.alpha.detach() * log_prob - q_pi
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha *
            (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()

        alpha_loss.backward()

        self.alpha_optimizer.step()

        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):

            target_param.data.copy_(
                self.tau * param.data +
                (1 - self.tau) * target_param.data
            )