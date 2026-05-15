import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# SEGMENT SAMPLING
# ============================================================


def sample_segment(trajectory,
                   segment_length=25):

    start = np.random.randint(
        0,
        len(trajectory) - segment_length
    )

    return trajectory[start:start + segment_length]


# ============================================================
# SIMULATED TEACHER
# ============================================================
def teacher_preference(seg1,
                       seg2):

    r1 = sum([x[2] for x in seg1])

    r2 = sum([x[2] for x in seg2])

    if r1 >= r2:

        return 0

    return 1


# ============================================================
# REWARD MODEL LOSS
# ============================================================


def preference_loss(reward_model,
                    batch,
                    device):

    logits = []

    labels = []

    for seg1, seg2, label in batch:

        r1 = 0
        r2 = 0

        for obs, action, reward, next_obs, done in seg1:

            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

            action = torch.FloatTensor(action).unsqueeze(0).to(device)

            r1 += reward_model(obs, action)
        for obs, action, reward, next_obs, done in seg2:

            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

            action = torch.FloatTensor(action).unsqueeze(0).to(device)

            r2 += reward_model(obs, action)

        logits.append(torch.cat([r1, r2], dim=1))

        labels.append(label)

    logits = torch.cat(logits, dim=0)

    labels = torch.LongTensor(labels).to(device)

    return F.cross_entropy(logits, labels)