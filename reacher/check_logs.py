import numpy as np

logs = np.load(
    "logs/ra_seed_6.npy",
    allow_pickle=True
).item()

print(logs.keys())

print()

for k, v in logs.items():

    if isinstance(v, list):

        print(k, np.array(v).shape)

    else:

        print(k, type(v))