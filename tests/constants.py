import torch

BATCH_SIZE = 3
SPARSE_DIM_1 = 6
SPARSE_DIM_2 = 4
SPARSE_DIM_3 = 4
EMBED_DIM = 16
N_HEADS = 2
N_KEYS_PER_QUERY = 5

POSITION_DIM = 3
N_FREQ_GROUPS = 2


ALWAYS_SPECIFIED = torch.tensor(
    [
        [0, 0, 0, 0],  # Origin in first batch
        [1, 1, 1, 1],  # Diagonal in second batch
        [2, 2, 2, 2],  # Diagonal in third batch
    ],
    dtype=torch.long,
)

ALWAYS_UNSPECIFIED = torch.tensor(
    [
        [0, 5, 3, 3],  # Near corner in first batch
        [1, 5, 3, 3],  # Same position in second batch
        [2, 5, 3, 3],  # Same position in third batch
    ],
    dtype=torch.long,
)
