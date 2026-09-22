# Author: Anubhav Joshi
import random
import numpy as np
import torch

## Constants
IMDB_DATASET_NAME = "stanfordnlp/imdb"
NUM_LABELS = 2

## Set Seed
"""
Since our objective of this project is to compare 5 different methods with each other,
any difference in results should be due to the methods themselves and not due to randomness in the training process.
So envery run need to start from the same seed.
"""
def set_seed(seed = int(42)):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


## Get Device
"""
Device selections
"""
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

## Quantized model check
def is_quantized(model):
    """
    True for models loaded in 4-bit/8-bit via bitsandbytes (e.g. the qlora strategy).
    Their weights are pinned to a device at load time.
    """
    return getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)