import numpy as np
import torch
import random
import torchinfo


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_info(model):
    input_size = [1, 3, 32, 32]
    col_names=['output_size', 'num_params', 'trainable']
    torchinfo.summary(model, input_size, depth=2)
