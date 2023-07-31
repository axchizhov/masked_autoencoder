import pytest
import torch
from src.masked_autoencoder import MaskedAutoencoder


def test_autoencoder():
    net = MaskedAutoencoder()

    x = torch.zeros((10, 3, 32, 32))

    loss, pred, mask = net(x)

    assert list(pred.shape) == [10, 16, 192]