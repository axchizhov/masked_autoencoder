import pytest
import torch
from src.autoencoder_v2.model import MAE_ViT


def test_autoencoder():
    net = MAE_ViT(image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 )

    x = torch.zeros((10, 3, 32, 32))
    
    predicted_img, mask = net(x)
    
    assert list(predicted_img.shape) == [10, 3, 32, 32]
    assert list(mask.shape) == [10, 3, 32, 32]
