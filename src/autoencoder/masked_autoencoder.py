import torch
import torch.nn as nn

from src.autoencoder.decoder import MaskedDecoder
from src.autoencoder.encoder import MaskedEncoder


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_chans=3,
        encoder_embed_dim=128,
        encoder_num_heads=16,
        encoder_depth=6,
        decoder_embed_dim=512,
        decoder_num_heads=4,
        decoder_depth=2,
    ):
        super().__init__()

        self.encoder = MaskedEncoder(
            img_size,
            patch_size,
            in_chans,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
        )

        num_patches = self.encoder.patch_embed.num_patches
        self.decoder = MaskedDecoder(
            in_chans,
            encoder_embed_dim,
            decoder_embed_dim,
            patch_size,
            num_patches,
            decoder_num_heads,
            decoder_depth,
        )

    def patchify(self, images):
        """
        images: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder.patch_embed.patch_size
        assert images.shape[2] == images.shape[3] and images.shape[2] % p == 0

        h = w = images.shape[2] // p
        x = images.reshape(shape=(images.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(images.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        images: (N, 3, H, W)
        """
        p = self.encoder.patch_embed.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        images = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return images

    def calculate_loss(self, images, pred, mask):
        """
        images: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(images)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, images):
        latent, mask, ids_restore = self.encoder(images)
        pred = self.decoder(latent, ids_restore)  # [N, L, p*p*3]

        loss = self.calculate_loss(images, pred, mask)
        return loss, pred, mask
