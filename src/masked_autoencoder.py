import torch
import torch.nn as nn

from src.patch_embedding import PatchEmbedding


class MaskedEncoder(nn.Module):
    def __init__(
        self, img_size, patch_size, in_chans, out_chans, depth, num_heads, mlp_ratio=4
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, out_chans)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_chans))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, out_chans), requires_grad=False
        )  # fixed sin-cos embedding

        hidden_dim = int(out_chans * mlp_ratio)
        self.layer = nn.TransformerEncoderLayer(out_chans, num_heads, hidden_dim, activation='gelu', norm_first=True)

        self.transformer = nn.TransformerEncoder(
            self.layer, num_layers=depth, norm=nn.LayerNorm(out_chans)
        )

        # self.initialize_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        return x, mask, ids_restore


class MaskedDecoder(nn.Module):
    def __init__(
        self,
        encoder_in_chans,
        in_chans,
        out_chans,
        patch_size,
        num_patches,
        num_heads,
        depth,
        mlp_ratio=4,
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(in_chans, out_chans, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, out_chans))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, out_chans), requires_grad=False
        )  # fixed sin-cos embedding

        hidden_dim = int(out_chans * mlp_ratio)
        self.layer = nn.TransformerEncoderLayer(out_chans, num_heads, hidden_dim, activation='gelu', norm_first=True)
        
        self.transformer = nn.TransformerEncoder(
            self.layer, num_layers=depth, norm=nn.LayerNorm(out_chans)
        )

        self.decoder_pred = nn.Linear(
            out_chans, patch_size**2 * encoder_in_chans, bias=True
        )  # decoder to patch

        # self.initialize_weights()

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.transformer(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    
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
            img_size, patch_size, in_chans, encoder_embed_dim, encoder_depth, encoder_num_heads
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
