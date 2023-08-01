import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

# from src.autoencoder.patch_embedding import PatchEmbedding


class MaskedEncoder(nn.Module):
    def __init__(
        self, img_size, patch_size, in_chans, out_chans, depth, num_heads, mlp_ratio=4
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, out_chans)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_chans))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, out_chans), requires_grad=False
        )  # fixed sin-cos embedding

        # hidden_dim = int(out_chans * mlp_ratio)
        # self.layer = nn.TransformerEncoderLayer(out_chans, num_heads, hidden_dim, activation='gelu', norm_first=True)

        # self.transformer = nn.TransformerEncoder(
        #     self.layer, num_layers=depth, norm=nn.LayerNorm(out_chans)
        # )

        self.blocks = nn.ModuleList(
            [
                Block(
                    out_chans,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(out_chans)

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

    def forward(self, x, mask_ratio=0.60):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # x = self.transformer(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
