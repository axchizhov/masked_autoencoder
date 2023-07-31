import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


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
        # x = self.transformer(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
