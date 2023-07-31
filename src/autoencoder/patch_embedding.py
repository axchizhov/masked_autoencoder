import torch
import torch.nn as nn


def positional_encoding(sequence_len, output_dim, n=10000):
    # Initialize the position encoding matrix with zeros
    P = torch.zeros((sequence_len, output_dim))

    # Iterate through the positions in the sequence
    for k in range(sequence_len):
        # Iterate through the dimensions of the encoding
        for i in range(
            0, output_dim, 2
        ):  # Increment by 2 to handle both sine and cosine parts
            denominator = torch.tensor(n, dtype=torch.float).pow(2 * i / output_dim)
            P[k, i] = torch.sin(k / denominator)
            P[k, i + 1] = torch.cos(k / denominator)

    return P


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # TODO: remove
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate the total number of patches in the image
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # Define the patch embedding layer using a convolutional layer
        # The kernel size is set to the patch size and the stride is also set to the patch size
        # This results in non-overlapping patches of the image being processed
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
            nn.Flatten(1),
        )

        # Create positional encoding of the size to be concatenated with the patches
        self.pos_encoding = positional_encoding(
            sequence_len=self.num_patches, output_dim=embed_dim, n=1000
        ).to(device)

        # Define the class token, which will be added to the sequence of patch embeddings
        # The class token is learnable and will be used to predict the class of the image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        # Get the batch size, channels, height, and width of the input image
        B, C, H, W = x.shape

        # Pass the image through the patch embedding layer
        x = self.patch_embedding(x)

        # Reshape the output of the patch embedding layer
        # Each row now represents a flattened patch of the image
        x = x.view(B, self.num_patches, -1)

        # # Create a batch of class tokens by expanding the class token tensor
        # cls_tokens = self.cls_token.expand(B, -1, -1)

        # # Add the positional encoding to each patch
        # x = torch.cat([self.pos_encoding.expand(B, -1, -1), x], dim=2)

        # # Concatenate the class tokens with the patch embeddings
        # x = torch.cat([cls_tokens, x], dim=1)

        return x
