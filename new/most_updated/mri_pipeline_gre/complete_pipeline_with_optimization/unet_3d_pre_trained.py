import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class PatchEmbedding(nn.Module):
    """
    Convert image patches and time steps into embeddings
    Treats temporal dimension as additional patches
    """

    def __init__(self, img_size=192, patch_size=16, time_steps=8, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate number of patches per image
        self.n_patches_per_img = (img_size // patch_size) ** 2
        self.total_patches = self.n_patches_per_img * time_steps

        # Projection layer: flatten patches and project to embed_dim
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable position embeddings for spatial patches
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.n_patches_per_img, embed_dim))

        # Learnable temporal embeddings for each time step
        self.temporal_embed = nn.Parameter(torch.randn(1, time_steps, embed_dim))

        # Class token (optional, can be used for global representation)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Input: [batch, time_steps, H, W]
        batch_size, time_steps, H, W = x.shape

        # Process each time step separately
        all_patches = []
        for t in range(time_steps):
            # Get single time step: [batch, H, W] -> [batch, 1, H, W]
            x_t = x[:, t].unsqueeze(1)

            # Create patches: [batch, embed_dim, n_patches_h, n_patches_w]
            patches = self.projection(x_t)

            # Flatten spatial dimensions: [batch, embed_dim, n_patches]
            patches = patches.flatten(2)

            # Transpose: [batch, n_patches, embed_dim]
            patches = patches.transpose(1, 2)

            # Add spatial position embeddings
            patches = patches + self.spatial_pos_embed

            # Add temporal embedding for this time step
            patches = patches + self.temporal_embed[:, t:t + 1, :]

            all_patches.append(patches)

        # Concatenate all time steps: [batch, total_patches, embed_dim]
        x = torch.cat(all_patches, dim=1)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for temporal MRI data"""

    def __init__(self, img_size=192, patch_size=16, time_steps=8, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, time_steps, 1, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Store dimensions for decoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.time_steps = time_steps
        self.n_patches_per_img = self.patch_embed.n_patches_per_img

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [batch, 1 + total_patches, embed_dim]

        # Store intermediate features for skip connections
        features = []

        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Store features at different depths for skip connections
            if i in [2, 5, 8, 11]:  # Arbitrary depths for skip connections
                features.append(x)

        x = self.norm(x)

        return x, features


class ConvDecoder(nn.Module):
    """Convolutional decoder to reconstruct spatial maps"""

    def __init__(self, embed_dim=768, img_size=192, patch_size=16, time_steps=8, n_outputs=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.n_patches_per_img = (img_size // patch_size) ** 2

        # Calculate feature map size after patch embedding
        self.feature_size = img_size // patch_size  # 192//16 = 12

        # Calculate how many upsampling steps we need
        # From 12x12 to 192x192 = 16x upsampling = 4 steps of 2x each
        self.upsampling_steps = int(math.log2(patch_size))  # log2(16) = 4

        # Projection from transformer features back to spatial
        self.patch_to_spatial = nn.Linear(embed_dim, embed_dim)

        # Temporal aggregation - combine features from all time steps
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * time_steps, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Progressive upsampling decoder
        decoder_layers = []
        in_channels = embed_dim
        out_channels = 512

        for i in range(self.upsampling_steps):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels
            out_channels = max(out_channels // 2, 64)  # Reduce channels progressively

        self.decoder = nn.ModuleList(decoder_layers)

        # Final output layer
        self.final_conv = nn.Conv2d(in_channels, n_outputs, kernel_size=1)
        self.output_activation = nn.ReLU()  # Ensure positive values for T1, T2, PD

    def forward(self, x):
        # x shape: [batch, 1 + total_patches, embed_dim]
        batch_size = x.shape[0]

        # Remove class token
        x = x[:, 1:]  # [batch, total_patches, embed_dim]

        # Project back to spatial
        x = self.patch_to_spatial(x)

        # Reshape to separate time steps and spatial patches
        # [batch, time_steps * n_patches_per_img, embed_dim]
        x = x.view(batch_size, self.time_steps, self.n_patches_per_img, self.embed_dim)

        # Reshape each time step to spatial grid
        temporal_features = []
        for t in range(self.time_steps):
            # [batch, n_patches_per_img, embed_dim] -> [batch, embed_dim, feature_size, feature_size]
            x_t = x[:, t].transpose(1, 2).view(batch_size, self.embed_dim,
                                               self.feature_size, self.feature_size)
            temporal_features.append(x_t)

        # Concatenate temporal features
        x = torch.cat(temporal_features, dim=1)  # [batch, embed_dim * time_steps, H, W]

        # Temporal fusion
        x = self.temporal_fusion(x)  # [batch, embed_dim, H, W]

        # Apply decoder layers
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        # Final output
        x = self.final_conv(x)
        x = self.output_activation(x)

        return x


class ViTQMRIUNet(nn.Module):
    """
    Vision Transformer U-Net for quantitative MRI parameter mapping

    Args:
        img_size: Input image size (default: 192)
        patch_size: Patch size for ViT (default: 16)
        time_steps: Number of time points
        embed_dim: Transformer embedding dimension (default: 768)
        depth: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 12)
        n_outputs: Number of output parameter maps (default: 3 for T1, T2, PD)
        pretrained: Whether to load pretrained ViT weights
    """

    def __init__(self, img_size=192, patch_size=16, time_steps=8, embed_dim=768,
                 depth=12, num_heads=12, n_outputs=3, pretrained=True):
        super().__init__()

        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            time_steps=time_steps,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )

        self.decoder = ConvDecoder(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            time_steps=time_steps,
            n_outputs=n_outputs
        )

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load pretrained ViT weights (ImageNet or other)"""
        try:
            # You can load from torchvision or timm
            import timm
            pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)

            # Transfer weights that match
            pretrained_state = pretrained_vit.state_dict()
            model_state = self.encoder.state_dict()

            # Transfer compatible weights
            transferred = 0
            for name, param in model_state.items():
                if name in pretrained_state and param.shape == pretrained_state[name].shape:
                    param.data.copy_(pretrained_state[name].data)
                    transferred += 1

            print(f"Transferred {transferred} pretrained parameters to ViT encoder")

        except ImportError:
            print("timm not available, skipping pretrained weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(self, x):
        # Encode with ViT
        encoded_features, skip_features = self.encoder(x)

        # Decode to parameter maps
        output = self.decoder(encoded_features)

        return output


# Factory function
def create_vit_qmri_model(time_steps, n_outputs, img_size=192, pretrained=True, device='cuda'):
    """
    Create a ViT-based qMRI model

    Args:
        time_steps: Number of time points in your sequence
        n_outputs: Number of output parameter maps (default: 3 for T1, T2, PD)
                  Common configurations:
                  - 3: T1, T2, PD
                  - 4: T1, T2, PD, T2*
                  - 5: T1, T2, PD, T2*, B0
                  - 2: T1, T2 (for T1/T2 mapping only)
                  - 1: Single parameter (e.g., T1-only mapping)
        img_size: Input image size (default: 192)
        pretrained: Whether to use pretrained ViT weights
        device: 'cuda' or 'cpu'

    Returns:
        model: The ViT-UNet model
    """
    model = ViTQMRIUNet(
        img_size=img_size,
        patch_size=16,
        time_steps=time_steps,
        embed_dim=768,
        depth=12,
        num_heads=12,
        n_outputs=n_outputs,
        pretrained=pretrained
    )

    model = model.to(device)
    return model


# Example usage
if __name__ == "__main__":
    # Example for 8 time steps
    time_steps = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_vit_qmri_model(time_steps=8, n_outputs=3, pretrained=True)

    # Test with dummy data
    batch_size = 1
    dummy_input = torch.randn(batch_size, time_steps, 192, 192).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # The output will be [batch_size, 3, 192, 192]
    # Channel 0: T1 map
    # Channel 1: T2 map
    # Channel 2: PD map