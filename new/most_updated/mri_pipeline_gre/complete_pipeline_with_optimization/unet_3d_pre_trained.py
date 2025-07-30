import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=192, patch_size=16, time_steps=8, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.n_patches_per_img = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.n_patches_per_img, embed_dim))
        self.temporal_embed = nn.Parameter(torch.randn(1, time_steps, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, time_steps, H, W = x.shape

        all_patches = []
        for t in range(time_steps):
            x_t = x[:, t].unsqueeze(1)
            patches = self.projection(x_t).flatten(2).transpose(1, 2)
            patches = patches + self.spatial_pos_embed + self.temporal_embed[:, t:t + 1, :]
            all_patches.append(patches)

        x = torch.cat(all_patches, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size=192, patch_size=16, time_steps=8, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, time_steps, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.time_steps = time_steps

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class ConvDecoder(nn.Module):
    def __init__(self, embed_dim=768, img_size=192, patch_size=16, time_steps=8, n_outputs=3, decoder_features=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.time_steps = time_steps
        self.feature_size = img_size // patch_size
        self.upsampling_steps = int(math.log2(patch_size))

        self.patch_to_spatial = nn.Linear(embed_dim, embed_dim)
        self.temporal_fusion = nn.Conv2d(embed_dim * time_steps, embed_dim, kernel_size=1)

        # Build decoder layers with configurable feature sizes
        decoder_layers = []
        in_ch = embed_dim
        for i in range(self.upsampling_steps):
            out_ch = max(decoder_features // (2 ** i), 64)
            decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch

        self.decoder = nn.ModuleList(decoder_layers)
        self.final_conv = nn.Conv2d(in_ch, n_outputs, kernel_size=1)
        self.output_activation = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x[:, 1:]  # Remove class token
        x = self.patch_to_spatial(x)

        # Reshape to spatial-temporal features
        x = x.view(batch_size, self.time_steps, self.feature_size, self.feature_size, self.embed_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, embed_dim, T, H, W]
        x = x.view(batch_size, self.embed_dim * self.time_steps, self.feature_size, self.feature_size)

        x = self.temporal_fusion(x)

        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        x = self.final_conv(x)
        return self.output_activation(x)


class ViTQMRIUNet(nn.Module):
    def __init__(self, img_size=192, patch_size=16, time_steps=8, n_outputs=3,
                 embed_dim=768, depth=12, num_heads=12, decoder_features=512, pretrained=True):
        super().__init__()

        self.encoder = ViTEncoder(img_size, patch_size, time_steps, embed_dim, depth, num_heads)
        self.decoder = ConvDecoder(embed_dim, img_size, patch_size, time_steps, n_outputs, decoder_features)

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        import timm
        pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        pretrained_state = pretrained_vit.state_dict()
        model_state = self.encoder.state_dict()

        transferred = 0
        skipped = 0

        for name, param in model_state.items():
            # More flexible matching for ViT components
            pretrained_name = name

            # Map our layer names to pretrained names
            if 'blocks.' in name and '.attn.' in name:
                # Our MultiheadAttention vs pretrained attention
                if '.in_proj_weight' in pretrained_name or '.out_proj.weight' in pretrained_name:
                    pretrained_name = pretrained_name.replace('.attn.', '.attn.')

            if pretrained_name in pretrained_state:
                pretrained_param = pretrained_state[pretrained_name]
                if param.shape == pretrained_param.shape:
                    param.data.copy_(pretrained_param.data)
                    transferred += 1
                else:
                    # Try to adapt different shaped parameters
                    if 'pos_embed' in name or 'patch_embed' in name:
                        # Skip position embeddings that don't match (different image size/patch size)
                        skipped += 1
                        continue
                    else:
                        skipped += 1
            else:
                skipped += 1

        total_params = len(model_state)
        print(f"Transferred {transferred}/{total_params} pretrained parameters to ViT encoder")
        print(f"Skipped {skipped} parameters due to shape mismatch or new components")

        if transferred < 50:
            print("⚠️  Warning: Very few parameters transferred. This might indicate:")
            print("   - Architecture mismatch with pretrained model")
            print("   - Different patch sizes or embedding dimensions")
            print("   - Custom components not in pretrained model")

    def forward(self, x):
        encoded_features = self.encoder(x)
        output = self.decoder(encoded_features)
        return output


# Model configurations for different tiers
MODEL_CONFIGS = {
    'tiny': {  # ~15M parameters
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'decoder_features': 256,
        'description': 'Tiny model for fast inference and limited compute'
    },
    'small': {  # ~60M parameters
        'embed_dim': 512,
        'depth': 8,
        'num_heads': 8,
        'decoder_features': 384,
        'description': 'Small model for good performance with moderate compute'
    },
    'base': {  # ~90M parameters
        'embed_dim': 768,
        'depth': 10,
        'num_heads': 12,
        'decoder_features': 512,
        'description': 'Base model for high performance'
    },
    'large': {  # ~120M parameters
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'decoder_features': 768,
        'description': 'Large model for maximum performance'
    }
}


def create_vit_qmri_model(time_steps, n_outputs=3, img_size=192, model_size='base', pretrained=True, device='cuda'):
    """
    Create a ViT-based qMRI model with different size configurations

    Args:
        time_steps: Number of time points in your sequence
        n_outputs: Number of output parameter maps (default: 3 for T1, T2, PD)
        img_size: Input image size (default: 192)
        model_size: Model size - 'tiny' (~15M), 'small' (~60M), 'base' (~90M), 'large' (~120M)
        pretrained: Whether to use pretrained ViT weights (only works well with 'base' and 'large')
        device: 'cuda' or 'cpu'

    Returns:
        model: The ViT-UNet model
    """

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_size]

    # Auto-calculate patch size - prefer larger patches to avoid OOM
    patch_size = 16  # Default

    # Try larger patches first to avoid OOM
    for p in [32, 16, 8, 4]:
        if img_size % p == 0 and img_size // p >= 4:  # At least 4x4 feature map
            patch_size = p
            break

    # If no standard size works, find a reasonable factor
    if img_size % patch_size != 0:
        for i in range(min(32, img_size // 4), 3, -1):  # Start from larger patches
            if img_size % i == 0:
                patch_size = i
                break

    n_patches = (img_size // patch_size) ** 2 * time_steps
    print(f"Creating {model_size.upper()} model: {config['description']}")
    print(f"Using patch_size={patch_size} for img_size={img_size}")
    print(f"Total patches per sample: {n_patches} ({img_size // patch_size}×{img_size // patch_size}×{time_steps})")

    # Warn about pretrained weights for non-base models
    if pretrained and model_size not in ['base', 'large']:
        print(f"⚠️  Warning: Pretrained weights may not transfer well to {model_size} model")
        print("   Consider using pretrained=False for tiny/small models")

    model = ViTQMRIUNet(
        img_size=img_size,
        patch_size=patch_size,
        time_steps=time_steps,
        n_outputs=n_outputs,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        decoder_features=config['decoder_features'],
        pretrained=pretrained
    )

    # Display parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size: {total_params * 4 / 1024 ** 2:.1f} MB")

    return model.to(device)


# Example usage
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test all model sizes
    model_sizes = ['tiny', 'small', 'base', 'large']
    img_size = 192

    print("=== Model Size Comparison ===")
    for size in model_sizes:
        print(f"\n--- {size.upper()} MODEL ---")
        model = create_vit_qmri_model(
            time_steps=50,
            n_outputs=3,
            img_size=img_size,
            model_size=size,
            pretrained=False,  # Disable pretrained for comparison
            device=device
        )

        # Test inference
        dummy_input = torch.randn(1, 50, img_size, img_size).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Input: {dummy_input.shape} → Output: {output.shape}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n=== Usage Examples ===")
    print("# Fast inference on mobile/edge devices")
    print("model = create_vit_qmri_model(8, 3, 192, model_size='tiny')")
    print()
    print("# Balanced performance")
    print("model = create_vit_qmri_model(8, 3, 192, model_size='small')")
    print()
    print("# High performance with pretrained weights")
    print("model = create_vit_qmri_model(8, 3, 192, model_size='base', pretrained=True)")
    print()
    print("# Maximum performance for research")
    print("model = create_vit_qmri_model(8, 3, 192, model_size='large', pretrained=True)")