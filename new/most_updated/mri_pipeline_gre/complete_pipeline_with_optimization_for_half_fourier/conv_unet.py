import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(7, 3, 3)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(Conv3DBlock(in_ch, out_ch), Conv3DBlock(out_ch, out_ch))

    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, pool_size=(2, 1, 1)):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(pool_size),
            DoubleConv3D(in_ch, out_ch)
        )

    def forward(self, x):
        return self.mpconv(x)


class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=(2, 1, 1), bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=scale_factor, stride=scale_factor)
            self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2, diff_d // 2,
                        diff_d - diff_d // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, features=[64, 128, 256, 512], bilinear=True):
        super().__init__()
        self.inc = DoubleConv3D(in_ch, features[0])

        # Asymmetric downsampling - aggressive in time, conservative in space
        self.down1 = Down3D(features[0], features[1], pool_size=(2, 1, 1))  # 50→25, 36→36
        self.down2 = Down3D(features[1], features[2], pool_size=(2, 2, 2))  # 25→12, 36→18
        self.down3 = Down3D(features[2], features[3], pool_size=(3, 1, 1))  # 12→4,  18→18

        factor = 2 if bilinear else 1
        self.down4 = Down3D(features[3], features[3] * 2 // factor,
                            pool_size=(2, 1, 1))  # 6→3, 18→18 (bottleneck: 3x18x18)

        # Asymmetric upsampling - restore time, preserve space
        self.up1 = Up3D(features[3] * 2, features[3] // factor, scale_factor=(2, 1, 1), bilinear=bilinear)  # 3→6, 18→18
        self.up2 = Up3D(features[3], features[2] // factor, scale_factor=(2, 1, 1), bilinear=bilinear)  # 6→12, 18→18
        self.up3 = Up3D(features[2], features[1] // factor, scale_factor=(2, 2, 2), bilinear=bilinear)  # 12→25, 18→36
        self.up4 = Up3D(features[1], features[0], scale_factor=(2, 1, 1), bilinear=bilinear)  # 25→50, 36→36

        self.outc = nn.Conv3d(features[0], out_ch, 1)
        self.final_relu = nn.ReLU(inplace=True)  # Added final ReLU

        # Global average pooling over time dimension to get single parameter maps
        self.global_pool = nn.AdaptiveAvgPool3d((1, None, None))

    def forward(self, x):
        # Input x: [batch, time, height, width] -> need [batch, channels, time, height, width]
        x = x.unsqueeze(1)  # Add channel dimension: [batch, 1, time, height, width]

        x1 = self.inc(x)  # [B, F0, 50, 36, 36]
        x2 = self.down1(x1)  # [B, F1, 25, 36, 36]
        x3 = self.down2(x2)  # [B, F2, 12, 36, 36]
        x4 = self.down3(x3)  # [B, F3, 4,  36, 36]
        x5 = self.down4(x4)  # [B, F4, 4,  36, 36] - bottleneck: 4x36x36

        x = self.up1(x5, x4)  # [B, F3, 4,  36, 36]
        x = self.up2(x, x3)  # [B, F2, 12, 36, 36]
        x = self.up3(x, x2)  # [B, F1, 25, 36, 36]
        x = self.up4(x, x1)  # [B, F0, 50, 36, 36]

        x = self.outc(x)  # [B, out_ch, 50, 36, 36]
        x = self.final_relu(x)  # Apply ReLU to ensure positive outputs

        # Pool over time dimension to get single parameter maps
        x = self.global_pool(x)  # [B, out_ch, 1, 36, 36]
        x = x.squeeze(2)  # [B, out_ch, 36, 36]

        return x


def create_3d_unet_mri(time_steps, input_shape, out_channels, model_size="medium"):
    """
    Create 3D U-Net for MRI parameter mapping

    Args:
        time_steps: Number of time steps (time dimension)
        input_shape: Tuple of (height, width) for spatial dimensions
        out_channels: Number of output channels (e.g., 3 for T1, T2, PD)
        model_size: Model size - "tiny", "small", "medium", "large", "huge"
    """
    configs = {
        "tiny": {"features": [16, 32, 64, 128], "bilinear": True},
        "small": {"features": [24, 48, 96, 192], "bilinear": True},
        "medium": {"features": [32, 64, 128, 256], "bilinear": False},
        "large": {"features": [48, 96, 192, 384], "bilinear": False},
        "huge": {"features": [64, 128, 256, 512], "bilinear": False}
    }

    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")

    config = configs[model_size]
    return UNet3D(in_ch=1, out_ch=out_channels, features=config["features"], bilinear=config["bilinear"])


# Usage example
if __name__ == "__main__":
    # Create model with your specifications
    model = create_3d_unet_mri(
        time_steps=50,  # Your time dimension
        input_shape=(36, 36),  # Your spatial dimensions
        out_channels=3,  # T1, T2, PD outputs
        model_size="medium"  # Options: "tiny", "small", "medium", "large", "huge"
    )

    # Test with your exact input shape
    x = torch.randn(1, 50, 36, 36)  # [batch, time, height, width]
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")  # Should be [1, 3, 36, 36]

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.1f}M)")