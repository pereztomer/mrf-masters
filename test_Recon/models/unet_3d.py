import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Double 3D convolution block with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pool_kernel=(2, 2, 2)):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(pool_kernel),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super(Up3D, self).__init__()

        # Use trilinear interpolation or transpose conv for upsampling
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle potential size mismatches
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class QMRIUNet3D(nn.Module):
    """
    3D U-Net for quantitative MRI parameter mapping

    Args:
        time_steps: Number of time points in input sequence
        n_outputs: Number of output parameter maps (default: 3 for T1, T2, PD)
        base_features: Number of features in first layer (default: 64)
        trilinear: Use trilinear upsampling instead of transpose convolutions
    """

    def __init__(self, time_steps, n_outputs=3, base_features=64, trilinear=True):
        super(QMRIUNet3D, self).__init__()
        self.time_steps = time_steps
        self.n_outputs = n_outputs
        self.trilinear = trilinear

        # Calculate pooling strategy to avoid temporal dimension becoming 0
        # We want to preserve some temporal dimension until the final aggregation
        temporal_pools = self._calculate_temporal_pooling(time_steps)

        # Encoder path
        self.inc = DoubleConv3D(1, base_features)  # Input: 1 channel
        self.down1 = Down3D(base_features, base_features * 2, pool_kernel=temporal_pools[0])
        self.down2 = Down3D(base_features * 2, base_features * 4, pool_kernel=temporal_pools[1])
        self.down3 = Down3D(base_features * 4, base_features * 8, pool_kernel=temporal_pools[2])

        # Bottleneck
        factor = 2 if trilinear else 1
        self.down4 = Down3D(base_features * 8, base_features * 16 // factor, pool_kernel=temporal_pools[3])

        # Decoder path
        self.up1 = Up3D(base_features * 16, base_features * 8 // factor, trilinear)
        self.up2 = Up3D(base_features * 8, base_features * 4 // factor, trilinear)
        self.up3 = Up3D(base_features * 4, base_features * 2 // factor, trilinear)
        self.up4 = Up3D(base_features * 2, base_features, trilinear)

        # Calculate remaining temporal dimension after all pooling
        remaining_temporal = self._calculate_remaining_temporal(time_steps, temporal_pools)

        # Final temporal aggregation and output
        # This reduces temporal dimension to 1 and expands to n_outputs
        self.temporal_aggregate = nn.Conv3d(base_features, base_features,
                                            kernel_size=(remaining_temporal, 1, 1), padding=(0, 0, 0))
        self.final_conv = nn.Conv3d(base_features, n_outputs, kernel_size=(1, 1, 1))

        # Optional: Add physics-informed output activations
        self.output_activation = nn.ReLU()  # Ensure positive values for T1, T2, PD

    def _calculate_temporal_pooling(self, time_steps):
        """Calculate appropriate pooling kernels to preserve temporal information"""
        pools = []
        current_size = time_steps

        for i in range(4):  # 4 downsampling layers
            if current_size >= 4:
                # Pool temporally
                pools.append((2, 2, 2))
                current_size = current_size // 2
            elif current_size >= 2:
                # Pool less aggressively in temporal dimension
                pools.append((1, 2, 2))
            else:
                # Don't pool temporally anymore
                pools.append((1, 2, 2))

        return pools

    def _calculate_remaining_temporal(self, time_steps, temporal_pools):
        """Calculate remaining temporal dimension after pooling"""
        current_size = time_steps
        for pool in temporal_pools:
            current_size = current_size // pool[0]
        return current_size

    def forward(self, x):
        # Input shape: [batch, time_steps, H, W]
        # Add channel dimension: [batch, 1, time_steps, H, W]
        x = x.unsqueeze(1)

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Temporal aggregation: reduce time dimension to 1
        x = self.temporal_aggregate(x)  # Should output [batch, features, 1, H, W]

        # Final output: [batch, n_outputs, 1, H, W]
        x = self.final_conv(x)

        # Apply activation and squeeze temporal dimension
        x = self.output_activation(x)

        # Ensure temporal dimension is 1 before squeezing
        if x.size(2) != 1:
            # If somehow temporal dimension is not 1, do global average pooling
            x = F.adaptive_avg_pool3d(x, (1, x.size(3), x.size(4)))

        x = x.squeeze(2)  # Remove temporal dimension: [batch, n_outputs, H, W]

        return x


# Example usage and model creation
def create_qmri_model(time_steps, device='cuda'):
    """
    Create a qMRI 3D U-Net model

    Args:
        time_steps: Number of time points in your sequence
        device: 'cuda' or 'cpu'

    Returns:
        model: The 3D U-Net model
    """
    model = QMRIUNet3D(
        time_steps=time_steps,
        n_outputs=3,  # T1, T2, PD
        base_features=64,
        trilinear=True
    )

    model = model.to(device)
    return model


# Example instantiation
if __name__ == "__main__":
    # Example for 8 time steps
    time_steps = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = create_qmri_model(time_steps, device)

    # Test with dummy data
    batch_size = 1
    dummy_input = torch.randn(batch_size, time_steps, 256, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # The output will be [batch_size, 3, 192, 192]
    # Channel 0: T1 map
    # Channel 1: T2 map
    # Channel 2: PD map