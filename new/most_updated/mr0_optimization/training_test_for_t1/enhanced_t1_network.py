"""
Enhanced T1 Mapping Network using Pretrained ResNet50 with improved architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Channel and spatial attention block"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa
        
        return x


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction block"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        
        # Different kernel sizes for multi-scale features
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch2 = self.conv3x3(x)
        branch3 = self.conv5x5(x)
        branch4 = self.conv7x7(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class EnhancedT1MappingNet(nn.Module):
    def __init__(self, input_channels, output_channels=1, backbone='resnet50'):
        super(EnhancedT1MappingNet, self).__init__()
        
        # Choose backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            feature_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            feature_channels = [64, 256, 512, 1024, 2048]
        else:  # resnet34
            resnet = models.resnet34(pretrained=True)
            feature_channels = [64, 64, 128, 256, 512]
        
        # Input processing for magnitude and phase
        self.magnitude_conv = nn.Sequential(
            nn.Conv2d(input_channels // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.phase_conv = nn.Sequential(
            nn.Conv2d(input_channels // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Fusion of magnitude and phase features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MultiScaleBlock(64, 64),
            AttentionBlock(64)
        )
        
        # Modified first conv for fused features
        self.conv1 = nn.Conv2d(64, feature_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy pretrained weights and adapt
        with torch.no_grad():
            if hasattr(resnet.conv1, 'weight'):
                # Average the RGB weights and replicate
                avg_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
                self.conv1.weight = nn.Parameter(avg_weight.repeat(1, 64, 1, 1))
        
        # Use pretrained ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Enhanced decoder with skip connections and attention
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels[4], feature_channels[3], 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels[3]),
            nn.ReLU(),
            AttentionBlock(feature_channels[3])
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels[3] * 2, feature_channels[2], 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels[2]),
            nn.ReLU(),
            AttentionBlock(feature_channels[2])
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels[2] * 2, feature_channels[1], 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_channels[1]),
            nn.ReLU(),
            AttentionBlock(feature_channels[1])
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(feature_channels[1] * 2, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            AttentionBlock(128)
        )
        
        # Final output layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),  # +64 from fusion features
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MultiScaleBlock(64, 64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, 1),
            nn.Sigmoid()  # Force values to [0, 1] range
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Handle small inputs - upsample to avoid 1x1 feature maps
        original_size = x.shape[-2:]
        if x.shape[-1] <= 32:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            need_resize = True
        else:
            need_resize = False
        
        # Split magnitude and phase channels
        magnitude = x[:, :x.shape[1]//2]  # First half channels
        phase = x[:, x.shape[1]//2:]      # Second half channels
        
        # Process magnitude and phase separately
        mag_features = self.magnitude_conv(magnitude)
        phase_features = self.phase_conv(phase)
        
        # Fuse magnitude and phase features
        fused = torch.cat([mag_features, phase_features], dim=1)
        fused_features = self.fusion_conv(fused)
        
        # ResNet encoder with skip connections
        x = self.conv1(fused_features)
        x = self.bn1(x)
        x = self.relu(x)
        skip0 = x  # For potential use
        x = self.maxpool(x)
        
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        x = self.layer4(skip3)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Enhanced decoder with skip connections
        x = self.decoder4(x)
        x = torch.cat([x, skip3], dim=1)
        
        x = self.decoder3(x)
        x = torch.cat([x, skip2], dim=1)
        
        x = self.decoder2(x)
        x = torch.cat([x, skip1], dim=1)
        
        x = self.decoder1(x)
        
        # Upsample fused_features to match decoder output size
        fused_upsampled = F.interpolate(fused_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, fused_upsampled], dim=1)
        
        # Final output
        x = self.final_conv(x)
        
        # Resize back to original size if we upsampled
        if need_resize:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        
        # Scale from [0, 1] to [0, 5] seconds for T1 values
        # x = x * 5
        
        return x


# Alternative: Even larger network using EfficientNet backbone
class EfficientT1MappingNet(nn.Module):
    def __init__(self, input_channels=8, output_channels=1):
        super(EfficientT1MappingNet, self).__init__()
        
        try:
            # Try to use EfficientNet if available
            import timm
            self.backbone = timm.create_model('efficientnet_b4', pretrained=True, features_only=True)
            feature_channels = [24, 32, 56, 160, 448]  # EfficientNet-B4 feature channels
        except ImportError:
            # Fallback to ResNet101
            print("timm not available, using ResNet101")
            resnet = models.resnet101(pretrained=True)
            feature_channels = [64, 256, 512, 1024, 2048]
            
        # Input processing (same as above)
        self.magnitude_conv = nn.Sequential(
            nn.Conv2d(input_channels // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.phase_conv = nn.Sequential(
            nn.Conv2d(input_channels // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Rest of the architecture similar to EnhancedT1MappingNet
        # ... (implementation details would follow similar pattern)


# Factory function to create networks
def create_t1_network(input_channels=8, output_channels=1, model_type='enhanced'):
    """
    Factory function to create T1 mapping networks
    
    Args:
        input_channels: Number of input channels (should be 2*n_TI for mag+phase)
        output_channels: Number of output channels (typically 1 for T1 map)
        model_type: 'original', 'enhanced', 'efficient'
    """
    if model_type == 'original':
        from your_original_module import T1MappingNet  # Your original network
        return T1MappingNet(input_channels, output_channels)
    elif model_type == 'enhanced':
        return EnhancedT1MappingNet(input_channels, output_channels, backbone='resnet50')
    elif model_type == 'large':
        return EnhancedT1MappingNet(input_channels, output_channels, backbone='resnet101')
    elif model_type == 'efficient':
        return EfficientT1MappingNet(input_channels, output_channels)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Example usage:
if __name__ == "__main__":
    # Test the network
    model = EnhancedT1MappingNet(input_channels=8, output_channels=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 8, 32, 32)  # batch_size=1, channels=8, height=32, width=32
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")