"""
T1 Mapping Network using Pretrained ResNet18
"""

import torch
import torch.nn as nn
import torchvision.models as models


class T1MappingNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=1):
        super(T1MappingNet, self).__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=True)

        # Modify first conv for custom input channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Copy pretrained weights for RGB channels, duplicate for extra channels
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = resnet.conv1.weight
            if input_channels > 3:
                self.conv1.weight[:, 3:, :, :] = resnet.conv1.weight[:, :1, :, :]

                # Use pretrained ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Custom decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 3, padding=1),
            nn.Sigmoid()  # Force values to [0, 1] range
        )

    def forward(self, x):
        # Handle small inputs - upsample to avoid 1x1 feature maps
        if x.shape[-1] <= 32:
            x = torch.nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            need_resize = True
        else:
            need_resize = False

        # ResNet encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Decoder
        x = self.decoder(x)

        # Resize back to 32x32 if we upsampled
        if need_resize:
            x = torch.nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

        # Step 1: Sigmoid ensures [0, 1] range (already applied in decoder)
        # Step 2: Scale from [0, 1] to [0, 5] seconds
        x = x * 5

        return x