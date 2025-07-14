import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        feats = [base_features * 2 ** i for i in range(depth)]

        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_c = in_channels
        for f in feats:
            self.down_blocks.append(ConvBlock(prev_c, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_c = f

        self.bottleneck = ConvBlock(prev_c, prev_c * 2)
        prev_c = prev_c * 2

        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        feats_rev = list(reversed(feats))
        for f in feats_rev:
            self.ups.append(nn.ConvTranspose2d(prev_c, f, 2, stride=2))
            self.up_blocks.append(ConvBlock(prev_c, f))
            prev_c = f

        self.final_conv = nn.Conv2d(prev_c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.final_conv(x)
