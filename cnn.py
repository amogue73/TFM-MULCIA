import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CircularConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        pad_h = self.padding
        pad_w = self.padding

        # Circular padding: (left, right, top, bottom)
        x = F.pad(x, (0, 0, pad_h, pad_h), mode='circular')
        x = self.conv(x)
        return x

model = CircularConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(1, 3, 32, 32)  # Batch of 1 image with 3 channels, 32x32 pixels
out = model(x)
print(out.shape)