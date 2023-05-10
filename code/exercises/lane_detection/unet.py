import torch.nn as nn
import torch

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class encode(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(encode, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

 
class decode(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(decode, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2) 
        self.conv = double_conv(2*out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting path
        self.double_conv = double_conv(3, 64)
        self.encode1 = encode(64, 128)
        self.encode2 = encode(128, 256)
        self.encode3 = encode(256, 512)
        self.encode4 = encode(512, 1024)
        # Expansive path
        self.decode1 = decode(1024, 512)
        self.decode2 = decode(512, 256)
        self.decode3 = decode(256, 128)
        self.decode4 = decode(128, 64)
        # Single convolution
        self.conv = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.encode1(x1)
        x3 = self.encode2(x2)
        x4 = self.encode3(x3)
        x5 = self.encode4(x4)
        x = self.decode1(x5, x4)
        x = self.decode2(x, x3)
        x = self.decode3(x, x2)
        x = self.decode4(x, x1)
        x = self.conv(x)
        return x