import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.double_conv(x)
    
class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DownConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
    def forward(self, x):
        x = self.conv(x)
        x1 = self.pool(x)
        return x,x1
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels) -> None:
        super(UpConv, self).__init__()
        adjusted_in_channels = in_channels + skip_channels
        self.transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv =  DoubleConv(adjusted_in_channels, out_channels)
        
        
    def forward(self, x,x1):
        x = self.transpose(x)
        x = torch.cat([x,x1],dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self,in_channels=1, num_classes=1):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = DownConv(in_channels, 64)
        self.down_convolution_2 = DownConv(64, 128)
        self.down_convolution_3 = DownConv(128, 256)
        self.down_convolution_4 = DownConv(256, 512)
        self.down_convolution_5 = DoubleConv(512, 1024)
        
        

        # Expanding path.
        self.up_transpose_1 = UpConv(1024, 512, 512)
        self.up_transpose_2 = UpConv(512, 256, 256)
        self.up_transpose_3 = UpConv(256, 128, 128)
        self.up_transpose_4 = UpConv(128, 64, 64)
        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # TODO: Write here!
        x1,x = self.down_convolution_1(x)
        x2,x = self.down_convolution_2(x)
        x3,x = self.down_convolution_3(x)
        x4,x = self.down_convolution_4(x)
        x5 = self.down_convolution_5(x)
        
        x = self.up_transpose_1(x5,x4)
        x = self.up_transpose_2(x,x3)
        x = self.up_transpose_3(x,x2)
        x = self.up_transpose_4(x,x1)
        
        out = self.out(x)
        return out