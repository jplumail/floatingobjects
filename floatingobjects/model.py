import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

def get_model(modelname, inchannels=12, pretrained=True):
    if modelname == "unet":
        # initialize model (random weights)
        model = UNet(n_channels=inchannels,
                     n_classes=1,
                     bilinear=False)
    elif modelname in ["resnetunet", "resnetunetscse"]:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34" if "resnet" in modelname else "efficientnet-b7",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            decoder_attention_type="scse" if modelname == "resnetunetscse" else None,
            classes=1,
        )
        model.encoder.conv1 = torch.nn.Conv2d(inchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    elif modelname in ["manet"]:
        import segmentation_models_pytorch as smp
        model = smp.MAnet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        model.encoder.conv1 = torch.nn.Conv2d(inchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif modelname == "classifier":
        model = Classifier(in_channels=inchannels)
    elif modelname == "dummy":
        model = DummyModel()
    else:
        raise ValueError(f"model {modelname} not recognized")
    return model

#============== some parts of the U-Net model ===============#
""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#=================== Assembling parts to form the network =================#
""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x.float())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, x):
        size = list(x.shape)
        size[1] = 1
        return torch.empty(size)
    
from torchvision.models import resnet18

class Classifier(nn.Module):

    def __init__(self, in_channels=12):
        super().__init__()
        self.resnet18 = resnet18(num_classes=1)
        self.resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.resnet18.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.register_buffer('mean_filter', torch.ones(1, 1, 16, 16), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(self.resnet18(x))
        return y_pred
    
    def sliding_windows(self, x, stride=1):
        # x of shape (N, C, H, W)
        # returns scores of shape (N, H, W)
        self.eval()
        N, C, H, W = x.shape

        x_unfold = F.unfold(x, (16, 16), padding=7, stride=stride)
        x_unfold = x_unfold.view(C, 16, 16, -1).permute((3, 0, 1, 2)) # shape (N*L, C, h, w)
        out_unfold = self.forward(x_unfold) # shape (N*L)
        out_fold = out_unfold.view(N, 1, H//stride, W//stride)
        scores = resize(out_fold, size=(H,W))

        # Mean filter 16x16
        #norm_tensor = F.conv2d(torch.ones_like(out_fold), self.mean_filter, padding='same')
        #out_mean = F.conv2d(out_fold, self.mean_filter, padding='same') / norm_tensor

        return scores[:, 0]