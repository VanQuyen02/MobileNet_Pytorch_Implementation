import torch.nn as nn
from torchsummary import summary


class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        #depthwise seperable conv = depthwise + point wise
        def conv_dpw(in_channels, out_channels, stride):
            return nn.Sequential(
                # depth wise
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                # point wise
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dpw(32, 64, 1),
            conv_dpw(64, 128, 2),
            conv_dpw(128, 128, 1),
            conv_dpw(128, 256, 2),
            conv_dpw(256, 256, 1),
            conv_dpw(256, 512, 2),
            conv_dpw(512, 512, 1),
            conv_dpw(512, 512, 1),
            conv_dpw(512, 512, 1),
            conv_dpw(512, 512, 1),
            conv_dpw(512, 512, 1),
            conv_dpw(512, 1024, 2),
            conv_dpw(1024, 1024, 1),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # model check
    model = MobileNetV1(ch_in=3, n_classes=1000)
    summary(model, input_size=(3, 224, 224), device='cpu')
