import torch
import torch.nn as nn

class D_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,5,3,1,bias=False),  #[N, 32, 32, 32]
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # [N, 64, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # [N, 128, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # [N, 256, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # [N, 1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class G_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256,512,4,1,0,bias=False), #[N, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.PReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1,bias=False),  # [N, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1,bias=False),  # [N, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.PReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1,bias=False),  # [N, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.PReLU(),

            nn.ConvTranspose2d(64, 3, 5, 3, 1,bias=False),  # [N, 3, 96, 96]
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.conv(x)

        return out



if __name__ == '__main__':
    # net = D_NET()
    # x = torch.rand(1,3,96,96)
    # output = net(x)
    #
    # print(output.shape)

    net = D_NET()

    for name, value in net.named_parameters():
        print('name: {0},\t grad: {1}'.format(name, value.grad))