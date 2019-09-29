import torch
import torch.nn as nn
import NET
import os
from torch.utils.data import DataLoader
import DataHandle
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 100
MODEL_D_NET = './MODEL/D_MODEL.pt'
MODEL_G_NET = './MODEL/G_MODEL.pt'

DDN_SIZE = 256  #隐藏特征数量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # 数据集
    data_set = DataHandle.DATA_SET(r"G:\项目\20190830\faces")
    train_data = DataLoader(data_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=2)

    # 网络
    d_net = NET.D_NET().to(device)
    g_net = NET.G_NET().to(device)

    if os.path.exists(MODEL_D_NET):
        d_net.load_state_dict(torch.load(MODEL_D_NET),strict=False)
    if os.path.exists(MODEL_G_NET):
        g_net.load_state_dict(torch.load(MODEL_G_NET),strict=False)

    # 损失函数
    loss_fn = nn.BCELoss()

    d_opt = torch.optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))  # ?????
    g_opt = torch.optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # tensorboard
    writer = SummaryWriter()
    for epoch in range(50):
        for i, imgData in enumerate(train_data):
            # 标签
            real_label = torch.ones(imgData.size(0), 1, 1, 1).to(device)
            fake_label = torch.zeros(imgData.size(0), 1, 1, 1).to(device)

            # 判别器训练
            real_out = d_net(imgData.to(device))
            real_score = real_out
            d_loss_real = loss_fn(real_out, real_label)

            z = torch.randn(imgData.size(0), DDN_SIZE, 1, 1).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            fake_score = fake_out
            d_loss_fake = loss_fn(fake_out, fake_label)

            d_loss = d_loss_real + d_loss_fake

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # 生成器训练
            z = torch.randn(imgData.size(0), DDN_SIZE, 1, 1).to(device)
            fake_img_1 = g_net(z)
            output = d_net(fake_img_1)
            g_loss = loss_fn(output, real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            for name, value in d_net.named_parameters():
                # print('name: {0},\t grad: {1}'.format(name, value.grad))
                writer.add_histogram(name,value,epoch)

            if i % 30 == 0:
                real_score = real_score.cpu().data.mean()
                fake_score = fake_score.cpu().data.mean()
                print("Epoch:[{}/{}],d_loss:{:.3f},"
                      "g_loss:{:.3f},real_score:{:.3f},fake_score:{:.3f}"
                      .format(i, epoch, d_loss, g_loss, real_score, fake_score))

                fake_img = fake_img.cpu().data

                save_image(fake_img, "./IMG/{}-fake.png".format(epoch),
                           nrow=10, normalize=True, scale_each=True)
                save_image(imgData, "./IMG/{}-real.png".format(epoch),
                           nrow=10, normalize=True, scale_each=True)

                writer.add_scalars("loss", {
                    "d_loss": d_loss,
                    "g_loss": g_loss,
                }, epoch)

                writer.add_scalars("score", {
                    "real": real_score,
                    "fake": fake_score,
                }, epoch)

        torch.save(d_net.state_dict(), MODEL_D_NET)
        torch.save(g_net.state_dict(), MODEL_G_NET)
