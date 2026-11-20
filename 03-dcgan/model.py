import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim + self.num_classes, out_channels=512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        emb = self.label_embedding(labels)
        cond_inputs = torch.cat([z, emb], dim=1)
        cond_inputs = cond_inputs.view(cond_inputs.size(0), -1, 1, 1)
        img = self.model(cond_inputs)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32), num_classes=10):
        super().__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.img_shape[0] + self.num_classes, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Flatten()
        )

    def forward(self, img, labels):
        emb = self.label_embedding(labels)
        emb = emb.view(emb.size(0), -1, 1, 1)
        emb = emb.expand(-1, -1, img.size(2), img.size(3))
        x = torch.cat([img, emb], dim=1)
        preds = self.model(x)
        return preds

