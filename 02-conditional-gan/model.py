import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28), num_classes=10):
        super().__init__()
        self.img_shape = img_shape
        out_dim = int(torch.prod(torch.tensor(img_shape)))

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, out_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        conditional_inputs = torch.cat([z, self.label_embedding(labels)], dim=1)
        img = self.model(conditional_inputs)
        return img.view(img.size(0), *self.img_shape)
            
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), dropout=0.3, num_classes=10):
        super().__init__()
        self.img_dim = int(torch.prod(torch.tensor(img_shape)))

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(self.img_dim + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        conditional_inputs = torch.cat([img_flat, self.label_embedding(labels)], dim=1)
        return self.model(conditional_inputs)
