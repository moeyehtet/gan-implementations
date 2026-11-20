import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Generator, Discriminator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.weights_init import weights_init_normal
from utils.visualization import save_dcgan_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("outputs", exist_ok=True)

    # load datasets - CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_dataset = datasets.CIFAR10("../../data/cifar10", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    num_classes = len(train_dataset.classes)

    # loss
    adversarial_loss = nn.BCEWithLogitsLoss()

    # initialize models
    generator = Generator(latent_dim=config["latent_dim"], num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)

    # initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))

    for epoch in range(config["num_epochs"]):
        for i, (imgs, labels) in enumerate(train_dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

            # ground truths
            valid_smooth = torch.full((batch_size, 1), 0.9, device=device)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # train discriminator
            z = torch.randn(batch_size, config["latent_dim"], device=device)
            gen_imgs = generator(z, gen_labels)

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid_smooth)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()

            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

            g_loss.backward()
            optimizer_G.step()

            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{config["num_epochs"]}, Batch {i+1}/{len(train_dataloader)}: D loss = {d_loss.item():.4f}, G loss = {g_loss.item():.4f}")
    
        n_per_class = 10
        fixed_z = torch.randn(num_classes * n_per_class, config["latent_dim"], device=device)
        save_dcgan_images(generator, fixed_z, num_classes, device, epoch, output_dir="outputs")

if __name__ == "__main__":
    main()

