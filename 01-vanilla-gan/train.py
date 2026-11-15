import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Generator, Discriminator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.weights_init import weights_init_normal
from utils.visualization import save_generated_images

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

    # loss
    adversarial_loss = nn.BCEWithLogitsLoss()

    # initialize models
    generator = Generator(latent_dim=config["latent_dim"]).to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(config["b1"], config["b2"]))

    # load dataset - MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = datasets.MNIST("../../data/mnist", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # training loop
    for epoch in range(config["num_epochs"]):
        for i, (imgs, _) in enumerate(train_dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)

            # ground truths
            valid_smooth = torch.full((batch_size, 1), 0.9, device=device)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # discriminator training
            z = torch.randn(batch_size, config["latent_dim"], device=device)
            gen_imgs = generator(z)

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()


            # generator training
            optimizer_G.zero_grad()

            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            if (i+1) % 400 == 0:
                print(f"Epoch {epoch+1}/{config["num_epochs"]}, Batch {i+1}/{len(train_dataloader)}: D loss = {d_loss.item():.4f}, G loss = {g_loss.item():.4f}")

        save_generated_images(generator, config["latent_dim"], device, epoch, output_dir="outputs")

if __name__ == "__main__":
    main()
