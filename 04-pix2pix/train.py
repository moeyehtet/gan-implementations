import os
import sys
import time
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Generator, Discriminator, Pix2Pix
from facades_dataset import FacadesDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.weights_init import weights_init_normal
from utils.visualization import save_pix2pix_images

def train_pix2pix(pix2pix_model, optimizer_G, optimizer_D, train_loader, val_loader, device, num_epochs, lambda_l1=100):
    start_time = time.time()
    g_losses, d_losses = [], []
    g_losses_adversarial, g_losses_l1 = [], []
    val_g_losses, val_d_losses = [], []

    for epoch in range(num_epochs):
        pix2pix_model.train()
        train_g_loss, train_d_loss = 0.0, 0.0
        train_g_loss_adversarial, train_g_loss_l1 = 0.0, 0.0

        for idx, (segmented_imgs, real_imgs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            segmented_imgs = segmented_imgs.to(device)
            real_imgs = real_imgs.to(device)

            # train discriminator
            optimizer_D.zero_grad()
            fake_imgs = pix2pix_model.generator(segmented_imgs)

            real_output = pix2pix_model.discriminator(segmented_imgs, real_imgs)
            fake_output = pix2pix_model.discriminator(segmented_imgs, fake_imgs.detach())
            d_loss = pix2pix_model.discriminator_loss(real_output, fake_output)
            
            d_loss.backward()
            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()
            
            fake_output = pix2pix_model.discriminator(segmented_imgs, fake_imgs)
            g_loss, g_loss_adversarial, g_loss_l1 = pix2pix_model.generator_loss(fake_output, fake_imgs, real_imgs, lambda_l1)

            g_loss.backward()
            optimizer_G.step()

            train_d_loss += d_loss.item()
            train_g_loss += g_loss.item()
            train_g_loss_adversarial += g_loss_adversarial.item()
            train_g_loss_l1 += g_loss_l1.item()

        d_losses.append(train_d_loss / len(train_loader))
        g_losses.append(train_g_loss / len(train_loader))
        g_losses_adversarial.append(train_g_loss_adversarial / len(train_loader))
        g_losses_l1.append(train_g_loss_l1 / len(train_loader))

        print(f"D loss = {d_losses[epoch]:.4f}, G loss = {g_losses[epoch]:.4f} (adversarial: {g_losses_adversarial[epoch]:.4f}, l1: {g_losses_l1[epoch]:.4f})")
        
        save_pix2pix_images(pix2pix_model, val_loader, device, epoch, output_dir="outputs")

    time_taken = time.time() - start_time
    print(f"Training completed in {int(time_taken // 60)} m, {int(time_taken % 60)} s.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = FacadesDataset("../data/facades", split="train", direction="BtoA")
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=2)

    val_dataset = FacadesDataset("../data/facades", split="val", direction="BtoA")
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'], shuffle=False, num_workers=2)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    pix2pix = Pix2Pix(generator, discriminator).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['b1'], config['b2']))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['b1'], config['b2']))

    train_pix2pix(pix2pix, optimizer_G, optimizer_D, train_loader, val_loader, device, config['num_epochs'], config['lambda_l1'])

if __name__ == "__main__":
    main()
