import torch
import torchvision.utils as vutils
import os

def save_generated_images(generator, latent_dim, device, epoch, output_dir="outputs", num_samples=64):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_images = generator(z).detach().cpu()
        fake_images = (fake_images + 1) / 2 # rescale [-1, 1] to [0, 1]
        grid = vutils.make_grid(fake_images, nrow=8, normalize=False)
        path = os.path.join(output_dir, f"epoch_{epoch+1:03d}.png")
        vutils.save_image(grid, path)
    generator.train()
