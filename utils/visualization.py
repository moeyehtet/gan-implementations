import torch
import torchvision.utils as vutils
import os

def save_gan_images(generator, latent_dim, device, epoch, output_dir="outputs", num_samples=64):
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

def save_cgan_images(generator, fixed_z, num_classes, device, epoch, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    n_per_class = fixed_z.size(0) // num_classes
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)

    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_z.to(device), labels).detach().cpu()
        fake_images = (fake_images + 1) / 2
        grid = vutils.make_grid(fake_images, nrow=n_per_class, normalize=False)
        path = os.path.join(output_dir, f"epoch_{epoch+1:03d}.png")
        vutils.save_image(grid, path)
    generator.train()

def save_dcgan_images(generator, fixed_z, num_classes, device, epoch, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    n_per_class = fixed_z.size(0) // num_classes
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_per_class)

    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_z.to(device), labels).detach().cpu()
        fake_images = (fake_images + 1) / 2
        grid = vutils.make_grid(fake_images, nrow=n_per_class, normalize=False)
        path = os.path.join(output_dir, f"epoch_{epoch+1:03d}.png")
        vutils.save_image(grid, path)
    generator.train()

def save_pix2pix_images(model, data_loader, device, epoch, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    num_samples = len(data_loader)

    model.eval()
    with torch.no_grad():
        segmented_imgs, real_imgs = next(iter(data_loader))
        segmented_imgs = segmented_imgs[:num_samples].to(device)
        real_imgs = real_imgs[:num_samples].to(device)

        fake_imgs = model.generator(segmented_imgs)

        rows = torch.stack([segmented_imgs, real_imgs, fake_imgs], dim=1)
        rows = rows.flatten(0, 1)
        grid = vutils.make_grid(rows, nrow=3, normalize=True, value_range=(-1, 1))
        path = os.path.join(output_dir, f"epoch_{epoch+1:03d}.png")
        vutils.save_image(grid, path)
    model.train()
