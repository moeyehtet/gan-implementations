# GAN Learning Journey

This repository documents my journey of implementing major GAN architectures from scratch in PyTorch, starting from Ian Goodfellow's 2014 foundational paper. This is meant to bridge between theoretical knowledge gained from the papers and the application in practice.

## Learning Roadmap

The planned learning roadmap is as follows:

1. Vanilla GAN (Goodfellow, Ian J., et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).)
2. Conditional GAN
3. DCGAN
4. WGAN
5. Pix2Pix
6. CycleGAN
7. Pix2PixHD
8. StyleGAN

## Milestones

| # | Milestone & Year | Core Breakthrough / Concept | Used Dataset | Completed Date |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Vanilla GAN** | Basic minimax game using Multi-Layer Perceptrons (MLPs) and Binary Cross Entropy loss. | MNIST | 15.11.2026 |

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```