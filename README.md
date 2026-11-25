# GAN Learning Journey

This repository documents my journey of implementing major GAN architectures from scratch in PyTorch, starting from Ian Goodfellow's 2014 foundational paper. This is meant to bridge between theoretical knowledge gained from the papers and the application in practice.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Learning Roadmap

The planned learning roadmap is as follows:

1. Vanilla GAN (I. J. Goodfellow et al., “Generative Adversarial Nets,” Adv. Neural Inf. Process. Syst., vol. 27, 2014.)
2. Conditional GAN (M. Mirza and S. Osindero, “Conditional Generative Adversarial Nets,” 2014, arXiv. doi: 10.48550/arXiv.1411.1784.)
3. DCGAN (Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).)
4. Pix2Pix
5. CycleGAN
6. Pix2PixHD
7. StyleGAN

## Milestones

| # | Milestone & Year | Core Breakthrough / Concept | Used Dataset | Completed Date |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Vanilla GAN** | Basic minimax game using Multi-Layer Perceptrons (MLPs) and Binary Cross Entropy loss. | MNIST | 15.11.2026 |
| **02** | **Conditional GAN** | Extension of vanilla GAN by adding conditioning with class labels to both generator and discriminator to control the modes of data being generated. | Fashion-MNIST | 18.11.2026 |
| **03** | **Deep Convolutional GAN** | Replaces MLPs with fully convolutional generator and discriminator for stable training on natural images. Implemented as class-conditional DCGAN to control the class of images being generated. | CIFAR-10 | 20.11.2026 |

## To-dos

1. Save checkpoints
2. Model saving 
