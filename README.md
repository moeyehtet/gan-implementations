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

1. Vanilla GAN[^1]
2. Conditional GAN[^2] 
3. DCGAN[^3] 
4. Pix2Pix[^4]
5. CycleGAN
6. Pix2PixHD
7. StyleGAN

## To-dos

1. Save checkpoints
2. Model saving

## Milestones

| # | Milestone & Year | Core Breakthrough / Concept | Used Dataset | Completed Date |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Vanilla GAN** | Basic minimax game using Multi-Layer Perceptrons (MLPs) and Binary Cross Entropy loss. | MNIST | 15.11.2026 |
| **02** | **Conditional GAN** | Extension of vanilla GAN by adding conditioning with class labels to both generator and discriminator to control the modes of data being generated. | Fashion-MNIST | 18.11.2026 |
| **03** | **Deep Convolutional GAN** | Replaces MLPs with fully convolutional generator and discriminator for stable training on natural images. Implemented as class-conditional DCGAN to control the class of images being generated. | CIFAR-10 | 20.11.2026 |
| **04** | **Pix2Pix** | Paired image-to-image translation with a conditional GAN. A U-Net generator with skip connections, a PatchGAN discriminator that outputs scores for local patches instead of the whole image and combination of adversarial loss and L1 loss. | CMP Facade Database | 25.11.2026 |

[^1]: I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Advances in Neural Information Processing Systems (NeurIPS), vol. 27, 2014, pp. 2672–2680.
[^2]: M. Mirza and S. Osindero, "Conditional generative adversarial nets," arXiv preprint arXiv:1411.1784, 2014.
[^3]: A. Radford, L. Metz, and S. Chintala, "Unsupervised representation learning with deep convolutional generative adversarial networks," in Proc. Int. Conf. Learn. Represent. (ICLR), San Juan, Puerto Rico, 2016, arXiv:1511.06434.
[^4]: Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2017).
