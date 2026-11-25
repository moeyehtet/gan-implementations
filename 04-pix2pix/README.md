# Pix2Pix

Implementation of **Image-to-Image Translation with Conditional Adversarial Networks**[^1], better known as pix2pix, in PyTorch, trained on CMP Facade Database[^2] to translate segmentation label maps into images of building facades. The generator uses a U-Net architecture[^3] and the discriminator is a 70x70 PatchGAN[^1]. Unlike conditional GANs in earlier milestones, the target image itself is passed as the condition instead of a class label.

## How to run

```bash
python train.py --config config.yaml
```

The training script expects the dataset at `../data/facades` with `train`, `val` and `test` subfolders.

## Dataset

The pix2pix network is trained on the CMP Facade Database[^2], using the pre-split version distributed with the pix2pix paper[^1]. Each file is a 256x512 JPEG image with a building photo on the right and the segmented label map on the left. The dataset is used in the direction from label map (input) to photo (target). The dataset is not included in the repository. It can be obtained from the [pix2pix dataset page](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/). Also see the original [CMP Facade Database](https://cmp.felk.cvut.cz/~tylecr1/facade/) for more details.

### Data Preprocessing

1. Each original image from the dataset is split down the middle into an input segmentation map and a target image.
2. The splitted images are stacked into a 6-channel tensor. The same random jitter is applied to the input and the target for the training images. The random jitter applied to the stacked images are resize to 286x286, random crop to 256x256 and random horizontal flip.
3. The pixel values of the stacked image pair are normalized to [-1, 1] range with a mean and standard deviation of 0.5.

## Network Architecture

### Generator

The generator uses a U-Net architecture, mapping an input image `x` of size 3x256x256 to a target-domain image of the same size with pixel values in [-1, 1] range. The encoder halves the resolution every layer down to a 1x1 bottleneck. The decoder doubles the resolution every layer up to the same size as the input image. Each decoder output is concatenated with the encoder feature map of the same resolution (skip connections) before the next decoder layer.

#### Encoder

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input `x` | 3x256x256 | |
| Conv(k=4, s=2, p=1) + LeakyReLU(0.2) | 64x128x128 | No BatchNorm on the first layer (`e1`) |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 128x64x64 | `e2` |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 256x32x32 | `e3` |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 512x16x16 | `e4` |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 512x8x8 | `e5` |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 512x4x4 | `e6` |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 512x2x2 | `e7` |
| Conv(k=4, s=2, p=1) + LeakyReLU(0.2) | 512x1x1 | Bottleneck (`e8`), no BatchNorm |

#### Decoder

| Layer | Output shape | After skip concatenation | Notes |
|-------|--------------|--------------------------|-------|
| ConvTranspose(k=4, s=2, p=1) + BatchNorm + ReLU + Dropout(0.5) | 512x2x2 | 1024x2x2 | Skip from `e7` |
| ConvTranspose + BatchNorm + ReLU + Dropout(0.5) | 512x4x4 | 1024x4x4 | Skip from `e6` |
| ConvTranspose + BatchNorm + ReLU + Dropout(0.5) | 512x8x8 | 1024x8x8 | Skip from `e5` |
| ConvTranspose + BatchNorm + ReLU | 512x16x16 | 1024x16x16 | Skip from `e4` |
| ConvTranspose + BatchNorm + ReLU | 256x32x32 | 512x32x32 | Skip from `e3` |
| ConvTranspose + BatchNorm + ReLU | 128x64x64 | 256x64x64 | Skip from `e2` |
| ConvTranspose + BatchNorm + ReLU | 64x128x128 | 128x128x128 | Skip from `e1` |
| ConvTranspose(k=4, s=2, p=1) + Tanh | 3x256x256 | | Output layer |

#### Things to note

- Convolution layers followed by BatchNorm use `bias=False`.
- The generator takes no noise vector like in previous GAN models. The only source of randomness is from the dropout in first three decoder layers.

### Discriminator

For discriminator, PatchGAN architecture from pix2pix paper is used.[^1] It takes the input image and the target-domain image (either real or generated) and outputs a 30x30 map of logit scores. Instead of giving a single score on the whole image, PatchGAN looks at 70x70 local patches of the image pair and determines whether the patch is real or fake. The final loss is averaged over all patches.

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input `x` and target `y` | 3x256x256 each | |
| Concatenate `[x, y]` | 6x256x256 | Joined along the channel dimension |
| Conv(k=4, s=2, p=1) + LeakyReLU(0.2) | 64x128x128 | No BatchNorm on the first layer |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 128x64x64 | |
| Conv(k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 256x32x32 | |
| Conv(k=4, s=1, p=1) + BatchNorm + LeakyReLU(0.2) | 512x31x31 | |
| Conv(k=4, s=1, p=1) | 1x30x30 | Patch logits (no Sigmoid) |

### Weight Initialization

The weights of all convolution and transposed convolution layers are drawn from $\mathcal{N}(0, 0.02)$. The weights for BatchNorm are drawn from $\mathcal{N}(0, 0.02)$ with zero bias.

### Loss and Optimizer

The discriminator uses Binary Cross Entropy loss with logit scores. However, the generator uses both the loss for a conditional GAN (Binary Cross Entropy loss with logit scores) and L1 loss comparing values pixel-by-pixel.[^1]

$$
\begin{align}
\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log (1 - D(x, G(x, z)))] \\
\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[ || y - G(x, z) ||_{1} ] \\
\mathcal{L}_{pix2pix} = \mathcal{L}_{cGAN} + \lambda * \mathcal{L}_{L1}
\end{align}
$$

The value of $\lambda$ can be modified to change the influence of the L1 term in the loss. The training is done with $\lambda = 100$.

The generator maximizes $\log D(x, G(x, z))$ instead of minimizing $\log (1 - D(x, G(x, z)))$ as suggested in the original GAN paper.[^4]

For optimizer, both networks use their own Adam optimizer with learning rate of 0.0002, beta1 of 0.5 and beta2 of 0.999.

## Training Loop

Each iteration processes one minibatch of size 1. The discriminator is updated first, followed by the generator.

### Discriminator Training Step

1. Generate fake image $G(x)$ from the input image $x$.
2. Compute BCE loss on the real pair $(x, y)$ against label `1`.
3. Compute BCD loss on the fake pair $(x, G(x))$ against label `0`. The fake image is detached so that no gradient reaches the generator.
4. Compute discriminator loss by adding the losses on the real and fake pairs and divide by 2.
5. Update the discriminator parameters.

### Generator Training Step

1. Pass the fake pair $(x, G(x))$ through the updated discriminator again. (The old computation cannot be reused because it was computed with weights for the discriminator before updating.)
2. Compute BCE loss against the label `1`.
3. Compute L1 loss between the generated image and the target image.
4. Compute generator loss by adding the BCE loss and the L1 loss which is weighted by $\lambda=100$.
5. Update generator parameters.

### Sample Visualization

After each training epoch, the generator is run on a fixed batch of 10 validation image pairs. The results are saved as a grid with one example per row and three columns: the input segmentation label map, the ground truth image and the generated image. The same validation images are used every time to make it possible to follow the improvement over training.

## Deviations from the Original Paper

1. The discriminator outputs a map of logit scores instead of a sigmoid output layer and trained with `BCEWithLogitsLoss`.
2. Sample images are generated in evaluation mode with dropout turned off and BatchNorm with running statistics while the paper keeps dropout active at test time.[^1]

[^1]: Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2017).
[^2]: Tyleček, Radim, and Radim Šára. "Spatial pattern templates for recognition of objects with regular structure." Proc. GCPR, Saarbrücken, Germany (2013).
[^3]: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional networks for biomedical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention (2015).
[^4]: Goodfellow, Ian J., et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).
