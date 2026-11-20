# Deep Convolutional Generative Adversarial Network

Implementation of **Deep Convolutional Generative Adversarial Network**[^1] extended by also conditioning the generator and discriminator with class labels to control modes of data being generated. The network is implemented in PyTorch and trained on CIFAR-10 dataset.

## How to run

```bash
python train.py --config config.yaml
```

## Network Architecture

### Generator

Maps a latent vector `z` of size 100 and a class label `y` to a 32x32 RGB image with pixel values in the range \[-1, 1\]. The class label is converted to a 10-dimensional embedding and concatenated with `z`.

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input `z` | 100 | Gaussian noise |
| Label embedding of `y` | 10 | `nn.Embedding(10, 10)` |
| Concatenate `[z, label_embedding(y)]` + Reshape | 110x1x1 | Joined along the feature dimension |
| ConvTranspose (k=4, s=1, p=0) + BatchNorm + ReLU | 512x4x4 | |
| ConvTranspose (k=4, s=2, p=1) + BatchNorm + ReLU | 256x8x8 | |
| ConvTranspose (k=4, s=2, p=1) + BatchNorm + ReLU | 128x16x16 | |
| ConvTranspose (k=4, s=2, p=1) + Tanh | 3x32x32 | |

### Discriminator

Maps a 32x32 RGB image and a class label `y` to a logit score which indicates whether the image is real or fake for the given class. The discriminator has its own label embedding not shared with the generator. The label embedding is expanded into a 10-channel spatial map (repeated for each pixel value) and concatenated with the input image, along the channel dimension.

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input image | 3x32x32 | |
| Label embedding of `y` | 10x32x32 | `nn.Embedding(10, 10)`, each value repeated over all positions |
| Concatenate `[image, label_map]` | 13x32x32 | Joined along the channel dimension |
| Conv (k=4, s=2, p=1) + LeakyReLU(0.2) | 64x16x16 | No BatchNorm on the first Conv layer |
| Conv (k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 128x8x8 | |
| Conv (k=4, s=2, p=1) + BatchNorm + LeakyReLU(0.2) | 256x4x4 | |
| Conv (k=4, s=1, p=0) + Flatten | 1 | Logit score |

Convolutions followed by BatchNorm use `bias=False`. There are no pooling layers and no dropout.

### Weight Initialization

Following the DCGAN paper[^1], the weights of all convolution and transposed convolution layers are drawn from $\mathcal{N}(0, 0.02)$. BatchNorm weights are drawn from $\mathcal{N}(0, 0.02)$ with zero bias.

### Loss and Optimizer

For loss function, both generator and discriminator networks use Binary Cross Entropy loss with logit scores. For the generator loss, instead of minimizing $\log (1 - D(G(z|y)|y))$ as in the original GAN paper, $\log D(G(z|y)|y)$ is maximized, as suggested as an alternative in the original GAN paper[^3].
 
For optimizer, both generator and discriminator networks use their own Adam optimizer with learning rate of 0.0002, beta1 of 0.5 and beta2 of 0.999.

## Training Loop
 
Each iteration processes one minibatch. The parameters of the discriminator are updated for $k=1$ steps first, followed by the parameter update of the generator. Images are normalized to \[-1, 1\] with a mean and standard deviation of 0.5 per channel to match the generator's Tanh output. The last incomplete batch is dropped (`drop_last=True`) so that BatchNorm always sees full batches.
 
### Discriminator Training Step
 
1. Sample random class labels `gen_labels` uniformly from the 10 classes and generate fake images $G(z|y)$ from latent noise vector `z` and `gen_labels`.
2. Compute binary cross entropy loss using logit scores on real images paired with their true labels, against the smoothed label `0.9`.
3. Compute binary cross entropy loss using logit scores on fake images paired with `gen_labels`, against the label `0`. Real and fake images are passed through the discriminator in separate batches.
4. Compute average discriminator loss by adding the losses on real and fake images and divide by 2.
5. Update discriminator parameters.
### Generator Training Step
 
1. Take the generated fake images from discriminator training step and compute binary cross entropy loss using logit scores against the label `1`. The fake images are paired with the same `gen_labels` used to generate them.
2. Update generator parameters.

### Sample Visualization
 
After each epoch, the generator is run in eval mode on a fixed set of noise vectors, with 10 samples per class. The images are arranged in a grid with one class per row, so that both the sample quality and the conditioning can be checked at a glance. Using the same noise vectors every epoch makes it possible to follow the improvement of each sample over training.
 
## Deviations from the Original Papers
 
1. Image size is 32x32 instead of 64x64 as in the DCGAN paper[^1], so the generator uses a 4x4 starting map and three stride-2 upsampling steps.
2. Conditioning is done through a learnable embedding layer instead of a one-hot vector as in the Conditional GAN paper[^2]. This is mathematically equivalent to passing a one-hot vector through a bias-free linear layer. The label is concatenated with `z` in the generator and added as extra input channels in the discriminator.
3. Maximizes $\log D(G(z|y)|y)$ instead of minimizing $\log (1 - D(G(z|y)|y))$[^3].
4. One-sided label smoothing (0.9) instead of hard real labels.
5. Trained on CIFAR-10 instead of the large-scale datasets used in the DCGAN paper[^1].
 
[^1]: Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
[^2]: Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
[^3]: Goodfellow, Ian J., et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).


