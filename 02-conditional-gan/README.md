# Conditional Generative Adversarial Network

Implementation of **Conditional Generative Adversarial Network** which also feeds the generator and the discriminator with class labels as a condition to control the modes of data being generated[^1]. The network is implemented in PyTorch and trained on Fashion MNIST dataset.

## How to run

```bash
python train.py --config config.yaml
```

## Network Architecture

### Generator

Maps a latent vector `z` of size 100 and a class label `y` to a 28x28 grayscale image with pixel values in the range \[-1, 1\]. The class label is converted to a 10-dimensional embedding and concatenated with `z`.

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input `z` | 100 | Gaussian noise |
| Label embedding of `y` | 10 | `nn.Embedding(10, 10)` |
| Concatenate `[z, label_embedding(y)]` | 110 | |
| Linear + LeakyReLU(0.2) | 128 | No BatchNorm on the first block |
| Linear + BatchNorm + LeakyReLU(0.2) | 256 | |
| Linear + BatchNorm + LeakyReLU(0.2) | 512 | |
| Linear + BatchNorm + LeakyReLU(0.2) | 1024 | |
| Linear + Tanh | 784 | Reshaped to (1, 28, 28) |

### Discriminator

Maps a 28x28 grayscale image and a class label `y` to a single logit score, which indicates whether the image is real or fake for the given class. The discriminator has its own embedding which is not shared with the generator.

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Flatten | 784 | |
| Label embedding of `y` | 10 | `nn.Embedding(10, 10)` |
| Concatenate `[image, label_embedding(y)]` | 794 | |
| Linear + LeakyReLU(0.2) + Dropout(0.3) | 512 | |
| Linear + LeakyReLU(0.2) + Dropout(0.3) | 256 | |
| Linear | 1 | Raw logit (no Sigmoid) |

### Loss and Optimizer

For loss function, both generator and discriminator networks use Binary Cross Entropy loss with logit scores. For the generator loss, instead of minimizing $\log (1 - D(G(z|y)|y))$ as in the original GAN paper, $\log D(G(z|y)|y)$ is maximized, as suggested as an alternative in the original GAN paper[^2].
 
For optimizer, both generator and discriminator networks use their own Adam optimizer with learning rate of 0.0002, beta1 of 0.5 and beta2 of 0.999.

## Training Loop

Each iteration processes one minibatch. The parameters of the discriminator are updated for $k=1$ steps first, followed by the parameter update of the generator.

### Discriminator Training Step

1. Sample random class labels `gen_labels` uniformly from the 10 classes and generate fake images $G(z|y)$ from latent noise vector `z` amd `gen_labels`.
2. Compute binary cross entropy loss using logit scores on real images paired with their true labels, against the smoothed label `0.9`.
3. Compute binary cross entropy loss using logit scores on fake images paired with `gen_labels`, against the label `0`.
4. Compute average discriminator loss by adding the losses on real and fake images and divide by 2.
5. Update discriminator parameters.

### Generator Training Step

1. Take the generated fake images from discriminator training step. Compute binary cross entropy loss by using logit scores against the label `1`. The fake images are paired with the same `gen_labels` used to generate them.
2. Update generator parameters.

## Deviations from the Original Paper

1. Used Leaky ReLU activation instead of Maxout.
2. Added Batch Normalization in generator network (except the first layer).
3. Maximizes $\log D(G(z|y)|y)$ instead of minimizing $\log (1 - D(G(z|y)|y))$[^2].
4. One-sided label smoothing (0.9) instead of hard labels for real images.
5. Class labels are fed through a learnable embedding layer instead of a one-hot vector.


[^1]: M. Mirza and S. Osindero, “Conditional Generative Adversarial Nets,” 2014, arXiv. doi: 10.48550/arXiv.1411.1784.
[^2]: I. J. Goodfellow et al., “Generative Adversarial Nets,” Adv. Neural Inf. Process. Syst., vol. 27, 2014.
```
