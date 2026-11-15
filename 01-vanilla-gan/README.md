# Vanilla Generative Adversarial Network

Implementation of Ian Goodfellow's original GAN paper **Generative Adversarial Nets**[^1] using Multi-layer Perceptrons (MLPs) on MNIST dataset in PyTorch. 

## How to run

```bash
python train.py --config config.yaml
```

## Network Architecture

### Generator

Maps a latent vector `z` of size 100 to a 28x28 grayscale image with pixel values in the range \[-1. 1\].

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Input `z` | 100 | Gaussian noise |
| Linear + LeakyReLU(0.2) | 128 | No BatchNorm on the first block |
| Linear + BatchNorm + LeakyReLU(0.2) | 256 | |
| Linear + BatchNorm + LeakyReLU(0.2) | 512 | |
| Linear + BatchNorm + LeakyReLU(0.2) | 1024 | |
| Linear + Tanh | 784 | Reshaped to (1, 28, 28) |

### Discriminator

Maps a 28x28 grayscale image to a single logit score.

| Layer | Output shape | Notes |
|-------|--------------|-------|
| Flatten | 784 | |
| Linear + LeakyReLU(0.2) + Dropout(0.3) | 512 | |
| Linear + LeakyReLU(0.2) + Dropout(0.3) | 256 | |
| Linear | 1 | Raw logit (no Sigmoid) |

### Loss and Optimizer

For loss function, both generator and discriminator networks use Binary Cross Entropy loss with logit scores. For the generator loss, instead of minimizing $\log (1 - D(G(z)))$ as in the original paper, $\log D(G(z))$ is maximized, as suggested as an alternative in the original paper[^1].

For optimizer, both generator and discriminator networks use their own Adam optimizer with learning rate of 0.0002, beta1 of 0.5 and beta2 of 0.999.

## Training Loop

Each iteration processes one minibatch. The parameters of the discriminator are updated for $k=1$ steps first, followed by the parameter update of the generator.

### Discriminator Training Step

1. Generate fake images $G(z)$ from latent noise vector `z`.
2. Compute binary cross entropy loss using logit scores on real images.
3. Compute binary cross entropy loss using logit scores on fake images.
4. Compute average discriminator loss by adding the losses on real and fake images and divide by 2.
5. Update discriminator parameters.

### Generator Training Step

1. Take the generated fake images from discriminator training step and compute binary cross entropy loss using logit scores against the label `1`.
2. Update generator parameters.

## Deviations from the Original Paper

1. Used Leaky ReLU activation instead of Maxout.
2. Added Batch Normalization in generator network is added.
3. Maximizes $\log D(G(z))$ instead of minimizing $\log (1 - D(G(z)))$ [^1]
4. One-sided label smoothing (0.9) instead of hard real labels

## References

[^1]: Goodfellow, Ian J., et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).