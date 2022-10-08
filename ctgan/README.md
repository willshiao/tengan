# WGAN with gradient penalty and consistency term

Pytorch implementation of [Improving the Improved Training of Wasserstein GANs](https://arxiv.org/abs/1803.01541) by Xiang Wei, Boqing Gong, Zixia Liu, Wei Lu and Liqiang Wang

## Usage

Set up a generator and discriminator model

```python
from models import Generator, Discriminator
generator = Generator(img_size=(32, 32, 1), latent_dim=100, dim=16)
discriminator = Discriminator(img_size=(32, 32, 1), dim=16)
```

The generator and discriminator are built to automatically scale with image sizes, so you can easily use images from your own dataset.

Train the generator and discriminator with the CT-GAN loss

```python
import torch
# Initialize optimizers
G_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(.9, .99))
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(.9, .99))

# Set up trainer
from training import Trainer
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())

# Train model for 200 epochs
trainer.train(data_loader, epochs=200, save_training_gif=True)
```

This will train the models and generate a gif of the training progress.

Note that CT-GANs take a *long* time to converge.

## Sources and inspiration

* https://github.com/caogang/wgan-gp
* https://github.com/kuc2477/pytorch-wgan-gp
* https://github.com/EmilienDupont/wgan-gp

Any improvements to the code are welcome!
