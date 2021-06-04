import vae_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

train_set = torchvision.datasets.MNIST(
    root = './data',
    train = True,
    download = False,
    transform = transforms.ToTensor()#transforms.Compose([
        # transforms.ToTensor()
    # ])
)

print("Train Shape:", type(train_set), type(train_set[0]), train_set[0][0].shape)
plt.imshow(train_set[4][0].permute(1,2,0), cmap='gray')
plt.show()

vae = vae_model.VanillaVAE()