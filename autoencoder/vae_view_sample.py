import autoencoder.vae_model as vae_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import pickle5 as pickle


train_set = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)
with open("data/minigrid_empty_obs.pkl", "rb") as f:
    train_set = torch.tensor(pickle.load(f))/255

model = torch.load("models/ae_minigrid_empty.pt")

for i in range(10):
    
    img = train_set[i] 
    
    pred_img = model.generate(torch.unsqueeze(img, 0))

    pred_img = pred_img.detach()[0]
    _, ax = plt.subplots(1,2)
    ax[0].imshow(img.permute(1,2,0))
    ax[1].imshow(pred_img.permute(1,2,0))
    # print(pred_img)
    # model.calc_loss(torch.unsqueeze(img, 0))
    plt.show()
    
    