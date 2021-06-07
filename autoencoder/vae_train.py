import autoencoder.vae_model as vae_model
from modules.cnn_autoencoder import CNNAutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import pickle5 as pickle


from utils.env_utils import minigrid_get_env

train_set = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()#transforms.Compose([
        # transforms.ToTensor()
    # ])
)
with open("data/minigrid_empty_obs.pkl", "rb") as f:
    train_set = torch.tensor(pickle.load(f))/255

# print("Train Shape:", type(train_set), type(train_set[0]), train_set.shape)
# for i in range(10):
#     plt.imshow(train_set[i].permute(1,2,0))
#     plt.show()

# model = vae_model.VanillaVAE(in_channels = 3, latent_dim=256, hidden_dims=[64, 128, 256, 512])
# model = vae_model.AutoEncoder()
# model = torch.load("models/ae_cifar3.pt")

venv = minigrid_get_env("MiniGrid-Empty-Random-6x6-v0", n_envs=1, partial=False)
model = CNNAutoEncoder(action_space=venv.action_space, observation_space=venv.observation_space)
model = torch.load("models/ae_minigrid_empty.pt")


train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
optimizer = optim.Adam(model.parameters(), lr = 0.00005)

for epoch in range(100):
    
    total_loss = 0
    cnt = 0
    for batch in tqdm(train_loader):
        # cnt+=1
        # if cnt > 2:
        #     break

        images = batch
        
        loss = model.calc_loss(images)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print("epoch:", epoch, ", loss:", total_loss)
    # if (epoch+1)%10==0:
    #     pred_image = model.generate(torch.unsqueeze(train_set[4], 0))
        
    #     pred_image = pred_image.detach() 
    #     plt.imshow(pred_image[0].permute(1,2,0))
    #     plt.show()
    
    torch.save(model, "models/ae_minigrid_empty.pt")