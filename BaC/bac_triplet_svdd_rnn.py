import torch as th
import torch.nn as nn
from tqdm import tqdm

import numpy as np

from .bac_rnn import BaCRNN
from imitation.util import util
import copy
class BaC_RNN_Triplet_SVDD(BaCRNN):
    def __init__(
        self,
        train_env,
        bac_classifier,
        expert_data,
        bc_trainer=None,
        not_expert_data=None,
        nepochs: int = 10,
        batch_size: int = 10,
        triplet_epochs: int = 40,
        svdd_epochs: int = 60,
        R = 0.0,
        nu = 0.1,
        use_fixed_anchor = True,
    ):
        super().__init__(
            train_env,
            bac_classifier,
            expert_data,
            bc_trainer,
            not_expert_data,
            nepochs=nepochs,
            batch_size=batch_size,
        )

        self.bac_triplet_optimizer = th.optim.Adam(self.bac_classifier.parameters())
        self.bac_triplet_loss = nn.TripletMarginLoss(margin=1.0)

        self.bac_svdd_optimizer = th.optim.Adam(self.bac_classifier.parameters())
        
        self.collect_max = 100
        self.triplet_epochs = triplet_epochs
        self.svdd_epochs = svdd_epochs
        self.R = th.tensor(R, device=self.bac_classifier.device())
        self.nu = nu
        self.objective = 'soft-boundary'

        self.use_fixed_anchor = use_fixed_anchor

        if self.use_fixed_anchor:
            self.fixed_anchor = copy.deepcopy(self.expert_data[:self.batch_size])
            self.expert_data = expert_data[self.batch_size:]
            self.expert_dataloader = util.endless_iter(expert_data)

    def bac_triplet_epoch(self):
        self.bac_classifier.train()

        full_loss = 0
        for j in range(10):

            # anchor = [next(self.expert_dataloader) for _ in range(self.batch_size)]
            if self.use_fixed_anchor:
                anchor = copy.deepcopy(self.fixed_anchor)
            else:
                anchor = [next(self.expert_dataloader) for _ in range(self.batch_size)]
            
            positive = [next(self.expert_dataloader) for _ in range(self.batch_size)]
            if j % 3 == 0:
                negative = [
                    next(self.not_expert_from_bc_dataloader)
                    for _ in range(self.batch_size)
                ]
            elif j % 3 == 1:
                negative = [
                    next(self.not_expert_dataloader) for _ in range(self.batch_size)
                ]
            else:
                negative = [
                    next(self.not_expert_from_expert_dataloader)
                    for _ in range(self.batch_size)
                ]

            anchor_hidden = self.bac_classifier.embedding(anchor)[1]
            positive_hidden = self.bac_classifier.embedding(positive)[1]
            negative_hidden = self.bac_classifier.embedding(negative)[1]

            triplet_loss = self.bac_triplet_loss(
                anchor_hidden, positive_hidden, negative_hidden
            )
            self.bac_triplet_optimizer.zero_grad()
            triplet_loss.backward()
            self.bac_triplet_optimizer.step()
            full_loss += triplet_loss.data
        print(f"triplet loss {full_loss / 10}")


    def bac_svdd_epoch(self, init = False):
        self.bac_classifier.train()
        if init:
            self.c = self.init_center_c()
            print(self.c)

        
        full_loss = 0
        for j in range(10):
                # Update network parameters via backpropagation: forward + backward + optimize
            expert_batch = [next(self.expert_dataloader) for _ in range(self.batch_size)]
            expert_hidden = self.bac_classifier.embedding(expert_batch)[1].view(self.batch_size, -1)
            # dist = th.sum((expert_hidden - self.c) ** 2, dim=1)
            dist = ((expert_hidden - self.c) ** 2)
            dist = th.sum(dist)

            # print(dist)
            if self.objective == 'soft-boundary':
                scores = dist - self.R ** 2
                loss = self.R ** 2 + (1 / self.nu) * th.mean(th.max(th.zeros_like(scores), scores))

            # else:
            #     loss = th.mean(dist)
            self.bac_svdd_optimizer.zero_grad()
            loss.backward()
            self.bac_svdd_optimizer.step()
            # Update hypersphere radius R on mini-batch distances
            if (self.objective == 'soft-boundary'):
                self.R.data = th.tensor(self.get_radius(dist, self.nu), device=self.bac_classifier.device())
            
            full_loss += loss.item()

        print(f"svdd loss {full_loss / 10}")
        print(f"R = {self.R}")

    def triplet_warmstart(self, filter=True):
        self.collect_not_expert_from_bc(filter=False)
        self.collect_not_expert(filter=False)
        self.collect_not_expert_from_expert(filter=False)

        for i in tqdm(range(self.triplet_epochs)):
            # collect not expert after every 20 epochs
            # if i % 30 == 0 and i > 1:
            #     self.collect_not_expert_from_bc(filter=False)
            #     self.collect_not_expert(filter=False)
            #     self.collect_not_expert_from_expert(filter=False)

            self.bac_triplet_epoch()
        
        print("Triplet Warmstart done")

    def train_bac_2halfs(self):
        self.triplet_warmstart()
        self.bac_svdd_epoch(init = True)
        for i in tqdm(range(self.svdd_epochs)):
            self.bac_svdd_epoch()

        print("BaC training done")

    def predict(self, traj):
        """
        predicts and returns either logit or prop
        """
        self.bac_classifier.eval()

        #traj is a single datapoint not batch
        expert_hidden = self.bac_classifier.embedding(traj)[1].view(-1)
        dist = ((expert_hidden - self.c) ** 2).view(-1)
        dist = th.sum(dist)
        if np.abs(dist.item()) <= self.R.data:
            print("in circle", dist)
        else:
            print("out circle", dist) 
        # print(dist)
        return dist

    def init_center_c(self, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = th.zeros(self.bac_classifier.in_size, device=self.bac_classifier.device())

        
        with th.no_grad():
            data = copy.deepcopy(self.fixed_anchor)
            for i in data:
                # single datapoint not batch
                outputs = self.bac_classifier.embedding(i)[1].view(-1)
                # print(outputs.shape)
                n_samples += 1
                c += outputs

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        
        return c


    def get_radius(self, dist: th.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)