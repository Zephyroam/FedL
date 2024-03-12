import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import copy

# Implementation for FedAvg clients

class UserAFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, lr, beta, lamda,
                 local_epochs, logger):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, lr, beta, lamda,
                         local_epochs, logger)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        LOSS = 0
        model_backup = copy.deepcopy(self.model)
        self.model.train()
        self.losses = np.zeros(self.local_epochs)
        for epoch in range(self.local_epochs):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.losses[epoch] = loss.item()
            self.optimizer.step()
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
        self.lr_gradients = self.get_lr_gradients(model_backup)
        self.losses = np.mean(self.losses)
        return LOSS