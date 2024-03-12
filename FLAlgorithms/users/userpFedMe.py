import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients


class UserpFedMe(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, lr, beta, lamda,
                 local_epochs, K, personal_lr, logger):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, lr, beta, lamda,
                         local_epochs, logger)

        self.K = K
        self.optimizer_pFedMe = pFedMeOptimizer(self.model.parameters(), lr=personal_lr, lamda=self.lamda)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        LOSS = 0
        self.model.train()
        for epoch in range(self.local_epochs):  # local update

            self.model.train()
            X, y = self.get_next_train_batch()

            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                self.optimizer_pFedMe.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.persionalized_model, _ = self.optimizer_pFedMe.step(self.local_model)
            # self.persionalized_model is \theta in the paper
            # self.local_model is w in the paper

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model, self.local_model):
                localweight.data = localweight.data - self.lamda * \
                    self.lr * (localweight.data - new_param.data)

        # update local model as local_weight_upated
        self.update_parameters(self.local_model)

        return LOSS
