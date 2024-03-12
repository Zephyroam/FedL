import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import MySGD, FEDLOptimizer
from FLAlgorithms.users.userbase import User

# Implementation for Per-FedAvg clients

class UserPerAvg(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, lr, beta, lamda,
                 local_epochs, logger):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size, lr, beta, lamda,
                         local_epochs, logger)

        self.optimizer = MySGD(self.model.parameters(), lr=self.lr)

    def train(self):
        LOSS = 0
        self.model.train()
        for epoch in range(self.local_epochs):  # local update 
            self.model.train()

            #step 1
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            #step 2
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(beta=self.beta)

            # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)

        return LOSS    

    def train_one_step(self):
        self.model.train()
        #step 1
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        
        #step 2
        X, y = self.get_next_test_batch()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)