import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler
import numpy as np
import copy
from collections import Counter

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size, lr, beta, lamda, local_epochs, logger):

        self.device = device
        self.model = copy.deepcopy(model[0])
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.id = id  # not integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.lr = lr
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        sampled_train_data = self.get_sampled_train_data(train_data)
        self.sampled_train_dataloaderfull = DataLoader(sampled_train_data, len(sampled_train_data))
        sampler_validate_data, sampler_batch_size = self.get_WeightedRandomSampler([data[1] for data in sampled_train_data])
        self.validaterandomloader = DataLoader(sampled_train_data, sampler_batch_size, sampler=sampler_validate_data)
        self.iter_validaterandomloader = iter(self.validaterandomloader)
        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.logger = logger
        self.gradients = []
        self.logger.info("Train Samples: {} Test Samples: {} Validate Samples: {} Sampled Validate Samples: {}".format(self.train_samples, self.test_samples, len(sampled_train_data), sampler_batch_size))
    
    def get_lr_gradients(self, model_backup):
        grads = []
        for param, param_old in zip(self.model.parameters(), model_backup.parameters()):
            grads.append((param_old.data.clone().detach() - param.data.clone().detach()))
        return grads

    def get_sampled_train_data(self, data):
        if len(data) > 10:
            sampled_size = int(len(data) * 0.333)
        else:
            sampled_size = len(data)
        _size = len(data) - sampled_size
        sampled_train_data, _ = torch.utils.data.random_split(data, [sampled_size, _size])
        return sampled_train_data

    def get_WeightedRandomSampler(self, data):
        weight = Counter(data)
        weight = dict(weight)
        weight = {k: 1 / weight[k] for k in weight}
        samples_weight = np.array([weight[t] for t in data])
        if len(data) >= 200:
            sampler_batch_size = len(data) // 10
        elif len(data) >= 100:
            sampler_batch_size = len(data) // 5
        elif len(data) >= 50:
            sampler_batch_size = len(data) // 3
        elif len(data) >= 20:
            sampler_batch_size = len(data) // 2
        else:
            sampler_batch_size = len(data)
        return WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=False), sampler_batch_size

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test(self, per):
        self.model.eval()
        test_acc = 0
        if per:
            self.update_parameters(self.persionalized_model)
        with torch.no_grad():
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        if per:
            self.update_parameters(self.local_model)
        return test_acc, self.test_samples

    def test_on_model(self, model, one_step=False, use_full_test_data=False):
        test_acc = 0
        self.update_parameters(model.parameters())
        # if one_step:
        #     self.train_one_step()
        self.model.eval()
        if use_full_test_data:
            with torch.no_grad():
                for x, y in self.sampled_train_dataloaderfull:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            self.update_parameters(self.local_model)
            return test_acc / self.test_samples
        else:
            with torch.no_grad():
                x, y = self.get_next_random_validate_batch()
                output = self.model(x)
                test_acc = (torch.sum(torch.argmax(output, dim=1) == y)).item()
            self.update_parameters(self.local_model)
            return test_acc / y.shape[0]

    def train_error_and_loss(self, per):
        self.model.eval()
        train_acc = 0
        loss = 0
        if per:
            self.update_parameters(self.persionalized_model)
        with torch.no_grad():
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)
        if per:
            self.update_parameters(self.local_model)
        return train_acc, loss, self.train_samples
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_random_validate_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_validaterandomloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_validaterandomloader = iter(self.validaterandomloader)
            (X, y) = next(self.iter_validaterandomloader)
        return (X.to(self.device), y.to(self.device))

    '''
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    '''