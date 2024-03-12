import torch
import torch.nn as nn
import os
import numpy as np
import copy

class Aggr_MtoSFed(object):
    def __init__(self, device, algorithm, model, num_users, total_users, gamma, per):
        self.device = device
        self.one_step = algorithm == 'PerAvg'
        self.model = model
        self.num_users = num_users
        self.total_users = total_users
        self.coef = nn.Parameter(torch.ones(self.total_users).to(self.device))
        self.initialized_coef = False
        self.gamma = gamma
        self.per = per

    def initialize_coef(self, users):
        total_train = 0
        for user in users:
            total_train += user.train_samples
        new_coef = [0. for _ in users]
        for user in users:
            id = int(user.id[2:])
            new_coef[id] = user.train_samples / total_train
        self.coef.data = torch.Tensor(new_coef).to(self.device)
        self.coef_standard = self.coef.clone()
 
    def aggregate_parameters(self, model, selected_users):
        ids = [int(user.id[2:]) for user in selected_users]
        ids_tensor = torch.LongTensor(ids).to(self.device)
        selected_coef = torch.index_select(self.coef_standard, 0, ids_tensor)
        selected_coef = selected_coef / selected_coef.sum()
        for param in model.parameters():
            param.data = torch.zeros_like(param.data)

        accs = []
        for user in selected_users:
            acc = user.test(self.per)
            accs.append(acc[0] / acc[1])
        acc_average = np.mean(accs)

        param_average = []
        for i, user in enumerate(selected_users):
            for j, param in enumerate(user.model.parameters()):
                if i == 0:
                    param_average.append(param.data.clone())
                else:
                    param_average[j] += param.data.clone()
        param_average = [param / self.num_users for param in param_average]

        for i, user in enumerate(selected_users):
            for j, (server_param, user_param) in enumerate(zip(model.parameters(), user.model.parameters())):
                server_param.data = server_param.data + user_param.data.clone() * selected_coef[i] - self.gamma * 2 / self.num_users * (accs[i] - acc_average) * (user_param.data.clone() - param_average[j].clone())
        return [model]
