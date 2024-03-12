import torch
import os
import numpy as np

class Aggr_Average(object):
    def __init__(self):
        pass
    
    def aggregate_parameters(self, model, selected_users):
        for param in model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0

        for user in selected_users:
            total_train += user.train_samples

        for user in selected_users:
            ratio = user.train_samples / total_train
            for server_param, user_param in zip(model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio
        return [model]