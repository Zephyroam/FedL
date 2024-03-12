import torch
import os

from FLAlgorithms.users.userafl import UserAFL
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, AverageMeter
import numpy as np
import time

# Implementation for FedAvg Server


class FedAFL(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, num_users, times, aggregation_method, generation, individual, gamma, topk, logger, ea_alg, q):
        super().__init__(device, dataset, algorithm,
                         model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, num_users, times, aggregation_method, generation, individual, gamma, topk, False, logger, ea_alg, q)

        # assert local_epochs == 1, "Only support local_epochs = 1"

        # Initialize data for all  users
        for i in range(self.total_users):
            id, train, test = read_user_data(i, self.data, dataset)
            user = UserAFL(device, id, train, test, model,
                           batch_size, lr, beta, lamda, local_epochs, logger)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.logger.info("Number of users / total users: {} / {}".format(num_users, self.total_users))
        self.logger.info("Finished creating FedAFL server.")
        self.model_test = self.models[0]
        self.average_time = AverageMeter()

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            self.logger.info("Round number: {}".format(glob_iter))

            # Evaluate model each interation
            self.logger.info("Evaluate global model")
            self.send_parameters_test_afl()
            self.evaluate(per=False)
            
            # send all parameter for users
            self.send_parameters()

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(self.num_users)
            for user in self.selected_users:
                user.train()
            
            start_time = time.time()
            self.aggregate_parameters_afl(glob_iter)
            execut_time = time.time() - start_time
            self.average_time.update(execut_time)
            self.logger.info('Aggregation executeTime: {}, average executeTime: {}'.format(execut_time, self.average_time.avg))

        self.save_results()
        # self.save_model()
