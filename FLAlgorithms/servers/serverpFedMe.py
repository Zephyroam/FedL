import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, AverageMeter
import numpy as np
import time

# Implementation for pFedMe Server


class pFedMe(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters,
                 local_epochs, num_users, K, personal_lr, times, aggregation_method, generation, individual, gamma, topk, logger, ea_alg, q):
        super().__init__(device, dataset, algorithm,
                         model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, num_users, times, aggregation_method, generation, individual, gamma, topk, True, logger, ea_alg, q)

        # Initialize data for all users
        self.K = K
        self.personal_lr = personal_lr
        for i in range(self.total_users):
            id, train, test = read_user_data(i, self.data, dataset)
            user = UserpFedMe(device, id, train, test, model, batch_size,
                              lr, beta, lamda, local_epochs, K, personal_lr, logger)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.logger.info("Number of users / total users: {} / {}".format(num_users, self.total_users))
        self.logger.info("Finished creating pFedMe server.")
        self.average_time = AverageMeter()

    def train(self):
        for glob_iter in range(self.num_glob_iters):
            self.logger.info("Round number: {}".format(glob_iter))
            # send all parameter for users
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            # self.logger.info("Evaluate global model")
            # self.evaluate(per=False)

            # do update for all users not only selected users
            for user in self.users:
                user.train()

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(self.num_users)

            # Evaluate gloal model on user for each interation
            self.evaluate(per=True)
            
            start_time = time.time()
            self.aggregate_parameters()
            execut_time = time.time() - start_time
            self.average_time.update(execut_time)
            self.logger.info('Aggregation executeTime: {}, average executeTime: {}'.format(execut_time, self.average_time.avg))

        self.save_results()
        # self.save_model()
