import torch
import os
import numpy as np
import h5py
import copy
import FLAlgorithms.aggregation as aggregation
from utils.model_utils import read_data
from utils.plot_utils import get_log_name, get_log_name_prefix


class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, num_users, times, aggregation_method, generation, individual, gamma, topk, per, logger, ea_alg, q):
        self.device = device
        self.one_step = algorithm == 'PerAvg'
        self.dataset = dataset
        self.data = read_data(dataset)
        self.total_users = len(self.data[0])
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.total_train_samples = 0
        self.models = [copy.deepcopy(model[0])]
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_glob_std = [], [], [], []
        self.times = times
        self.aggregation_method = aggregation_method
        self.topk = topk
        self.aggregation = aggregation.get_aggr_method(aggregation_method, num_glob_iters, device, algorithm, model[0], num_users, self.total_users, lr, generation, individual, gamma, topk, per, logger, ea_alg, q)
        self.path = './results/{}/{}/{}/{}/{}_{}_{}_{}_{}/'.format(dataset, model[1], aggregation_method, algorithm, generation, individual, gamma, topk, ea_alg)
        self.logger = logger
        
    def send_parameters(self):
        for user in self.users:
            if len(self.models) == 1:
                user.set_parameters(self.models[0])
            else:
                performances = np.array([])
                for model in self.models:
                    perf = user.test_on_model(model, self.one_step)
                    performances = np.append(performances, perf)
                user.set_parameters(self.models[np.where(performances==np.max(performances))[0][0]])
    
    def send_parameters_test_afl(self):
        for user in self.users:
            user.set_parameters(self.model_test)
                
    def aggregate_parameters(self):
        if hasattr(self.aggregation, 'initialized_coef') and self.aggregation.initialized_coef == False:
            self.aggregation.initialize_coef(self.users)
            self.aggregation.initialized_coef = True
        self.models = self.aggregation.aggregate_parameters(copy.deepcopy(self.models[0]), self.selected_users)

    def aggregate_parameters_afl(self, round_num):
        assert self.algorithm == 'FedAvg' and self.aggregation_method == 'AFL'
        self.models, self.model_test = self.aggregation.aggregate_parameters(copy.deepcopy(self.models[0]), self.model_test, self.selected_users, round_num)

    def select_users(self, num_users):
        if num_users == len(self.users):
            self.logger.info("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        return np.random.choice(self.users, num_users, replace=False)

    # Save loss, accuracy to h5 flie
    def save_results(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        alg = get_log_name(get_log_name_prefix(self.batch_size, self.lr, self.beta, self.lamda, self.local_epochs, self.num_users), self.times)
        with h5py.File(self.path + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
            hf.create_dataset('rs_glob_std', data=self.rs_glob_std)
            hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            hf.close()

    def test(self, per):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        acc_users = []
        for c in self.users:
            ct, ns = c.test(per=per)
            tot_correct.append(ct)
            num_samples.append(ns)
            acc_users.append(ct / ns)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, acc_users

    def train_error_and_loss(self, per):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss(per=per)
            tot_correct.append(ct)
            num_samples.append(ns)
            losses.append(cl)
        ids = [c.id for c in self.users]
        return ids, num_samples, tot_correct, losses

    def evaluate(self, per):
        stats = self.test(per=per)
        stats_train = self.train_error_and_loss(per=per)
        self.record_log(stats, stats_train)

    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test(per=False)
        stats_train = self.train_error_and_loss(per=False)

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        self.record_log(stats, stats_train)

    def record_log(self, stats, stats_train):
        glob_acc = np.sum(stats[2]) / np.sum(stats[1])
        glob_std = np.std(stats[3])
        train_acc = np.sum(stats_train[2]) / np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1],
                         stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_glob_std.append(glob_std)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.logger.info("Average Global Accuracy: {}".format(glob_acc))
        self.logger.info("Accuracy std among users: {}".format(glob_std))
        # self.logger.info("Average Global Trainning Accuracy: {}".format(train_acc))
        # self.logger.info("Average Global Trainning Loss: {}".format(train_loss))

    '''
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)
    '''