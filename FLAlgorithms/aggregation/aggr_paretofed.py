import geatpy as ea
import torch
import torch.nn as nn
import os
import numpy as np
import copy
import time
import utils.model_utils as model_utils

def get_ea_alg(ea_alg):
    if ea_alg == 'nsga2':
        return ea.moea_NSGA2_templet
    elif ea_alg == 'awga':
        return ea.moea_awGA_templet
    elif ea_alg == 'moead':
        return ea.moea_MOEAD_templet
    elif ea_alg == 'nsga3':
        return ea.moea_NSGA3_templet
    elif ea_alg == 'rvea':
        return ea.moea_RVEA_templet

class Problem_aggr(ea.Problem):
    def __init__(self, num_users, coef_standard, model, selected_users, test_function):
        name = 'Problem_aggr'
        M = 2
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * num_users  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * num_users  # 决策变量下界
        ub = [1] * num_users  # 决策变量上界
        lbin = [1] * num_users  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * num_users  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.coef_standard = coef_standard
        self.model = model
        self.selected_users = selected_users
        self.test_function = test_function
        self.get_model_standard()
        ea.Problem.__init__(self, name, M, maxormins, num_users, varTypes, lb, ub, lbin, ubin)

    def get_model_standard(self):
        self.model_standard = copy.deepcopy(self.model)
        for param in self.model_standard.parameters():
            param.data = torch.zeros_like(param.data)

        for i, user in enumerate(self.selected_users):
            for server_param, user_param in zip(self.model_standard.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * self.coef_standard[i]
    
    '''
    def get_constraints(self):
        diff_models = []
        for standard_param, now_param in zip(self.model_standard.parameters(), self.model.parameters()):
            diff_models.extend(torch.abs(standard_param.data - now_param.data).reshape(-1))
        # print(diff_models)
        diff_models = torch.Tensor(diff_models).sum(dim=0)
        return diff_models
    '''

    def get_obj(self, performance):
        performance = np.array(performance)
        average = np.sum(performance * np.array(self.coef_standard.tolist()))
        std = np.std(performance)
        return average, std

    def evalVars(self, Vars):
        averages, stds, cv1, cv2 = [], [], [], []
        for var in Vars:
            selected_coef = torch.Tensor(var)
            performance = self.test_function(self.selected_users, self.model, selected_coef)
            average, std = self.get_obj(performance)
            averages.append(average)
            stds.append(std)
            # cv1.append(np.abs(np.sum(var) - 1))
            # cv2.append(self.get_constraints() - 10)
        averages = np.array(averages).reshape(-1, 1)
        stds = np.array(stds).reshape(-1, 1)
        # cv1 = np.array(cv1).reshape(-1, 1)
        # cv2 = np.array(cv2).reshape(-1, 1)
        ObjV = np.hstack([averages, stds])
        # CV = np.hstack([cv1])
        return ObjV
        # return ObjV, CV


class Aggr_ParetoFed(object):
    def __init__(self, device, algorithm, model, num_users, total_users, num_glob_iters, generation, individual, topk, per, logger, ea_alg):
        self.device = device
        self.one_step = algorithm == 'PerAvg'
        self.model = model
        self.num_users = num_users
        self.total_users = total_users
        self.num_glob_iters = num_glob_iters
        self.generation = generation
        self.individual = individual
        self.topk = topk
        self.per = per
        self.coef = torch.ones(self.total_users).to(self.device)
        self.initialized_coef = False
        self.now_epoch = 0
        self.logger = logger
        self.ea_alg = get_ea_alg(ea_alg)

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

    def aggregate(self, selected_users, model, selected_coef):
        for param in model.parameters():
            param.data = torch.zeros_like(param.data)

        selected_coef = selected_coef / selected_coef.sum()
        for i, user in enumerate(selected_users):
            for server_param, user_param in zip(model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * selected_coef[i]

    def test_on_given_coef(self, selected_users, model, selected_coef):
        self.now_epoch += 1
        use_full_test_data = self.now_epoch >= int(0.85 * self.generation) * self.individual
        with torch.no_grad():
            self.aggregate(selected_users, model, selected_coef)
            
            performance = []
            for user in selected_users:
                perf = user.test_on_model(model, self.one_step, use_full_test_data)
                performance.append(perf)
        return performance

    def get_coefs(self, optimal_coef, optimal_performance):
        selected_coefs = []
        selected_coef = optimal_coef[np.where(optimal_performance[:, 0]==np.max(optimal_performance[:, 0]))[0][0]]
        selected_coefs.append(selected_coef)
        if self.topk != 1 and optimal_performance.shape[0] > 1:
            tops = np.argsort(optimal_performance[:, 0])[-self.individual // 2:-1]
            selected_index = np.random.choice(tops, min(self.topk - 1, len(tops)), replace=False)
            selected_coef = optimal_coef[selected_index]
            selected_coefs.extend(selected_coef)
        return selected_coefs

    def aggregate_parameters(self, model, selected_users):
        ids = [int(user.id[2:]) for user in selected_users]
        ids_tensor = torch.LongTensor(ids).to(self.device)
        selected_coef_backup = torch.index_select(self.coef_standard, 0, ids_tensor)
        selected_coef_backup = selected_coef_backup / selected_coef_backup.sum()
        
        # if use_full_test_data:
        #     self.logger.info('Using full test data')
        # else:
        #     self.logger.info('Using sampled test data')
        problem_aggr = Problem_aggr(self.num_users, selected_coef_backup, model, selected_users, self.test_on_given_coef)
        algorithm = self.ea_alg(problem_aggr,
                                      ea.Population(Encoding='RI', NIND=self.individual),
                                      MAXGEN=self.generation,
                                      logTras=1)
                        
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        self.logger.info('The number of non-dominated solutions is: {}'.format(res['optPop'].sizes))
        optimal_coef, optimal_performance = res['Vars'], res['ObjV']
        selected_coefs = self.get_coefs(optimal_coef, optimal_performance)
        # self.logger.info(selected_coefs)
        models = []
        for coef in selected_coefs:
            model_new = copy.deepcopy(model)
            self.aggregate(selected_users, model_new, coef)
            models.append(model_new)
        return models