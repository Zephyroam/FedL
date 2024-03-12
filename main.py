#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers import *
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
from utils.model_utils import *
import torch
import math

torch.manual_seed(0)

def main(args, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters,
         local_epochs, numusers, K, personal_lr, times, aggr_method, generation, individual, gamma, topk, ea_alg, q, gpu):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    path = './results/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}/'.format(dataset, model, aggr_method, algorithm, generation, individual, gamma, topk, ea_alg, q)
    logger = Log(path, 'log').getlog()
    print_log(args, logger)
    for i in range(times):
        logger.info("Running time: {}".format(i))
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60,10).to(device), model
                    
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
                
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            else: 
                model = DNN(60,20,10).to(device), model
        
        if(model == "resnet20"):
            if(dataset == "Cifar10"):
                model = resnet20().to(device), model
            else:
                raise ValueError("Resnet20 is only for Cifar10")

        # select algorithm
        if algorithm == "FedAvg" and aggr_method == 'AFL':
            server = FedAFL(device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, numusers, i, aggr_method, generation, individual, gamma, topk, logger, ea_alg, q)
        elif algorithm == "FedAvg" and aggr_method == 'qFFedAvg':
            server = qFFedAvg(device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, numusers, i, aggr_method, generation, individual, gamma, topk, logger, ea_alg, q)
        elif algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, numusers, i, aggr_method, generation, individual, gamma, topk, logger, ea_alg, q)
        elif algorithm == "pFedMe":
            server = pFedMe(device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, numusers, K, personal_lr, i, aggr_method, generation, individual, gamma, topk, logger, ea_alg, q)
        elif algorithm == "PerAvg":
            server = PerAvg(device, dataset, algorithm, model, batch_size, lr, beta, lamda, num_glob_iters, local_epochs, numusers, i, aggr_method, generation, individual, gamma, topk, logger, ea_alg, q)
        

        server.train()
        # server.test(per=False)

    # Average data 
    log_name_prefix = get_log_name_prefix(batch_size, lr, beta, lamda, local_epochs, numusers)
    average_data(log_name_prefix, num_glob_iters, times, path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_log(args, logger):
    logger.info("Summary of training process:")
    logger.info("Dataset: {}".format(args.dataset))
    logger.info("Model: {}".format(args.model))
    logger.info("Batch size: {}".format(args.batch_size))
    logger.info("Local learning rate: {}".format(args.lr))
    logger.info("Average moving parameter: {}".format(args.beta))
    logger.info("Regularization term: {}".format(args.lamda))
    logger.info("Number of global iterations: {}".format(args.num_global_iters))
    logger.info("Number of local epochs: {}".format(args.local_epochs))
    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Number of users: {}".format(args.numusers))
    logger.info("Computation steps: {}".format(args.K))
    logger.info("Personalized learning rate: {}".format(args.personal_lr))
    logger.info("Running time: {}".format(args.times))
    logger.info("Aggregation method: {}".format(args.aggr))
    logger.info("Aggregation generation: {}".format(args.generation))
    logger.info("Aggregation individual: {}".format(args.individual))
    logger.info("Aggregation topk: {}".format(args.topk))
    logger.info("Aggregation gamma: {}".format(args.gamma))
    logger.info("EA algorithm: {}".format(args.ea_alg))
    logger.info("q parameter for qFFedAvg: {}".format(args.q))

if __name__ == "__main__":
    setup_seed(20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="mclr", choices=["dnn", "mclr", "cnn", "resnet20"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--algorithm", type=str, default="pFedMe", choices=["pFedMe", "PerAvg", "FedAvg"]) 
    parser.add_argument("--numusers", type=int, default=5, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_lr", type=float, default=0.09, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=5, help="running time")
    parser.add_argument("--aggr", type=str, default='ParetoFed', choices=['ParetoFed', 'MtoSFed', 'Average', "AFL", "qFFedAvg"], help="Aggregation method")
    parser.add_argument("--generation", type=int, default=50, help="Aggregation generation")
    parser.add_argument("--individual", type=int, default=20, help="Aggregation individual")
    parser.add_argument("--topk", type=int, default=2, help="Aggregation topk")
    parser.add_argument("--gamma", type=float, default=0.1, help="Aggregation gamma")
    parser.add_argument("--ea-alg", type=str, default='nsga2', choices=["nsga2", "awga", "moead", "nsga3", "rvea"], help="EA algorithm")
    parser.add_argument("--q", type=float, default=1, help="q parameter for qFFedAvg")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    args = parser.parse_args()

    main(
        args=args,
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        numusers = args.numusers,
        K=args.K,
        personal_lr=args.personal_lr,
        times = args.times,
        aggr_method = args.aggr,
        generation = args.generation,
        individual = args.individual,
        gamma = args.gamma,
        topk = args.topk,
        ea_alg = args.ea_alg,
        q = args.q,
        gpu=args.gpu
        )
