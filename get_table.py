#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils.plot_utils import *
import torch

def get_path(dataset, model, aggr_method, algorithm, generation, individual, gamma, topk, ea_alg, q):
    path = './results/{}/{}/{}/{}/{}_{}_{}_{}_{}_{}/'.format(dataset, model, aggr_method, algorithm, generation, individual, gamma, topk, ea_alg, q)
    return path

def get_average(data, length, end):
    return np.mean(data[end-length:end])

def get_tabel(log_name_prefix, Numb_Glob_Iters, dataset, model, length, end, paths=['./results/']):
    assert isinstance(paths, list)
    times = len(paths)
    glob_acc, glob_std, train_acc, train_loss = get_avg_training_log(log_name_prefix, Numb_Glob_Iters, paths)

    for i in range(times):
        acc = get_average(glob_acc[i, :], length, end)
        std = get_average(glob_std[i, :], length, end)
        print("{}_{}_{}:".format(dataset, model, get_label(paths[i])))
        print("\terror: {:.2f}".format((1 - acc) * 100), end='')
        print("\tstd: {:.2f}".format(std * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', default=["mclr"], choices=["dnn", "mclr", "cnn", "resnet20"])
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=0.001, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--numusers", type=int, default=5, help="Number of Users per round")
    parser.add_argument("--algorithms_list", nargs='+', type=str, default=["pFedMe"], choices=["FedAvg", "pFedMe", "PerAvg"]) 
    parser.add_argument("--aggrs", nargs='+', type=str, default='ParetoFed', choices=['ParetoFed', 'MtoSFed', 'Average', "AFL", "qFFedAvg"], help="Aggregation method")
    parser.add_argument("--generations", nargs='+', type=int, default=[50], help="Aggregation generation")
    parser.add_argument("--individuals", nargs='+', type=int, default=[20], help="Aggregation individual")
    parser.add_argument("--topks", nargs='+', type=int, default=[2], help="Aggregation topk")
    parser.add_argument("--gammas", nargs='+', type=float, default=[0.1], help="Aggregation gamma")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--end", type=int, default=999, help="end")
    parser.add_argument("--ea-algs", nargs='+', type=str, default=['nsga2'], choices=["nsga2", "awga", "moead", "nsga3", "rvea"], help="EA algorithm")
    parser.add_argument("--qs", type=float, nargs='+', default=[1], help="q parameter for qFFedAvg")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithms_list))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset: {}".format(args.dataset))
    print("Model: {}".format(args.models))
    print("Batch size: {}".format(args.batch_size))
    print("Learning rate: {}".format(args.lr))
    print("Beta: {}".format(args.beta))
    print("Lambda: {}".format(args.lamda))
    print("Aggregation method: {}".format(args.aggrs))
    print("Aggregation generation: {}".format(args.generations))
    print("Aggregation individual: {}".format(args.individuals))
    print("Aggregation topk: {}".format(args.topks))
    print("Aggregation gamma: {}".format(args.gammas))
    print("length: {}".format(args.length))
    print("end: {}".format(args.end))
    print("EA algorithm: {}".format(args.ea_algs))
    print("q: {}".format(args.qs))
    print("=" * 80)

    for model in args.models:
        paths = []
        for algorithm in args.algorithms_list:
            if 'Average' in args.aggrs:
                path = get_path(args.dataset, model, 'Average', algorithm, 50, 20, 0.1, 2, 'nsga2', 1)
                paths.append(path)
            if 'MtoSFed' in args.aggrs:
                for gamma in args.gammas:
                    path = get_path(args.dataset, model, 'MtoSFed', algorithm, 50, 20, gamma, 2, 'nsga2', 1)
                    paths.append(path)
            if 'ParetoFed' in args.aggrs:
                for generation in args.generations:
                    for individual in args.individuals:
                        for topk in args.topks:
                            for ea_alg in args.ea_algs:
                                path = get_path(args.dataset, model, 'ParetoFed', algorithm, generation, individual, 0.1, topk, ea_alg, 1)
                                paths.append(path)
            if 'AFL' in args.aggrs:
                path = get_path(args.dataset, model, 'AFL', algorithm, 50, 20, 0.1, 2, 'nsga2', 1)
                paths.append(path)
            if 'qFFedAvg' in args.aggrs:
                for q in args.qs:
                    path = get_path(args.dataset, model, 'qFFedAvg', algorithm, 50, 20, 0.1, 2, 'nsga2', q)
                    paths.append(path)
        log_name_prefix = get_log_name_prefix(args.batch_size, args.lr, args.beta, args.lamda, args.local_epochs, args.numusers)
        get_tabel(log_name_prefix, args.num_global_iters, args.dataset, model, args.length, args.end, paths)

