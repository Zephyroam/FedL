from FLAlgorithms.aggregation.aggr_average import Aggr_Average
from FLAlgorithms.aggregation.aggr_paretofed import Aggr_ParetoFed
from FLAlgorithms.aggregation.aggr_mtosfed import Aggr_MtoSFed
from FLAlgorithms.aggregation.aggr_afl import Aggr_AFL
from FLAlgorithms.aggregation.aggr_qffedavg import Aggr_qFFedAvg


def get_aggr_method(aggregation_method, num_glob_iters, device, algorithm, model, num_users, total_users, lr, generation, individual, gamma, topk, per, logger, ea_alg, q):
    if aggregation_method == 'ParetoFed':
        return Aggr_ParetoFed(device, algorithm, model, num_users, total_users, num_glob_iters, generation, individual, topk, per, logger, ea_alg)
    elif aggregation_method == 'MtoSFed':
        return Aggr_MtoSFed(device, algorithm, model, num_users, total_users, gamma, per)
    elif aggregation_method == 'Average':
        return Aggr_Average()
    elif aggregation_method == 'AFL':
        lr_lambda = 0.1
        return Aggr_AFL(algorithm, num_users, total_users, lr_lambda, lr, per)
    elif aggregation_method == 'qFFedAvg':
        return Aggr_qFFedAvg(algorithm, num_users, total_users, lr, q, per)
    else:
        raise ValueError('Unknown aggregation method: %s' % aggregation_method)