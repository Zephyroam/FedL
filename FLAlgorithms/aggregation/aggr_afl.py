import torch
import os
import numpy as np
'''
def project(y):
    u = sorted(y, reverse=True)
    x = []
    rho = 0
    for i in range(len(y)):
        if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i]))) > 0:
            rho = i + 1
    lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
    for i in range(len(y)):
        x.append(max(y[i]+lambda_, 0))
    return torch.tensor(x)
'''


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and (v >= 0).all():
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v)[0],dims=(0,))
    cssv = torch.cumsum(u,dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho = 0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w

class Aggr_AFL(object):
    def __init__(self, algorithm, num_users, total_users, lr_lambda, lr, per):
        self.one_step = algorithm == 'PerAvg'
        self.num_users = num_users
        self.total_users = total_users
        self.lr_lambda = lr_lambda
        self.lr = lr
        self.per = per
        self.latest_lambdas = torch.ones(self.total_users) / self.total_users
    
    def aggregate_parameters(self, model, model_test, selected_users, round_num):
        total_weight = 0
        lr_gradients = [0] * len(selected_users[0].lr_gradients)

        for user in selected_users:
            id = int(user.id[2:])
            for i, gradient in enumerate(user.lr_gradients):
                lr_gradients[i] += gradient * self.latest_lambdas[id]
            total_weight += self.latest_lambdas[id]
            self.latest_lambdas[id] += self.lr_lambda * user.losses
            # print(user.losses)

        lr_gradients = [g / total_weight for g in lr_gradients]

        for model_param, grad in zip(model.parameters(), lr_gradients):
            model_param.data -= grad

        self.latest_lambdas = euclidean_proj_simplex(self.latest_lambdas)
        # Avoid zero probability
        lambda_zeros = self.latest_lambdas < 1e-3
        if lambda_zeros.sum() > 0:
            self.latest_lambdas[lambda_zeros] = 1e-3
            self.latest_lambdas /= self.latest_lambdas.sum()

        for model_test_param, model_param in zip(model_test.parameters(), model.parameters()):
            model_test_param.data = (model_test_param.data * round_num + model_param.data.clone()) / (round_num + 1)

        return [model], model_test