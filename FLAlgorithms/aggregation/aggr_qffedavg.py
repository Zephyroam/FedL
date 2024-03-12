import torch
import os
import numpy as np

def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = grad_list[0].cpu() # shape now: (784, 26)

    for i in range(1, len(grad_list)):
        client_grads = np.append(client_grads, grad_list[i].cpu()) # output a flattened array

    return np.sum(np.square(client_grads))

class Aggr_qFFedAvg(object):
    def __init__(self, algorithm, num_users, total_users, lr, q, per):
        self.one_step = algorithm == 'PerAvg'
        self.num_users = num_users
        self.total_users = total_users
        self.lr = lr
        self.q = q
        self.per = per

    def aggregate_parameters(self, model, selected_users):

        Deltas = []
        hs = []
        for user in selected_users:
            deltas = []
            id = int(user.id[2:])
            for i, gradient in enumerate(user.lr_gradients):
                deltas.append(gradient / self.lr * np.float_power(user.losses + 1e-10, self.q))
            Deltas.append(deltas)
            # estimation of the local Lipchitz constant
            hs.append(self.q * np.float_power(user.losses + 1e-10, (self.q - 1)) * norm_grad(user.lr_gradients) + (1.0 / self.lr) * np.float_power(user.losses + 1e-10, self.q))

        demominator = np.sum(np.asarray(hs))
        scaled_deltas = []
        for client_delta in Deltas:
            scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

        updates = []
        for i in range(len(Deltas[0])):
            tmp = scaled_deltas[0][i]
            for j in range(1, len(Deltas)):
                tmp += scaled_deltas[j][i]
            updates.append(tmp)

        for param, update in zip(model.parameters(), updates):
            param.data.add_(-update)

        return [model]
