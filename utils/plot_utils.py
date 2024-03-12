import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
plt.rcParams.update({'font.size': 14})

def get_log_name_prefix(batch_size, lr, beta, lamda, local_epochs, num_users):
    prefix = '{}_{}_{}_{}_{}_{}'.format(batch_size, lr, beta, lamda, local_epochs, num_users)
    return prefix

def get_log_name(prefix, i):
    if str(i) == 'avg':
        return prefix + "_avg"
    else:
        return prefix + '_' + str(i)

def get_label(path):
    data = path.split('/')[4:-1]
    aggr_method, algorithm = data[0], data[1]
    generation, individual, gamma, topk, ea_alg = data[2].split('_')
    res = '{}_{}'.format(aggr_method, algorithm)
    if aggr_method == 'ParetoFed':
        res = '{}_{}_{}_{}_{}'.format(res, generation, individual, topk, ea_alg)
    elif aggr_method == 'MtoSFed':
        res = '{}_{}'.format(res, gamma)
    return res

def simple_read_data(name, path='./results/'):
    hf = h5py.File(path + '{}.h5'.format(name), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_glob_std = np.array(hf.get('rs_glob_std')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc, rs_glob_std

def get_avg_training_log(log_name_prefix, num_glob_iters, paths=['./results/']):
    times = len(paths)
    train_acc = np.zeros((times, num_glob_iters))
    train_loss = np.zeros((times, num_glob_iters))
    glob_acc = np.zeros((times, num_glob_iters))
    glob_std = np.zeros((times, num_glob_iters))
    for i in range(times):
        try:
            train_acc[i, :], train_loss[i, :], glob_acc[i, :], glob_std[i, :] = np.array(simple_read_data(get_log_name(log_name_prefix, "avg"), paths[i]))[:, :num_glob_iters]
        except:
            print("{} does not exist".format(paths[i]))
    return glob_acc, glob_std, train_acc, train_loss

def get_all_training_log(log_name_prefix, num_glob_iters, times, path='./results/'):
    train_acc = np.zeros((times, num_glob_iters))
    train_loss = np.zeros((times, num_glob_iters))
    glob_acc = np.zeros((times, num_glob_iters))
    glob_std = np.zeros((times, num_glob_iters))
    for i in range(times):
        train_acc[i, :], train_loss[i, :], glob_acc[i, :], glob_std[i, :] = np.array(simple_read_data(get_log_name(log_name_prefix, i), path))[:, :num_glob_iters]
    return glob_acc, glob_std, train_acc, train_loss

def average_data(log_name_prefix, num_glob_iters, times, path='./results/'):
    glob_acc, glob_std, train_acc, train_loss = get_all_training_log(log_name_prefix, num_glob_iters, times, path)
    glob_acc_data = np.average(glob_acc, axis=0)
    glob_std_data = np.average(glob_std, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    # store average value to h5 file
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())
    
    print("std:", np.std(max_accurancy))
    print("Mean:", np.mean(max_accurancy))

    name = get_log_name(log_name_prefix, "avg")
    if len(glob_acc) != 0 & len(train_acc) & len(train_loss):
        if not os.path.exists(path):
            os.makedirs(path)
        with h5py.File(path + '{}.h5'.format(name), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_glob_std', data=glob_std_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()
    # plot_summary_one_figure(num_glob_iters, dataset, 20, [path])

def plot_summary_one_figure(log_name_prefix, num_glob_iters, dataset, model, start, paths=['./results/']):
    assert isinstance(paths, list)
    times = len(paths)
    glob_acc_, glob_std_, train_acc_, train_loss_ = get_avg_training_log(log_name_prefix, num_glob_iters, paths)

    glob_acc = average_smooth(glob_acc_, window='flat')
    glob_std = average_smooth(glob_std_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    plt.figure()
    fig = plt.figure(1)
    plt.grid(True)
    # print("max value of test accuracy",glob_acc.max())
    for i in range(times):
        try:
            assert glob_std[i, start] != 0
            plt.plot(glob_std[i, start:], label=get_label(paths[i]))
        except:
            print("data of {} is 0".format(paths[i]))
    plt.legend()
    plt.ylabel('Test std')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper() + '_' + model.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig('glob_std.pdf', bbox_inches="tight")

    plt.figure()
    for i in range(times):
        try:
            assert glob_acc[i, -1] != 0
            plt.plot(glob_acc[i, start:], label=get_label(paths[i]))
        except:
            print("data of {} is 0".format(paths[i]))
    plt.legend()
    #plt.ylim([0.6, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper() + '_' + model.upper())
    plt.savefig('glob_acc.pdf', bbox_inches="tight")

'''
def get_max_value_index(num_glob_iters, lamb, lr, algorithms_list, dataset):
    times = len(algorithms_list)
    glob_acc, glob_std, train_acc, train_loss = get_avg_training_log(num_glob_iters, lamb, lr, algorithms_list, dataset)
    for i in range(times):
        print("Algorithm: ", get_label(paths[i]), "Max testing Accuracy: ", glob_acc[i].max(), "Index: ", np.argmax(glob_acc[i]), "local update:", local_epochs[i])
'''

def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len < 3:
        return data
    for i in range(len(data)):
        x = data[i]
        s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)
