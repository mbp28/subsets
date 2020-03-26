"""
Compute the accuracy with selected words by using L2X.
"""
import os
import os.path
from collections import defaultdict
import numpy as np
import pickle

def main():
    ks = [2, 4, 6, 8, 10]
    seeds = range(1,20)
    tasks = ['knapsack']
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]

    for k in ks:
        for task in tasks:
            for tau in taus:
                bounds = defaultdict(list)
                for seed in seeds:
                    try:
                        checkpoint = load_checkpoint(k, task, tau, seed)
                        for key, val in checkpoint.items():
                            bounds[key].append(val)
                    except:
                        pass
                print('k: {}, Task: {} Tau: {:.2f}'.format(k, task, tau))
                for key, vals in bounds.items():
                    print('\t {} {:.2f}'.format(key, np.mean(vals)))
                print('\n')

def load_checkpoint(k, task, tau, seed):
    filename = f"bounds/{k}-{task}-{tau}-{seed}.pkl"
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)

    return checkpoint

if __name__ == '__main__':
    main()
