"""
Compute the accuracy with selected words by using L2X.
"""
import os
import os.path
import math
import numpy as np

from subsets.L2X.imdb_word.explain import create_original_model


def main():
    ks = [2, 4, 6, 8, 10]
    seeds = range(1,20)
    tasks = ['l2x', 'subsets', 'knapsack', 'lml']
    taus = [0.1, 0.5, 1.0, 2.0, 5.0]

    # Load original model
    model = create_original_model()
    weights_name = [
        i for i in os.listdir('./models') if i.startswith('original')][0]
    model.load_weights('./models/' + weights_name, by_name=True)
    # Load original predictions
    pred_val = np.load('data/pred_val.npy')
    for k in ks:
        for task in tasks:
            for tau in taus:
                accs = []
                for seed in seeds:
                    acc = test(model, pred_val, k, task, tau, seed)
                    accs.append(acc)
                mean = np.mean(accs)
                std = np.std(accs)
                cfd = std * 1.96 / math.sqrt(len(seeds)) # significant at 5% level, CLT justifies normal approx
                print('k: {}, Task: {} Tau: {:.2f}'.format(k, task, tau))
                print('\t Test Acc: mean {:.3f}, std: {:.3f}'.format(mean, std))
                print('\t Test Acc: {:.1f} ± {:.1f}'.format(100 * mean, 100* cfd))
            print('\n')

def test(model, pred_val, k, task, tau, seed): # model is original model
    fname = f'data/pred_val-{k}-{task}-{tau}-{seed}.npy'
    if os.path.exists(fname):
        new_pred_val = np.load(fname)
    else:
        x_val_selected = np.load(f'data/x_val-{k}-{task}-{tau}-{seed}.npy')
        new_pred_val = model.predict(x_val_selected, verbose=0, batch_size=1000)
        np.save(fname, new_pred_val)

    test_acc = np.mean(
        np.argmax(pred_val, axis=-1) == np.argmax(new_pred_val, axis=-1))
    return test_acc # is test accuracy


if __name__ == '__main__':
    main()
