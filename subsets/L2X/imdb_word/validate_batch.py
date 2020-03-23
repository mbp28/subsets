"""
Compute the accuracy with selected words by using L2X.
"""
import os
import os.path

import numpy as np

from subsets.L2X.imdb_word.explain import create_original_model


def main():
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

    for task in tasks:
        for tau in taus:
            accs = []
            for seed in seeds:
                acc = test(model, pred_val, task, tau, seed)
                accs.append(acc)
            mean = np.mean(accs)
            std = np.std(accs)
            print('Task {}'.format(task))
            print('\t Tau {:.2f}'.format(tau))
            print('\t \t Test Acc {:.3f} ± {:.3f}'.format(mean, std))

def test(model, pred_val, task, tau, seed): # model is original model
    fname = f'data/pred_val-{task}-{tau}-{seed}.npy'
    if os.path.exists(fname):
        new_pred_val = np.load(fname)
    else:
        x_val_selected = np.load(f'data/x_val-{task}-{tau}-{seed}.npy')
        new_pred_val = model.predict(x_val_selected, verbose=1, batch_size=1000)
        np.save(fname, new_pred_val)

    test_acc = np.mean(
        np.argmax(pred_val, axis=-1) == np.argmax(new_pred_val, axis=-1))
    return test_acc # is test accuracy


if __name__ == '__main__':
    main()
