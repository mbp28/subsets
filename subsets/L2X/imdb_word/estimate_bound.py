import argparse

import subsets.L2X.imdb_word.explain as _explain
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation
from keras.datasets import imdb
from keras.engine.topology import Layer
from keras import backend as K
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import Model, Sequential

from pdb import set_trace
import pickle

import numpy as np
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from subsets.sample_knapsack import gumbel_keys

def main(k, task, tau, seed):
    num_epochs = 10
    checkpoint= {}
    # Get data
    x_train, y_train, x_test, y_test, id_to_word = _explain.load_data()
    pred_train = np.load('data/pred_train.npy')
    pred_test = np.load('data/pred_val.npy')
    x_train, x_val, pred_train, pred_val = train_test_split(
        x_train, pred_train, test_size=0.1, random_state=111)
    data = [(x_train, pred_train), (x_val, pred_val), (x_test, y_test)]
    # Soft bound
    model = get_model(k, task, tau)
    model.load_weights(f'models/{k}-{task}-{tau}-{seed}.hdf5', by_name=True)
    bounds = ['train_soft', 'val_soft', 'test_soft']
    for bound, (x, y) in  zip(bounds, data):
        v = evaluate_bound(model, x, y, num_epochs)
        checkpoint[bound] = v
    # Hard bound
    model = get_model(k, 'hard', tau)
    model.load_weights(f'models/{k}-{task}-{tau}-{seed}.hdf5', by_name=True)
    bounds = ['train_hard', 'val_hard', 'test_hard']
    for bound, (x, y) in  zip(bounds, data):
        v = evaluate_bound(model, x, y, num_epochs)
        checkpoint[bound] = v
    # Save checkpoint
    with open(f'bounds/{k}-{task}-{tau}-{seed}.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate_bound(model, x_data, pred_data, num_epochs):
    bound = AverageMeter()
    not_nan = False
    for _ in range(num_epochs):
        val, _ = model.evaluate(x=x_data, y=pred_data, batch_size=1000)
        if val != np.nan:
            not_nan = True
            bound.update(val)
    if not_nan:
        return bound.avg
    else:
        return np.nan

class SampleHard(Layer):
    """
    Layer for continuous approx of subset sampling

    """
    def __init__(self, k, **kwargs):
        self.k = k
        super(SampleHard, self).__init__(**kwargs)

    def call(self, logits):
        # logits: [BATCH_SIZE, d, 1]
        logits = tf.squeeze(logits, 2)
        logits = gumbel_keys(logits) # perturb-log-probs

        # Just select top-k in one-hot mask fashion
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        sample = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        return tf.expand_dims(sample, -1)

    def compute_output_shape(self, input_shape):
        return input_shape


##
def get_model(k, task, tau):
    ## First define the model
    # P(S|X)
    with tf.variable_scope('selection_model'):
        X_ph = Input(shape=(_explain.maxlen,), dtype='int32')

        logits_T = _explain.construct_gumbel_selector(X_ph, _explain.max_features, _explain.embedding_dims, _explain.maxlen)
        if task == 'subsets':
            subset_sampler = Lambda(_explain.SampleSubset(tau, k).sample) #instead of call, just sample (can use keras predict feature)
            T = subset_sampler(logits_T)
        elif task == 'l2x':
            subset_sampler = Lambda(_explain.SampleConcrete(tau, k).sample)
            T = subset_sampler(logits_T)
        elif task == 'knapsack':
            subset_sampler = Lambda(_explain.SampleKnapsack(tau, k).sample)
            T = subset_sampler(logits_T)
        elif task == 'lml':
            subset_sampler = Lambda(_explain.SampleLML(tau, k).sample)
            T = subset_sampler(logits_T)
        elif task == 'hard':
            subset_sampler = SampleHard(k)
            T = subset_sampler(logits_T)
        else:
            raise ValueError

    # q(X_S)
    Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k),
        output_shape=lambda x: [x[0],x[2]])

    with tf.variable_scope('prediction_model'):
        emb2 = Embedding(_explain.max_features, _explain.embedding_dims,
            input_length=_explain.maxlen)(X_ph)

        net = Mean(_explain.Multiply()([emb2, T]))
        net = Dense(_explain.hidden_dims)(net)
        net = Activation('relu')(net)
        preds = Dense(2, activation='softmax',
            name = 'new_dense')(net)

    model = Model(inputs=X_ph, outputs=preds)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',#optimizer,
                  metrics=['acc'])
    return model

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str,
        choices = ['original','l2x', 'subsets', 'knapsack', 'lml'], default = 'original')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    print(args)

    main(args.k, args.task, args.tau, args.seed)


#
#
# def L2X(train = True, task='l2x', tau=0.01):
#     """
#     Generate scores on features on validation by L2X.
#
#     Train the L2X model with variational approaches
#     if train = True.
#
#     """
#     k = args.k
#     print('Loading dataset...')
#     x_train, y_train, x_val, y_val, id_to_word = load_data()
#     pred_train = np.load('data/pred_train.npy')
#     pred_val = np.load('data/pred_val.npy')
#     print('Creating model...')
#
#     # P(S|X)
#     with tf.variable_scope('selection_model'):
#         X_ph = Input(shape=(maxlen,), dtype='int32')
#
#         logits_T = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen)
#         if task == 'subsets':
#             subset_sampler = SampleSubset(tau, k)
#             T = subset_sampler(logits_T)
#         elif task == 'l2x':
#             subset_sampler = SampleConcrete(tau, k)
#             T = subset_sampler(logits_T)
#         elif task == 'knapsack':
#             subset_sampler = SampleKnapsack(tau, k)
#             T = subset_sampler(logits_T)
#         elif task == 'lml':
#             subset_sampler = SampleLML(tau, k)
#             T = subset_sampler(logits_T)
#         else:
#             raise ValueError
#
#     # q(X_S)
#     Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k),
#         output_shape=lambda x: [x[0],x[2]])
#
#     with tf.variable_scope('prediction_model'):
#         emb2 = Embedding(max_features, embedding_dims,
#             input_length=maxlen)(X_ph)
#
#         net = Mean(Multiply()([emb2, T]))
#         net = Dense(hidden_dims)(net)
#         net = Activation('relu')(net)
#         preds = Dense(2, activation='softmax',
#             name = 'new_dense')(net)
#
#     model = Model(inputs=X_ph, outputs=preds)
#
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',#optimizer,
#                   metrics=['acc'])
#     train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
#     val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
#     print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))
#
#     if train:
#         # split x_train and pred_train into training and validation
#         x_train, x_train_val, pred_train, pred_train_val = train_test_split(
#             x_train, pred_train, test_size=0.1, random_state=111)
#
#         filepath=f"models/{k}-{task}-{tau}-{seed}.hdf5"
#         checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
#             verbose=1, save_best_only=True, mode='auto')
#         callbacks_list = [checkpoint]
#         st = time.time()
#         model.fit(x_train, pred_train,
#             validation_data=(x_train_val, pred_train_val),
#             callbacks = callbacks_list,
#             epochs=50, batch_size=batch_size)
#         duration = time.time() - st
#         print('Training time is {}'.format(duration))
#
#     model.load_weights(f'models/{k}-{task}-{tau}-{seed}.hdf5', by_name=True)
#
#     pred_model = Model(X_ph, logits_T)
#     pred_model.compile(loss='categorical_crossentropy',
#         optimizer='adam', metrics=['acc'])
#
#     st = time.time()
#     scores = pred_model.predict(x_val,
#         verbose = 1, batch_size = batch_size)[:,:,0]
#     scores = np.reshape(scores, [scores.shape[0], maxlen])
#
#     return scores, x_val
#
#
#
# def var_bound():
#     pass
