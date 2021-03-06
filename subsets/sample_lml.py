import tensorflow as tf
from pdb import set_trace
import numpy as np

EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny
def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random_uniform(
        tf.shape(w),
        minval=EPSILON,
        maxval=1.0)
    z = -tf.log(-tf.log(uniform))
    w = w + z
    return w

def sample_lml(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    lml_func = get_lml_func(k, eps=1e-6, n_iter=100, branch=None, verbose=0)
    w = gumbel_keys(w) / t
    y = lml_func(w)
    return y

def get_lml_func(k, eps=1e-4, n_iter=100, branch=None, verbose=0):
    r'''we define a closure to return the tensorflow function which is
    applied on x'''
    @tf.custom_gradient
    def lml_func(x):
        # now main body forward pass
        y, nu = lml_func_forward(x, k, eps, n_iter, branch, verbose)
        def grad(dy):
            return lml_func_backward(dy, x, y, nu)
        return y, grad
    return lml_func


def lml_func_forward(x, k, eps, n_iter, branch, verbose):
    if branch is None:
        if tf.test.is_gpu_available():
            branch = 100
        else:
            branch = 10

    n_batch = tf.shape(x)[0]
    nx = tf.shape(x)[1]

    x_sorted = tf.sort(x, axis=1, direction='DESCENDING')

    # The sigmoid saturates the interval [-7, 7]
    nu_lower = -x_sorted[:, k-1] - 7.
    nu_upper = -x_sorted[:, k]   + 7.

    ls = tf.cast(tf.linspace(0.0, 1.0, branch), x.dtype)

    def _cond(lower, upper):
        r = upper - lower
        mask = r > eps
        return tf.constant(tf.reduce_sum(tf.cast(mask, tf.int32)) != 0)

    def _loop(lower, upper):
        # dependencies
        r = upper - lower # vector of size batchsize
        mask = r > eps
        n_update = tf.reduce_sum(tf.cast(mask, tf.int32))
        nus = tf.expand_dims(r[mask], 1)*ls + tf.expand_dims(lower[mask], 1)
        nus = tf.reshape(nus, [n_update, branch])
        _xs = tf.reshape(x[mask], [n_update, 1, nx]) + tf.expand_dims(nus, 2)
        fs = tf.reduce_sum(tf.sigmoid(_xs), axis=2) - k
        i_lower = (tf.reduce_sum(tf.cast(fs < 0, tf.int32), axis=1) - 1)
        J = i_lower < 0
        # First if-condition
        i_lower = tf.maximum(i_lower, 0) # we could have -1 for i_lower if with numerical instability every fs >= 0
        i_upper = tf.minimum(i_lower + 1, branch-1) # we could have every fs < 0
        i_lower = tf.stack([tf.range(n_update), i_lower], axis=1)
        i_upper = tf.stack([tf.range(n_update), i_upper], axis=1)
        lower_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), tf.gather_nd(nus, i_lower), (n_batch,))
        upper_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), tf.gather_nd(nus, i_upper), (n_batch,))
        #set_trace()
        new_lower = tf.where(mask, lower_gathered, lower)
        new_upper = tf.where(mask, upper_gathered, upper)

        # Second if-condition
        #new_lower = tf.where(J, new_lower - 7, new_lower)

        return new_lower, new_upper

    nu_lower, nu_upper = tf.while_loop(cond=_cond,
                                        body=_loop,
                                        loop_vars=(nu_lower, nu_upper),
                                        maximum_iterations=5) # for some reason cannot set higher (e.g. 100), need to explore

    nu = (nu_lower + nu_upper) / 2
    return tf.sigmoid(x + tf.expand_dims(nu, 1)), nu

def lml_func_backward(dy, x, y, nu):
    r'''this is not correct for the case where k > nx'''
    n_batch = tf.shape(x)[0]
    nx = tf.shape(x)[1]

    Hinv = 1./ (1./y + 1./(1.-y))
    dnu = bdot(Hinv, dy) / tf.reduce_sum(Hinv, axis=1)
    dx = -Hinv * (tf.expand_dims(dnu, 1) - dy)
    return dx

def bdot(x, y):
    return tf.squeeze(tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))) # shape = (batchsize,)


def _loop(lower, upper):
    # dependencies
    print('loop')
    r = upper - lower # vector of size batchsize
    mask = r > eps
    n_update = tf.reduce_sum(tf.cast(mask, tf.int32))
    nus = tf.expand_dims(r[mask], 1)*ls + tf.expand_dims(lower[mask], 1)
    nus = tf.reshape(nus, [n_update, branch])
    _xs = tf.reshape(x[mask], [n_update, 1, nx]) + tf.expand_dims(nus, 2)
    fs = tf.reduce_sum(tf.sigmoid(_xs), axis=2) - k
    i_lower = (tf.reduce_sum(tf.cast(fs < 0, tf.int32), axis=1) - 1)
    J = i_lower < 0
    # First if-condition
    i_lower = tf.maximum(i_lower, 0) # we could have -1 for i_lower if with numerical instability every fs >= 0
    i_upper = tf.minimum(i_lower + 1, branch-1) # we could have every fs < 0
    i_lower = tf.stack([tf.range(n_update), i_lower], axis=1)
    i_upper = tf.stack([tf.range(n_update), i_upper], axis=1)
    lower_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), tf.gather_nd(nus, i_lower), (n_batch,))
    upper_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), tf.gather_nd(nus, i_upper), (n_batch,))
    #set_trace()
    new_lower = tf.where(mask, lower_gathered, lower)
    new_upper = tf.where(mask, upper_gathered, upper)

    # Second if-condition
    #new_lower = tf.where(J, new_lower - 7, new_lower)

    return new_lower, new_upper

if __name__ == '__main__':
    import numpy as np
    def reproduce_error_loop(seed):
        def _loop(lower, upper):
            # dependencies
            r = upper - lower # vector of size batchsize
            mask = r > eps
            n_update = tf.reduce_sum(tf.cast(mask, tf.int32))
            nus = tf.expand_dims(r[mask], 1)*ls + tf.expand_dims(lower[mask], 1)
            nus = tf.reshape(nus, [n_update, branch])
            _xs = tf.reshape(x[mask], [n_update, 1, nx]) + tf.expand_dims(nus, 2)
            fs = tf.reduce_sum(tf.sigmoid(_xs), axis=2) - k
            i_lower = (tf.reduce_sum(tf.cast(fs < 0, tf.int32), axis=1) - 1)
            J = i_lower < 0
            # First if-condition
            i_lower = tf.maximum(i_lower, 0) # we could have -1 for i_lower if with numerical instability every fs >= 0
            i_upper = tf.minimum(i_lower + 1, branch-1) # we could have every fs < 0
            i_lower = tf.stack([tf.range(n_update), i_lower], axis=1)
            i_upper = tf.stack([tf.range(n_update), i_upper], axis=1)
            lower_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), tf.gather_nd(nus, i_lower), (n_batch,))
            upper_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), tf.gather_nd(nus, i_upper), (n_batch,))
            #set_trace()
            new_lower = tf.where(mask, lower_gathered, lower)
            new_upper = tf.where(mask, upper_gathered, upper)

            # Second if-condition
            #try:
            #    new_lower = tf.where(J, new_lower - 7, new_lower)
            #except:
            #    set_trace()

            return new_lower, new_upper
        # Set up
        np.random.seed(seed)
        batchsize = 125
        m = 400
        n = 10
        x = 100 * np.random.randn(batchsize, m)
        t = tf.constant(x, dtype=tf.float32)
        # Set-up function
        x = t
        k = n
        eps=1e-6
        n_iter=100
        branch=None
        verbose=0
        # function
        if branch is None:
            if tf.test.is_gpu_available():
                branch = 100
            else:
                branch = 10

        n_batch = tf.shape(x)[0]
        nx = tf.shape(x)[1]

        x_sorted = tf.sort(x, axis=1, direction='DESCENDING')

        # The sigmoid saturates the interval [-7, 7]
        nu_lower = -x_sorted[:, k-1] - 7.
        nu_upper = -x_sorted[:, k]   + 7.

        ls = tf.cast(tf.linspace(0.0, 1.0, branch), x.dtype)

        out1 = _loop(nu_lower, nu_upper)
        out2 = _loop(*out1)
        out3 = _loop(*out2)
        out4 = _loop(*out3)
        out5 = _loop(*out4)
        out6 = _loop(*out5)
        out7 = _loop(*out6)
        out8 = _loop(*out7)

        return out1, out2, out3, out4, out5, out6, out7, out8

    for seed in range(2, 1000):
        print('seed', seed)
        outs  = reproduce_error_loop(seed)
        with tf.Session() as sess:
            for i, out in enumerate(outs):
                print('iter', i)
                try:
                    output = sess.run(out)
                    print(output)
                except:
                    set_trace()
            print(seed)
    print('Done!')
    #t = tf.constant([[2.3, 2.4, 1.9, 2.1, 2.2],[3.4, 5.2, 3.2, 3.1, 4.1]])
    #out = sample_lml(t, n, 0.5)
    # dy = np.random.randn(2, 10)
    # dy = tf.constant(dy)
    # y = np.random.uniform(0+1e-8, 1-1e-8, size=(2, 10))
    # y = tf.constant(y)
    # x = np.random.randn(2, 10)
    # x = tf.constant(x)
    # nu = np.random.randn(2, 10)
    # nu = tf.constant(nu)
    # out = lml_func_backward(dy, x, y, nu)
    # seed = 0
    # np.random.seed(seed)
    # batchsize = 125
    # m = 400
    # n = k = 10
    # x = 100 * np.random.randn(batchsize, m)
    # x = tf.constant(x, dtype=tf.float32)
    # lower = tf.constant(np.load('/Users/mpaulus/git/subsets/arr_lower_bad.npy'))
    # upper = tf.constant(np.load('/Users/mpaulus/git/subsets/arr_upper_bad.npy'))
    # eps=1e-6
    # n_iter=100
    # branch=None
    # verbose=0
    # # function
    # if branch is None:
    #     if tf.test.is_gpu_available():
    #         branch = 100
    #     else:
    #         branch = 10
    #
    # n_batch = tf.shape(x)[0]
    # nx = tf.shape(x)[1]
    #
    # x_sorted = tf.sort(x, axis=1, direction='DESCENDING')
    # ls = tf.cast(tf.linspace(0.0, 1.0, branch), x.dtype)
    #
    # # dependencies
    # print('loop')
    # r = upper - lower # vector of size batchsize
    # mask = r > eps
    # n_update = tf.reduce_sum(tf.cast(mask, tf.int32))
    # nus = tf.expand_dims(r[mask], 1)*ls + tf.expand_dims(lower[mask], 1)
    # nus = tf.reshape(nus, [n_update, branch])
    # _xs = tf.reshape(x[mask], [n_update, 1, nx]) + tf.expand_dims(nus, 2)
    # fs = tf.reduce_sum(tf.sigmoid(_xs), axis=2) - k
    # i_lower = (tf.reduce_sum(tf.cast(fs < 0, tf.int32), axis=1) - 1)
    # J = i_lower < 0
    # # First if-condition
    # i_lower = tf.maximum(i_lower, 0)
    # i_upper = i_lower + 1
    # i_lower = tf.stack([tf.range(n_update), i_lower], axis=1)
    # i_upper = tf.stack([tf.range(n_update), i_upper], axis=1)
    # lower_taken = tf.gather_nd(nus, i_lower)
    # upper_taken = tf.gather_nd(nus, i_upper)
    # lower_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), lower_taken, (n_batch,))
    # upper_gathered = tf.scatter_nd(tf.cast(tf.where(mask), dtype=tf.int32), upper_taken, (n_batch,))
    # #set_trace()
    # new_lower = tf.where(mask, lower_gathered, lower)
    # new_upper = tf.where(mask, upper_gathered, upper)
    #
    # # Second if-condition
    # new_lower = tf.where(J, new_lower - 7, new_lower)
    #
    # sess = tf.Session()
    # #output = sess.run(new_lower)
    # output = sess.run(new_lower)
    # print(output)
    # set_trace()
