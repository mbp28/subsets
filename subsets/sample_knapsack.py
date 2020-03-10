import tensorflow as tf
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
import numpy as np
import scipy
import math

EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny

def sample_knapsack(w, k, t=0.1):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    shape = tf.shape(w)
    w = gumbel_keys(w) / t
    z = w - math.log(k)
    threshold = tf.fill(shape, 1 / k)
    alpha = csoftmax(z, threshold)
    out = k * alpha
    return out

def gumbel_keys(w):
    # sample some gumbels
    uniform = tf.random_uniform(
        tf.shape(w),
        minval=EPSILON,
        maxval=1.0)
    z = tf.log(-tf.log(uniform))
    w = w + z
    return w

 # Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def csoftmax_for_slice(input):
    """ It is a implementation of the constrained softmax (csoftmax) for slice.
        Based on the paper:
        https://andre-martins.github.io/docs/emnlp2017_final.pdf "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers" (page 4)
    Args:
        input: A list of [input tensor, cumulative attention].
    Returns:
        output: A list of [csoftmax results, masks]
    """

    [ten, u] = input

    shape_t = ten.shape
    shape_u = u.shape

    ten -= tf.reduce_mean(ten)
    q = tf.exp(ten)
    active = tf.ones_like(u, dtype=tf.int32)
    mass = tf.constant(0, dtype=tf.float32)
    found = tf.constant(True, dtype=tf.bool)

    def loop(q_, mask, mass_, found_):
        q_list = tf.dynamic_partition(q_, mask, 2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(q_)[0]), mask, 2)  # 0 element it False,
        #  1 element if true

        p = q_list[1] * (1.0 - mass_) / tf.reduce_sum(q_list[1])
        p_new = tf.dynamic_stitch(condition_indices, [q_list[0], p])

        # condition verification and mask modification
        less_mask = tf.cast(tf.less(u, p_new), tf.int32)  # 0 when u is bigger than p, 1 when u is less than p
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(p_new)[0]), less_mask,
                                                 2)  # 0 when u is bigger than p, 1 when u is less than p

        split_p_new = tf.dynamic_partition(p_new, less_mask, 2)
        split_u = tf.dynamic_partition(u, less_mask, 2)

        alpha = tf.dynamic_stitch(condition_indices, [split_p_new[0], split_u[1]])
        mass_ += tf.reduce_sum(split_u[1])

        mask = mask * (tf.ones_like(less_mask) - less_mask)

        found_ = tf.cond(tf.equal(tf.reduce_sum(less_mask), 0),
                         lambda: False,
                         lambda: True)

        alpha = tf.reshape(alpha, q_.shape)

        return alpha, mask, mass_, found_

    (csoft, mask_, _, _) = tf.while_loop(cond=lambda _0, _1, _2, f: f,
                                         body=loop,
                                         loop_vars=(q, active, mass, found))

    return [csoft, mask_]

def csoftmax(tensor, inv_cumulative_att):
    """ It is a implementation of the constrained softmax (csoftmax).
        Based on the paper:
        https://andre-martins.github.io/docs/emnlp2017_final.pdf "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    Args:
        tensor: A tensorflow tensor is score. This tensor have dimensionality [None, n_tokens]
        inv_cumulative_att: A inverse cumulative attention tensor with dimensionality [None, n_tokens]
    Returns:
        cs: Tensor at the output with dimensionality [None, n_tokens]
    """
    shape_ten = tensor.shape
    shape_cum = inv_cumulative_att.shape

    merge_tensor = [tensor, inv_cumulative_att]
    cs, _ = tf.map_fn(csoftmax_for_slice, merge_tensor, dtype=[tf.float32, tf.float32])  # [bs, L]
    return cs
