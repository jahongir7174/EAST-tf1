import copy
import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from src.nets.base_nets import conv_blocks as ops, mobilenet as lib

op = lib.op
expand_input = ops.expand_input_by_factor

squeeze_excite = functools.partial(ops.squeeze_excite, squeeze_factor=4,
                                   inner_activation_fn=tf.nn.relu,
                                   gating_fn=lambda x: tf.nn.relu6(x + 3) * 0.16667)

_se4 = lambda expansion_tensor, input_tensor: squeeze_excite(expansion_tensor)


def hard_swish(x):
    with tf.name_scope('hard_swish'):
        return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def reduce_to_1x1(input_tensor, default_size=7, **kwargs):
    h, w = input_tensor.shape.as_list()[1:3]
    if h is not None and w == h:
        k = [h, h]
    else:
        k = [default_size, default_size]
    return slim.avg_pool2d(input_tensor, kernel_size=k, **kwargs)


def mbv3_op(ef, n, k, s=1, act=tf.nn.relu, se=None):
    return op(ops.expanded_conv, expansion_size=expand_input(ef),
              kernel_size=(k, k), stride=s, num_outputs=n,
              inner_activation_fn=act,
              expansion_transform=se)


mbv3_op_se = functools.partial(mbv3_op, se=_se4)

DEFAULTS = {(ops.expanded_conv,):
                dict(normalizer_fn=slim.batch_norm, residual=True),
            (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {'normalizer_fn': slim.batch_norm,
                                                                         'activation_fn': tf.nn.relu,
                                                                         },
            (slim.batch_norm,): {'center': True,
                                 'scale': True
                                 }, }

# Compatible checkpoint: http://mldash/5511169891790690458#scalars
V3_LARGE = dict(defaults=dict(DEFAULTS),
                spec=([op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3), activation_fn=hard_swish),
                       mbv3_op(ef=1, n=16, k=3),
                       mbv3_op(ef=4, n=24, k=3, s=2),
                       mbv3_op(ef=3, n=24, k=3, s=1),
                       mbv3_op_se(ef=3, n=40, k=5, s=2),
                       mbv3_op_se(ef=3, n=40, k=5, s=1),
                       mbv3_op_se(ef=3, n=40, k=5, s=1),
                       mbv3_op(ef=6, n=80, k=3, s=2, act=hard_swish),
                       mbv3_op(ef=2.5, n=80, k=3, s=1, act=hard_swish),
                       mbv3_op(ef=184 / 80., n=80, k=3, s=1, act=hard_swish),
                       mbv3_op(ef=184 / 80., n=80, k=3, s=1, act=hard_swish),
                       mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
                       mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
                       mbv3_op_se(ef=6, n=160, k=5, s=2, act=hard_swish),
                       mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
                       mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish)]))


@slim.add_arg_scope
def mobile_net(input_tensor,
               depth_multiplier=1.0,
               scope='MobilenetV3',
               conv_defs=None,
               finegrain_classification_mode=False,
               **kwargs):
    if conv_defs is None:
        conv_defs = V3_LARGE
    if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic multiplier parameter use "depth_multiplier" instead.')
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        conv_defs['spec'][-1] = conv_defs['spec'][-1]._replace(multiplier_func=lambda params, multiplier: params)
    depth_args = {}
    with slim.arg_scope((lib.depth_multiplier,), **depth_args):
        return lib.mobilenet(input_tensor,
                             conv_defs=conv_defs,
                             scope=scope,
                             multiplier=depth_multiplier,
                             **kwargs)


arg_scope = lib.arg_scope


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
    return mobile_net(input_tensor, depth_multiplier=depth_multiplier, base_only=True, **kwargs)


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


large = wrapped_partial(mobile_net, conv_defs=V3_LARGE)


def _reduce_consecutive_layers(conv_defs, start_id, end_id, multiplier=0.5):
    defs = copy.deepcopy(conv_defs)
    for d in defs['spec'][start_id:end_id + 1]:
        d.params.update({'num_outputs': np.int(np.round(d.params['num_outputs'] * multiplier))})
    return defs


V3_LARGE_DETECTION = _reduce_consecutive_layers(V3_LARGE, 13, 16)

__all__ = ['arg_scope', 'mobile_net', 'V3_LARGE', 'large', 'V3_LARGE_DETECTION']
