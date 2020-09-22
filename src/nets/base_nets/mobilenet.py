import collections
import contextlib
import copy
import os

import tensorflow as tf
from tensorflow.contrib import slim


@slim.add_arg_scope
def apply_activation(x, name=None, activation_fn=None):
    return activation_fn(x, name=name) if activation_fn else x


def _fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


@contextlib.contextmanager
def _set_arg_scope_defaults(defaults):
    if hasattr(defaults, 'items'):
        items = list(defaults.items())
    else:
        items = defaults
    if not items:
        yield
    else:
        func, default_arg = items[0]
        with slim.arg_scope(func, **default_arg):
            with _set_arg_scope_defaults(items[1:]):
                yield


@slim.add_arg_scope
def depth_multiplier(output_params, multiplier, divisible_by=8, min_depth=8):
    if 'num_outputs' not in output_params:
        return
    d = output_params['num_outputs']
    output_params['num_outputs'] = _make_divisible(d * multiplier, divisible_by, min_depth)


_Op = collections.namedtuple('Op', ['op', 'params', 'multiplier_func'])


def op(opfunc, multiplier_func=depth_multiplier, **params):
    multiplier = params.pop('multiplier_transform', multiplier_func)
    return _Op(opfunc, params=params, multiplier_func=multiplier)


class NoOpScope(object):
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def safe_arg_scope(funcs, **kwargs):
    filtered_args = {name: value for name, value in kwargs.items() if value is not None}
    if filtered_args:
        return slim.arg_scope(funcs, **filtered_args)
    else:
        return NoOpScope()


@slim.add_arg_scope
def mobilenet_base(inputs, conv_defs, multiplier=1.0, final_endpoint=None, output_stride=None,
                   use_explicit_padding=False, scope=None, is_training=False):
    if multiplier <= 0:
        raise ValueError('multiplier is not greater than zero.')

    conv_defs_defaults = conv_defs.get('defaults', {})
    conv_defs_overrides = conv_defs.get('overrides', {})
    if use_explicit_padding:
        conv_defs_overrides = copy.deepcopy(conv_defs_overrides)
        conv_defs_overrides[(slim.conv2d, slim.separable_conv2d)] = {'padding': 'VALID'}

    if output_stride is not None:
        if output_stride == 0 or (output_stride > 1 and output_stride % 2):
            raise ValueError('Output stride must be None, 1 or a multiple of 2.')

    with _scope_all(scope, default_scope='Mobilenet'), safe_arg_scope([slim.batch_norm], is_training=is_training), \
         _set_arg_scope_defaults(conv_defs_defaults), _set_arg_scope_defaults(conv_defs_overrides):
        current_stride = 1
        rate = 1
        net = inputs
        end_points = {}
        scopes = {}
        for i, opdef in enumerate(conv_defs['spec']):
            params = dict(opdef.params)
            opdef.multiplier_func(params, multiplier)
            stride = params.get('stride', 1)
            if output_stride is not None and current_stride == output_stride:
                layer_stride = 1
                layer_rate = rate
                rate *= stride
            else:
                layer_stride = stride
                layer_rate = 1
                current_stride *= stride
            params['stride'] = layer_stride
            if layer_rate > 1:
                if tuple(params.get('kernel_size', [])) != (1, 1):
                    params['rate'] = layer_rate
            if use_explicit_padding:
                if 'kernel_size' in params:
                    net = _fixed_padding(net, params['kernel_size'], layer_rate)
                else:
                    params['use_explicit_padding'] = True
            end_point = 'layer_%d' % (i + 1)
            try:
                net = opdef.op(net, **params)
            except Exception:
                print('Failed to create op %i: %r params: %r' % (i, opdef, params))
                raise
            end_points[end_point] = net
            scope = os.path.dirname(net.name)
            scopes[scope] = end_point
            if final_endpoint is not None and end_point == final_endpoint:
                break
        for t in net.graph.get_operations():
            scope = os.path.dirname(t.name)
            bn = os.path.basename(t.name)
            if scope in scopes and t.name.endswith('output'):
                end_points[scopes[scope] + '/' + bn] = t.outputs[0]
        return net, end_points


@contextlib.contextmanager
def _scope_all(scope, default_scope=None):
    with tf.variable_scope(scope, default_name=default_scope) as s, \
            tf.name_scope(s.original_name_scope):
        yield s


@slim.add_arg_scope
def mobilenet(inputs, reuse=None, scope='Mobilenet', **mobilenet_args):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

    with tf.variable_scope(scope, 'Mobilenet', reuse=reuse) as scope:
        inputs = tf.identity(inputs, 'input')
        net, end_points = mobilenet_base(inputs, scope=scope, **mobilenet_args)
        end_points['pool5'] = net
        end_points['pool4'] = end_points['layer_11/output']
        end_points['pool3'] = end_points['layer_5/output']
        end_points['pool2'] = end_points['layer_3/output']
        f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]
    return net, end_points, f


def arg_scope(is_training=True, weight_decay=0.00004, stddev=0.09, dropout_keep_prob=0.8, bn_decay=0.997):
    batch_norm_params = {'decay': bn_decay, 'is_training': is_training}
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.separable_conv2d],
            weights_initializer=weight_intitializer,
            normalizer_fn=slim.batch_norm), \
         slim.arg_scope([mobilenet_base, mobilenet], is_training=is_training), \
         safe_arg_scope([slim.batch_norm], **batch_norm_params), \
         safe_arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob), \
         slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)), \
         slim.arg_scope([slim.separable_conv2d], weights_regularizer=None) as s:
        return s
