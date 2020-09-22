import contextlib
import functools

import tensorflow as tf
from tensorflow.contrib import slim


def _fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _split_divisible(num, num_ways, divisible_by=8):
    assert num % divisible_by == 0
    assert num / num_ways >= divisible_by
    base = num // num_ways // divisible_by * divisible_by
    result = []
    accumulated = 0
    for i in range(num_ways):
        r = base
        while accumulated + r < num * (i + 1) / num_ways:
            r += divisible_by
        result.append(r)
        accumulated += r
    assert accumulated == num
    return result


@contextlib.contextmanager
def _v1_compatible_scope_naming(scope):
    if scope is None:
        with tf.variable_scope(None, default_name='separable') as s, \
                tf.name_scope(s.original_name_scope):
            yield ''
    else:
        scope += '_'
        yield scope


@slim.add_arg_scope
def split_separable_conv2d(input_tensor,
                           num_outputs,
                           scope=None,
                           normalizer_fn=None,
                           stride=1,
                           rate=1,
                           endpoints=None,
                           use_explicit_padding=False):
    with _v1_compatible_scope_naming(scope) as scope:
        dw_scope = scope + 'depthwise'
        endpoints = endpoints if endpoints is not None else {}
        kernel_size = [3, 3]
        padding = 'SAME'
        if use_explicit_padding:
            padding = 'VALID'
            input_tensor = _fixed_padding(input_tensor, kernel_size, rate)
        net = slim.separable_conv2d(input_tensor,
                                    None,
                                    kernel_size,
                                    depth_multiplier=1,
                                    stride=stride,
                                    rate=rate,
                                    normalizer_fn=normalizer_fn,
                                    padding=padding,
                                    scope=dw_scope)

        endpoints[dw_scope] = net

        pw_scope = scope + 'pointwise'
        net = slim.conv2d(net,
                          num_outputs, [1, 1],
                          stride=1,
                          normalizer_fn=normalizer_fn,
                          scope=pw_scope)
        endpoints[pw_scope] = net
    return net


def expand_input_by_factor(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def split_conv(input_tensor,
               num_outputs,
               num_ways,
               scope,
               divisible_by=8,
               **kwargs):
    b = input_tensor.get_shape().as_list()[3]

    if num_ways == 1 or min(b // num_ways, num_outputs // num_ways) < divisible_by:
        return slim.conv2d(input_tensor, num_outputs, [1, 1], scope=scope, **kwargs)

    outs = []
    input_splits = _split_divisible(b, num_ways, divisible_by=divisible_by)
    output_splits = _split_divisible(num_outputs, num_ways, divisible_by=divisible_by)
    inputs = tf.split(input_tensor, input_splits, axis=3, name='split_' + scope)
    base = scope
    for i, (input_tensor, out_size) in enumerate(zip(inputs, output_splits)):
        scope = base + '_part_%d' % (i,)
        n = slim.conv2d(input_tensor, out_size, [1, 1], scope=scope, **kwargs)
        n = tf.identity(n, scope + '_output')
        outs.append(n)
    return tf.concat(outs, 3, name=scope + '_concat')


@slim.add_arg_scope
def expanded_conv(input_tensor,
                  num_outputs,
                  expansion_size=expand_input_by_factor(6),
                  stride=1,
                  rate=1,
                  kernel_size=(3, 3),
                  residual=True,
                  normalizer_fn=None,
                  split_projection=1,
                  split_expansion=1,
                  split_divisible_by=8,
                  expansion_transform=None,
                  depthwise_location='expansion',
                  depthwise_channel_multiplier=1,
                  endpoints=None,
                  use_explicit_padding=False,
                  padding='SAME',
                  inner_activation_fn=None,
                  depthwise_activation_fn=None,
                  project_activation_fn=tf.identity,
                  depthwise_fn=slim.separable_conv2d,
                  expansion_fn=split_conv,
                  projection_fn=split_conv,
                  scope=None):
    conv_defaults = {}
    dw_defaults = {}
    if inner_activation_fn is not None:
        conv_defaults['activation_fn'] = inner_activation_fn
        dw_defaults['activation_fn'] = inner_activation_fn
    if depthwise_activation_fn is not None:
        dw_defaults['activation_fn'] = depthwise_activation_fn
    # pylint: disable=g-backslash-continuation
    with tf.variable_scope(scope, default_name='expanded_conv') as s, \
            tf.name_scope(s.original_name_scope), \
            slim.arg_scope((slim.conv2d,), **conv_defaults), \
            slim.arg_scope((slim.separable_conv2d,), **dw_defaults):
        prev_depth = input_tensor.get_shape().as_list()[3]
        if depthwise_location not in [None, 'input', 'output', 'expansion']:
            raise TypeError('%r is unknown value for depthwise_location' % depthwise_location)
        if use_explicit_padding:
            if padding != 'SAME':
                raise TypeError('`use_explicit_padding` should only be used with "SAME" padding.')
            padding = 'VALID'
        depthwise_func = functools.partial(depthwise_fn,
                                           num_outputs=None,
                                           kernel_size=kernel_size,
                                           depth_multiplier=depthwise_channel_multiplier,
                                           stride=stride,
                                           rate=rate,
                                           normalizer_fn=normalizer_fn,
                                           padding=padding,
                                           scope='depthwise')
        input_tensor = tf.identity(input_tensor, 'input')
        net = input_tensor

        if depthwise_location == 'input':
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net, activation_fn=None)
            net = tf.identity(net, name='depthwise_output')
            if endpoints is not None:
                endpoints['depthwise_output'] = net

        if callable(expansion_size):
            inner_size = expansion_size(num_inputs=prev_depth)
        else:
            inner_size = expansion_size

        if inner_size > net.shape[3]:
            if expansion_fn == split_conv:
                expansion_fn = functools.partial(expansion_fn,
                                                 num_ways=split_expansion,
                                                 divisible_by=split_divisible_by,
                                                 stride=1)
            net = expansion_fn(net,
                               inner_size,
                               scope='expand',
                               normalizer_fn=normalizer_fn)
            net = tf.identity(net, 'expansion_output')
            if endpoints is not None:
                endpoints['expansion_output'] = net

        if depthwise_location == 'expansion':
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net)
            net = tf.identity(net, name='depthwise_output')
            if endpoints is not None:
                endpoints['depthwise_output'] = net

        if expansion_transform:
            net = expansion_transform(expansion_tensor=net, input_tensor=input_tensor)
        if projection_fn == split_conv:
            projection_fn = functools.partial(projection_fn,
                                              num_ways=split_projection,
                                              divisible_by=split_divisible_by,
                                              stride=1)
        net = projection_fn(net,
                            num_outputs,
                            scope='project',
                            normalizer_fn=normalizer_fn,
                            activation_fn=project_activation_fn)
        if endpoints is not None:
            endpoints['projection_output'] = net
        if depthwise_location == 'output':
            if use_explicit_padding:
                net = _fixed_padding(net, kernel_size, rate)
            net = depthwise_func(net, activation_fn=None)
            net = tf.identity(net, name='depthwise_output')
            if endpoints is not None:
                endpoints['depthwise_output'] = net

        if callable(residual):
            net = residual(input_tensor=input_tensor, output_tensor=net)
        elif residual and stride == 1 and net.get_shape().as_list()[3] == input_tensor.get_shape().as_list()[3]:
            net += input_tensor
        return tf.identity(net, name='output')


@slim.add_arg_scope
def squeeze_excite(input_tensor,
                   divisible_by=8,
                   squeeze_factor=3,
                   inner_activation_fn=tf.nn.relu,
                   gating_fn=tf.sigmoid,
                   squeeze_input_tensor=None,
                   pool=None):
    with tf.variable_scope('squeeze_excite'):
        if squeeze_input_tensor is None:
            squeeze_input_tensor = input_tensor
        input_size = input_tensor.shape.as_list()[1:3]
        pool_height, pool_width = squeeze_input_tensor.shape.as_list()[1:3]
        stride = 1
        if pool is not None and pool_height >= pool:
            pool_height, pool_width, stride = pool, pool, pool
        input_channels = squeeze_input_tensor.shape.as_list()[3]
        output_channels = input_tensor.shape.as_list()[3]
        squeeze_channels = _make_divisible(input_channels / squeeze_factor, divisor=divisible_by)

        pooled = tf.nn.avg_pool(squeeze_input_tensor,
                                (1, pool_height, pool_width, 1),
                                strides=(1, stride, stride, 1),
                                padding='VALID')
        squeeze = slim.conv2d(pooled,
                              kernel_size=(1, 1),
                              num_outputs=squeeze_channels,
                              normalizer_fn=None,
                              activation_fn=inner_activation_fn)
        excite_outputs = output_channels
        excite = slim.conv2d(squeeze, num_outputs=excite_outputs,
                             kernel_size=[1, 1],
                             normalizer_fn=None,
                             activation_fn=gating_fn)
        if pool is not None:
            excite = tf.image.resize_images(excite, input_size, align_corners=True)
        result = input_tensor * excite
    return result
