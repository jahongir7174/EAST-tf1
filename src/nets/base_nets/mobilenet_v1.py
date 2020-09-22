from collections import namedtuple

import tensorflow as tf

slim = tf.contrib.slim
alpha = 1.0
depth_multiplier = 1

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

MOBILENETV1_CONV_DEFS = [Conv(kernel=[3, 3], stride=2, depth=32),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=64),
                         DepthSepConv(kernel=[3, 3], stride=2, depth=128),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=128),
                         DepthSepConv(kernel=[3, 3], stride=2, depth=256),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=256),
                         DepthSepConv(kernel=[3, 3], stride=2, depth=512),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=512),
                         DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
                         DepthSepConv(kernel=[3, 3], stride=1, depth=1024)]


def _fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def mobilenet_v1_base(inputs, final_endpoint='Conv2d_13_pointwise', min_depth=8, depth_multiplier=1.0,
                      conv_defs=None, output_stride=None, use_explicit_padding=False, scope=None):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if conv_defs is None:
        conv_defs = MOBILENETV1_CONV_DEFS
    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')
    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            current_stride = 1
            rate = 1
            net = inputs
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i
                if output_stride is not None and current_stride == output_stride:
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=layer_stride,
                                                rate=layer_rate,
                                                scope=end_point)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                    end_point = end_point_base + '_pointwise'

                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1], stride=1, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def mobile_net(inputs,
               is_training=True,
               min_depth=8,
               depth_multiplier=1.0,
               conv_defs=None,
               reuse=None,
               scope='MobilenetV1'):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = mobilenet_v1_base(inputs, scope=scope,
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier,
                                                conv_defs=conv_defs)
    end_points['pool5'] = net
    end_points['pool4'] = end_points['Conv2d_11_pointwise']
    end_points['pool3'] = end_points['Conv2d_5_pointwise']
    end_points['pool2'] = end_points['Conv2d_3_pointwise']
    f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]
    return net, end_points, f


def arg_scope(is_training=True,
              weight_decay=0.00004,
              stddev=0.09,
              regularize_depthwise=False,
              batch_norm_decay=0.9997,
              batch_norm_epsilon=0.001,
              batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
              normalizer_fn=slim.batch_norm):
    batch_norm_params = {'center': True,
                         'scale': True,
                         'decay': batch_norm_decay,
                         'epsilon': batch_norm_epsilon,
                         'updates_collections': batch_norm_updates_collections, }
    if is_training is not None:
        batch_norm_params['is_training'] = is_training

    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init, activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as sc:
                    return sc
