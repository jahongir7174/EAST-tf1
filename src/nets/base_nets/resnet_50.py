import collections

import tensorflow as tf
from tensorflow.contrib import slim


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        output = tf.nn.relu(shortcut + residual)
        return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None, outputs_collections=None):
    current_stride = 1
    rate = 1
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=1,
                                            rate=rate)
                        rate *= unit_stride
                    else:
                        net = block.unit_fn(net,
                                            depth=unit_depth,
                                            depth_bottleneck=unit_depth_bottleneck,
                                            stride=unit_stride,
                                            rate=1)
                        current_stride *= unit_stride
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return net


def arg_scope(weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {'decay': batch_norm_decay,
                         'epsilon': batch_norm_epsilon,
                         'scale': batch_norm_scale,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS, }
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def resnet_v1(inputs, blocks, is_training=True, output_stride=None, include_root_block=True, reuse=None, scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                    net = slim.utils.collect_named_outputs(end_points_collection, 'pool2', net)
                net = stack_blocks_dense(net, blocks, output_stride)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                try:
                    end_points['pool3'] = end_points['resnet_v1_50/block1']
                    end_points['pool4'] = end_points['resnet_v1_50/block2']
                except:
                    end_points['pool3'] = end_points['Detection/resnet_v1_50/block1']
                    end_points['pool4'] = end_points['Detection/resnet_v1_50/block2']
                end_points['pool5'] = net
                f = [end_points['pool5'], end_points['pool4'], end_points['pool3'], end_points['pool2']]
                return net, end_points, f


def resnet_v1_50(inputs, is_training=True, output_stride=None, reuse=None, scope='resnet_v1_50'):
    blocks = [Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
              Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
              Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
              Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v1(inputs, blocks, is_training, output_stride=output_stride,
                     include_root_block=True, reuse=reuse, scope=scope)
