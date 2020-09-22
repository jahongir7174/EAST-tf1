import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from src.utils import config


def up_sampling(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])


def build_model(arg_scope, fe_model, weight_decay=1e-5, is_training=True):
    if config.base_model == 'efficient':
        _, end_points, f = fe_model
    else:
        with slim.arg_scope(arg_scope):
            _, end_points, f = fe_model
    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True, 'is_training': is_training}
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            x = up_sampling(f[0])
            x = tf.concat([x, f[1]], axis=-1)
            x = slim.conv2d(x, num_outputs=128, kernel_size=1)
            x = slim.conv2d(x, num_outputs=128, kernel_size=3)

            x = up_sampling(x)
            x = tf.concat([x, f[2]], axis=-1)
            x = slim.conv2d(x, num_outputs=64, kernel_size=1)
            x = slim.conv2d(x, num_outputs=64, kernel_size=3)

            x = up_sampling(x)
            x = tf.concat([x, f[3]], axis=-1)
            x = slim.conv2d(x, num_outputs=32, kernel_size=1)
            x = slim.conv2d(x, num_outputs=32, kernel_size=3)

            x = slim.conv2d(x, num_outputs=32, kernel_size=3)

            f_score = slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            geo_map = slim.conv2d(x, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * config.text_scale
            angle_map = (slim.conv2d(x, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi / 2
            f_geometry = tf.concat([geo_map, angle_map], axis=-1)
    return x, f_score, f_geometry


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    _loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', _loss)
    return _loss


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    cls_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    cls_loss *= 0.01
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    l_log = -tf.math.log((area_intersect + 1.0) / (area_union + 1.0))
    l_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(l_log * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(l_theta * y_true_cls * training_mask))
    l_g = l_log + 20 * l_theta
    return tf.reduce_mean(l_g * y_true_cls * training_mask) + cls_loss
