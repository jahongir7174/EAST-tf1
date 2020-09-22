import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import multiprocessing
from src.nets import detection
from src.nets.base_nets import mobilenet_v3, efficientnet_builder, mobilenet_v1, resnet_50
from src.utils import config, dataset
from src.utils import util

gpu_list = list(range(len(config.gpu_list.split(','))))


def get_model(images):
    if config.base_model == 'resnet_50':
        with slim.arg_scope(resnet_50.arg_scope()):
            arg_scope = resnet_50.arg_scope()
            fe_model = resnet_50.resnet_v1_50(images, is_training=True)
    elif config.base_model == 'mobilenet_v1':
        arg_scope = mobilenet_v1.arg_scope(is_training=True)
        fe_model = mobilenet_v1.mobile_net(images)
    elif config.base_model == 'mobilenet_v3':
        arg_scope = mobilenet_v3.arg_scope(is_training=True)
        fe_model = mobilenet_v3.mobile_net(images)
    else:
        arg_scope = None
        fe_model = efficientnet_builder.build_model_base(images, 'efficientnet-b0', True)
    return arg_scope, fe_model


def get_model_path():
    if config.base_model == 'resnet_50':
        model_name = 'resnet_50.ckpt'
        model_path = config.resnet_50_path
    elif config.base_model == 'mobilenet_v1':
        model_name = 'mobilenet_v1.ckpt'
        model_path = config.mobilenet_v1_path
    elif config.base_model == 'mobilenet_v3':
        model_name = 'mobilenet_v3.ckpt'
        model_path = config.mobilent_v3_path
    else:
        model_name = 'efficient_net.ckpt'
        model_path = config.efficient_path
    return model_name, model_path


def compute_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        arg_scope, fe_model = get_model(images)
        _, f_score, f_geometry = detection.build_model(arg_scope, fe_model)
    model_loss = detection.loss(score_maps, f_score, geo_maps, f_geometry, training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return total_loss, model_loss


def average_gradients(_grads):
    average_grads = []
    for grad_and_vars in zip(*_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main():
    model_name, model_path = get_model_path()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_list
    if not tf.gfile.Exists(model_path):
        tf.gfile.MkDir(model_path)
    else:
        if not config.restore:
            tf.gfile.DeleteRecursively(model_path)
            tf.gfile.MkDir(model_path)
    reader = dataset.TfRecordsReader()
    tf_record_paths = [(config.tf_records_path + name) for name in os.listdir(config.tf_records_path) if
                       name.endswith('.tfrecords')]
    input_images, input_score_maps, input_geo_maps, input_training_masks, _ = reader.inputs(tf_record_paths,
                                                                                            config.batch_size,
                                                                                            multiprocessing.cpu_count())

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate,
                                               global_step,
                                               decay_steps=10000,
                                               decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    # split
    input_images_split = tf.split(input_images, len(gpu_list))
    input_score_maps_split = tf.split(input_score_maps, len(gpu_list))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpu_list))
    input_training_masks_split = tf.split(input_training_masks, len(gpu_list))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpu_list):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss = compute_loss(iis, isms, igms, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(model_path, tf.get_default_graph())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if config.restore:
            print('continue training from previous checkpoint')
            saver.restore(sess, model_path)
        else:
            sess.run(tf.global_variables_initializer())

        nb_iterations = len(tf_record_paths) // config.batch_size
        current_loss = sys.float_info.max

        print("The model has {} trainable parameters".format(util.get_nb_weights()))
        print("--- Training with {} ---".format(config.base_model))
        for step in range(config.nb_epochs):
            p_bar = util.ProgressBar(total=nb_iterations)

            sum_loss = 0
            for _ in range(nb_iterations):
                ml, tl, _ = sess.run([model_loss, total_loss, train_op])
                sum_loss += tl
                if np.isnan(tl):
                    print('Loss diverged, stop training')
                    break
                p_bar.current += 1
                p_bar(epoch=step, m_loss=ml, t_loss=tl)
            p_bar.done(epoch=step, m_loss=ml, t_loss=tl)
            if current_loss > sum_loss:
                saver.save(sess, os.path.join(model_path + model_name), global_step=global_step)
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op])
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    main()
