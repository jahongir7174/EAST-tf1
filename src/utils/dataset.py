from multiprocessing import Process

import tensorflow as tf

from src.utils import config
from src.utils.data_util import _write_tf_records


class TfRecordsWriter:
    def __init__(self, data_paths, nb_process, _queue, _sentinel):
        self.__data_paths = data_paths
        self._nb_process = nb_process
        self._SAMPLE_INFO_QUEUE = _queue
        self._SENTINEL = _sentinel
        self._init_queue()

    def _init_queue(self):
        for __data_path in self.__data_paths:
            self._SAMPLE_INFO_QUEUE.put(__data_path)

        for i in range(self._nb_process):
            self._SAMPLE_INFO_QUEUE.put(self._SENTINEL)

    def run(self):
        process_pool = []
        for i in range(self._nb_process):
            process = Process(target=_write_tf_records,
                              name='Subprocess_{:d}'.format(i + 1),
                              args=(self._SAMPLE_INFO_QUEUE, self._SENTINEL))
            process_pool.append(process)
            process.start()
        for process in process_pool:
            process.join()


class TfRecordsReader:
    @staticmethod
    def _extract_features_batch(_nb_batch):
        features = tf.parse_example(_nb_batch,
                                    features={'image': tf.FixedLenFeature([], tf.string),
                                              'path': tf.FixedLenFeature([], tf.string),
                                              'score_map': tf.FixedLenFeature([], tf.string),
                                              'geo_map': tf.FixedLenFeature([], tf.string),
                                              'training_mask': tf.FixedLenFeature([], tf.string)})
        nb_batches = features['image'].shape[0]
        _shape = config.input_size // 4

        images = tf.decode_raw(features['image'], tf.uint8)
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.reshape(images, [nb_batches, config.input_size, config.input_size, 1])

        paths = features['path']

        score_maps = tf.decode_raw(features['score_map'], tf.float32)
        score_maps = tf.cast(x=score_maps, dtype=tf.float32)
        score_maps = tf.reshape(score_maps, [nb_batches, _shape, _shape, 1])

        geo_maps = tf.decode_raw(features['geo_map'], tf.float32)
        geo_maps = tf.cast(x=geo_maps, dtype=tf.float32)
        geo_maps = tf.reshape(geo_maps, [nb_batches, _shape, _shape, 5])

        training_masks = tf.decode_raw(features['training_mask'], tf.float32)
        training_masks = tf.cast(x=training_masks, dtype=tf.float32)
        training_masks = tf.reshape(training_masks, [nb_batches, _shape, _shape, 1])
        return images, score_maps, geo_maps, training_masks, paths

    def inputs(self, file_names, _batch_size, _nb_threads):
        reader = tf.data.TFRecordDataset(file_names).batch(_batch_size, drop_remainder=True)
        reader = reader.map(map_func=self._extract_features_batch, num_parallel_calls=_nb_threads)
        reader = reader.shuffle(buffer_size=1000)
        reader = reader.repeat(count=config.nb_epochs + 1)
        iterator = reader.make_one_shot_iterator()
        return iterator.get_next()
