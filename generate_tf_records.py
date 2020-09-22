import multiprocessing

from src.utils import data_util, config, dataset

if __name__ == '__main__':
    my_queue = multiprocessing.Manager().Queue()
    sentinel = ("", [])
    paths = data_util.get_images(config.training_image_path)
    writer = dataset.TfRecordsWriter(paths, multiprocessing.cpu_count(), my_queue, sentinel)
    writer.run()

#     reader = dataset.TfRecordsReader()
#     tf_record_paths = [('dataset/tf_records/' + name) for name in os.listdir('dataset/tf_records/') if
#                        name.endswith('.tfrecords')]
#     images, score_maps, geo_maps, training_masks, paths = reader.inputs(tf_record_paths, 16, psutil.cpu_count())
#     with tf.Session() as session:
#         images, score_maps, geo_maps, training_masks, paths = session.run([images,
#                                                                            score_maps,
#                                                                            geo_maps,
#                                                                            training_masks,
#                                                                            paths])
#         print(images.shape)
