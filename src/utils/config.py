input_size = 512
batch_size = 1
learning_rate = 0.0001
nb_epochs = 10
moving_average_decay = 0.997
gpu_list = '0'
resnet_50_path = 'weights/resnet_50/'
mobilenet_v1_path = 'weights/mobilenet_v1/'
mobilent_v3_path = 'weights/mobilenet_v3/'
efficient_path = 'weights/efficient/'
restore = False
training_image_path = 'dataset/images/'
training_label_path = 'dataset/labels/'
tf_records_path = 'dataset/tf_records/'
test_data_path = 'dataset/test/'
base_model = 'efficient'  # mobilenet_v3, mobilenet_v1, resnet_50, efficient
min_text_size = 10
min_crop_side_ratio = 0.1
geometry = 'RBOX'
text_scale = 512
output_dir = 'results/'
no_write_images = False
