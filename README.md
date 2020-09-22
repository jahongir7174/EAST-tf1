# EAST
EAST text detection network implementation using TensorFlow API 1 (multi gpu support)

Available base networks: Mobilenet V1, Mobilenet V3, ResNet50, EfficientNet 

Custom dataset training 
1. Put your images into images folder inside dataset and your gt data into labels folder inside dataset folder
2. run `python generate_tf_records.py` file. It generates .tfrecords data
3. run `python train.py` for training

Testing
1. Put your test images into test folder inside dataset folder
2. Compile lanms based on your operating system
3. run `python test.py`

References

https://github.com/argman/EAST
