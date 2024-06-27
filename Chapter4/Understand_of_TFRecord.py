import tensorflow as tf
import tensorflow_datasets as tfds
import os

filename = os.path.join(os.path.expanduser('~') + '/tensorflow_datasets/mnist/3.0.1/mnist-test.tfrecord-00000-of-00001')
raw_dataset = tf.data.TFRecordDataset(filename)

for raw_record in raw_dataset.take(1):
    print(repr(raw_record))