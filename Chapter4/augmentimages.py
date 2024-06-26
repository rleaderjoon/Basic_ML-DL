import tensorflow as tf
import tensorflow_datasets as tfds
import math

def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rotate(image, 40, interpolation = 'NEAREST')
    return image, label

data = tfds.load('horses_or_humans', split = 'train', as_supervised=True)

train_batches = data.batch(1)

print(train_batches)

