import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image)
    return image, label

data = tfds.load('horses_or_humans', split = 'train', as_supervised=True)
data = data.map(augmentimages)

for image, label in data.take(1):
    plt.figure()
    plt.imshow(tf.squeeze(image))
    plt.title(f'Label: {label.numpy()}')
    plt.show()

