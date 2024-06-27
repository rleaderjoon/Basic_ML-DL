import tensorflow as tf
import tensorflow_datasets as tfds

#data = tfds.load('cats_vs_dogs', split = 'train[:20%]', as_supervised = True)
#data = tfds.load('cats_vs_dogs', split = 'train[:10000]', as_supervised = True)
#data = tfds.load('cats_vs_dogs', split = 'train[-1000:] + train[:1000]', as_supervised = True)

train_data = tfds.load('cats_vs_dogs', split = 'train[:80%]', as_supervised = True)
validation_data = tfds.load('cats_vs_dogs', split = 'train[80%:90%]', as_supervised  = True)
test_data = tfds.load('cats_vs_dogs', split = 'train[-10%:]', as_supervised = True)

print(tf.data.experimental.cardinality(train_data))