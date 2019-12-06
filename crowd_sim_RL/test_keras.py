import tensorflow as tf
import os
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
