import tensorflow as tf
import os
import numpy as np

inputs = tf.random.normal([1,1,2])
print(inputs)
rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))
output = rnn(inputs)
print(output.shape)