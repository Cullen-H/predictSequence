import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

index_array = [1,3,4,5,6]
sequence_array = [10,30,40,50,60]

xs = np.array(index_array, dtype=float)
ys = np.array(sequence_array, dtype=float)

model.fit(xs, ys, epochs=500)

results = model.predict([10])

print(results)
