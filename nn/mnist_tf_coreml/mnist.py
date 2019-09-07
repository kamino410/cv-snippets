import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_io


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape([-1, 28, 28, 1]) / 255.
test_images = test_images.reshape([-1, 28, 28, 1]) / 255.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=15)
print(model.evaluate(test_images, test_labels))

print('inputs:', [layer.op.name for layer in model.inputs])
print('outputs:', [layer.op.name for layer in model.outputs])

model.save('mnist.h5')
