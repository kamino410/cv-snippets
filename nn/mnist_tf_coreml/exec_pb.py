import tensorflow as tf
from tensorflow import keras


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images / 255.

with tf.gfile.GFile('./mnist.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(
        graph_def,
        name='my_graph')

graph_def = graph.as_graph_def()

for op in graph.get_operations():
    print(op.name)
    print(' ', op.outputs)

out_layer = graph.get_tensor_by_name('my_graph/dense_1/Softmax:0')
with tf.Session(graph=graph) as sess:
    res = sess.run(out_layer, feed_dict={
        'my_graph/flatten_input:0': test_images})
    print(res)
