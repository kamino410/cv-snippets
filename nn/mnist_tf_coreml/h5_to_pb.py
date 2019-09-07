import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('input_h5')
parser.add_argument('output_pb')

args = parser.parse_args()


model = load_model(args.input_h5)

output_node_names = [node.op.name for node in model.outputs]
sess = tf.compat.v1.keras.backend.get_session()

frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    output_node_names)

with open(args.output_pb, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
