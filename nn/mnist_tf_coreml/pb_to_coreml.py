import tensorflow as tf
import tfcoreml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('input_pb')
parser.add_argument('output_mlmodel')
args = parser.parse_args()


model = tfcoreml.convert(tf_model_path=args.input_pb,
                         mlmodel_path=args.output_mlmodel,
                         output_feature_names=['dense_1/Softmax:0'],
                         input_name_shape_dict={
                             'flatten_input:0': [1, 28, 28, 1]},
                         image_input_names=['flatten_input:0'])

spec = model.get_spec()
print(spec.description.output)
