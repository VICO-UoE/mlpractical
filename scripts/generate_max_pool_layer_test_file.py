import numpy as np
from mlp.layers import MaxPooling2DLayer
import argparse

parser = argparse.ArgumentParser(description='Welcome to the max pooling layer test script')

parser.add_argument('--student_id', nargs="?", type=str, help='Your student id in the format "sxxxxxxx"')

args = parser.parse_args()

student_id = args.student_id


def fprop_bprop_conv(inputs, rng):
    activation_layer = MaxPooling2DLayer(input_height=8, input_width=8, size=2, stride=2)
    fprop = activation_layer.fprop(inputs)
    grads_wrt_outputs = rng.normal(size=fprop.shape)
    bprop = activation_layer.bprop(inputs, fprop, grads_wrt_outputs)

    return grads_wrt_outputs, fprop, bprop


def get_student_seed(student_id):
    student_seed_number = int(student_id[1:])
    return student_seed_number


seed = get_student_seed(student_id)
rng = np.random.RandomState(seed)

input_output_dict = dict()
test_inputs = rng.normal(size=(2, 3, 8, 8))

test_grads_wrt_outputs, fprop_max, bprop_max = fprop_bprop_conv(inputs=test_inputs,
                                          rng=rng)

np.savez("test_max_pooling_results_pack", test_inputs=test_inputs, test_grads_wrt_outputs=test_grads_wrt_outputs,
         fprop_max=fprop_max, bprop_max=bprop_max)

results = np.load("test_max_pooling_results_pack.npz")
for key, value in results.items():
    print(key, value)
