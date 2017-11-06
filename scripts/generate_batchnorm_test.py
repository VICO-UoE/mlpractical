import numpy as np
from mlp.layers import BatchNormalizationLayer
import argparse

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')

parser.add_argument('--student_id', nargs="?", type=str, help='Your student id in the format "sxxxxxxx"')

args = parser.parse_args()

student_id = args.student_id

def generate_inputs(student_id):
    student_number = student_id
    tests = np.arange(96).reshape((2, 3, 4, 4))
    tests[:, 0, :, :] = float(student_number[1:3]) / 10 - 5
    tests[:, :, 1, :] = float(student_number[3:5]) / 10 - 5
    tests[:, 2, :, :] = float(student_number[5:7]) / 10 - 5
    tests[0, 1, :, :] = float(student_number[7]) / 10 - 5
    return tests

test_inputs = generate_inputs(student_id)
test_inputs = np.reshape(test_inputs, newshape=(2, -1))
test_grads_wrt_outputs = np.arange(-48, 48).reshape((2, -1))

#produce BatchNorm Layer fprop and bprop
activation_layer = BatchNormalizationLayer(input_dim=48)

beta = np.array(48*[0.3])
gamma = np.array(48*[0.8])

activation_layer.params = [gamma, beta]
BN_fprop = activation_layer.fprop(test_inputs)
BN_bprop = activation_layer.bprop(
    test_inputs, BN_fprop, test_grads_wrt_outputs)
BN_grads_wrt_params = activation_layer.grads_wrt_params(
    test_inputs, test_grads_wrt_outputs)

test_output = "BatchNormalization:\nFprop: {}\nBprop: {}\nGrads_wrt_params: {}\n"\
    .format(BN_fprop, BN_bprop, BN_grads_wrt_params)

with open("{}_batchnorm_test_file.txt".format(student_id), "w+") as out_file:
    out_file.write(test_output)