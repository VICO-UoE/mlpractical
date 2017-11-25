import numpy as np
from mlp.layers import ConvolutionalLayer
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
test_grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
inputs = np.arange(96).reshape((2, 3, 4, 4))
kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
biases = np.arange(2)

#produce ConvolutionalLayer fprop, bprop and grads_wrt_params
activation_layer = ConvolutionalLayer(num_input_channels=3, num_output_channels=2, input_dim_1=4, input_dim_2=4,
                                      kernel_dim_1=2, kernel_dim_2=2)
activation_layer.params = [kernels, biases]
conv_fprop = activation_layer.fprop(test_inputs)
conv_bprop = activation_layer.bprop(
    test_inputs, conv_fprop, test_grads_wrt_outputs)
conv_grads_wrt_params = activation_layer.grads_wrt_params(test_inputs,
                                                          test_grads_wrt_outputs)
test_output = "ConvolutionalLayer:\nFprop: {}\nBprop: {}\n" \
              "Grads_wrt_params: {}\n".format(conv_fprop,
            conv_bprop,
            conv_grads_wrt_params)

cross_correlation_kernels = kernels[:, :, ::-1, ::-1]
activation_layer = ConvolutionalLayer(num_input_channels=3, num_output_channels=2, input_dim_1=4, input_dim_2=4,
                                      kernel_dim_1=2, kernel_dim_2=2)
activation_layer.params = [cross_correlation_kernels, biases]
conv_fprop = activation_layer.fprop(test_inputs)
conv_bprop = activation_layer.bprop(
    test_inputs, conv_fprop, test_grads_wrt_outputs)
conv_grads_wrt_params = activation_layer.grads_wrt_params(test_inputs,
                                                          test_grads_wrt_outputs)

test_cross_correlation_output = "Cross_Correlation_ConvolutionalLayer:\nFprop: {}\nBprop: {}\n" \
              "Grads_wrt_params: {}\n".format(conv_fprop,
            conv_bprop,
            conv_grads_wrt_params)

test_output = test_output + "\n" + test_cross_correlation_output
with open("{}_conv_test_file.txt".format(student_id), "w+") as out_file:
    out_file.write(test_output)