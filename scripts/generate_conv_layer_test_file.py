import numpy as np
from mlp.layers import ConvolutionalLayer
import argparse

parser = argparse.ArgumentParser(description='Welcome to Conv test script')

parser.add_argument('--student_id', nargs="?", type=str, help='Your student id in the format "sxxxxxxx"')

args = parser.parse_args()

student_id = args.student_id


def fprop_bprop_conv(inputs, grads_wrt_outputs, kernels, biases):
    activation_layer = ConvolutionalLayer(num_input_channels=3, num_output_channels=2, input_height=4, input_width=4,
                                          kernel_height=2, kernel_width=2)
    activation_layer.params = [kernels, biases]
    fprop = activation_layer.fprop(inputs)
    bprop = activation_layer.bprop(inputs, fprop, grads_wrt_outputs)
    grads_wrt_weights, grads_wrt_biases = activation_layer.grads_wrt_params(inputs, grads_wrt_outputs)

    return fprop, bprop, grads_wrt_weights, grads_wrt_biases


def get_student_seed(student_id):
    student_seed_number = int(student_id[1:])
    return student_seed_number


seed = get_student_seed(student_id)
rng = np.random.RandomState(seed)

input_output_dict = dict()
test_inputs = rng.normal(size=(2, 3, 4, 4))

test_grads_wrt_outputs = rng.normal(size=(2, 2, 3, 3))
test_cross_correlation_grads_wrt_outputs = test_grads_wrt_outputs[:, :, ::-1, ::-1]

test_kernels = rng.normal(size=(2, 3, 2, 2))
test_biases = rng.normal(size=2)
test_cross_correlation_kernels = test_kernels[:, :, ::-1, ::-1]

fprop_conv, bprop_conv, grads_wrt_weights_conv, grads_wrt_biases_conv = fprop_bprop_conv(inputs=test_inputs,
                                                                                         grads_wrt_outputs=test_grads_wrt_outputs,
                                                                                         kernels=test_kernels,
                                                                                         biases=test_biases)

fprop_cor, bprop_cor, grads_wrt_weights_cor, grads_wrt_biases_cor = fprop_bprop_conv(inputs=test_inputs,
                                                                                     grads_wrt_outputs=test_cross_correlation_grads_wrt_outputs,
                                                                                     kernels=test_cross_correlation_kernels,
                                                                                     biases=test_biases)

np.savez("test_convolution_results_pack", test_inputs=test_inputs, test_grads_wrt_outputs=test_grads_wrt_outputs,
         test_cross_correlation_grads_wrt_outputs=test_cross_correlation_grads_wrt_outputs, test_kernels=test_kernels,
         test_biases=test_biases, test_cross_correlation_kernels=test_cross_correlation_kernels,
         fprop_conv=fprop_conv, bprop_conv=bprop_conv, grads_wrt_weights_conv=grads_wrt_weights_conv,
         grads_wrt_biases_conv=grads_wrt_biases_conv, fprop_cor=fprop_cor, bprop_cor=bprop_cor,
         grads_wrt_weights_cor=grads_wrt_weights_cor,
         grads_wrt_biases_cor=grads_wrt_biases_cor)

results = np.load("test_convolution_results_pack.npz")
for key, value in results.items():
    print(key, value)