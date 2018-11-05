import mlp.data_providers as data_providers
import numpy as np
from mlp.pytorch_experiment_scripts.arg_extractor import get_args
from mlp.pytorch_experiment_scripts.experiment_builder import ExperimentBuilder
from mlp.pytorch_experiment_scripts.model_architectures import ConvolutionalNetwork
import torch

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

train_data = data_providers.EMNISTDataProvider('train', batch_size=args.batch_size,
                                               rng=rng)  # initialize our rngs using the argument set seed
val_data = data_providers.EMNISTDataProvider('valid', batch_size=args.batch_size,
                                             rng=rng)  # initialize our rngs using the argument set seed
test_data = data_providers.EMNISTDataProvider('test', batch_size=args.batch_size,
                                              rng=rng)  # initialize our rngs using the argument set seed

custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    dim_reduction_type=args.dim_reduction_type,
    num_output_classes=train_data.num_classes, num_filters=args.num_filters, num_layers=args.num_layers, use_bias=False)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
