import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Epoch you want to continue training from while restarting an experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--image_num_channels', nargs="?", type=int, default=3,
                        help='The channel dimensionality of our image-data')
    parser.add_argument('--image_height', nargs="?", type=int, default=32, help='Height of image data')
    parser.add_argument('--image_width', nargs="?", type=int, default=32, help='Width of image data')
    parser.add_argument('--num_stages', nargs="?", type=int, default=3,
                        help='Number of convolutional stages in the network. A stage is considered a sequence of '
                             'convolutional layers where the input volume remains the same in the spacial dimension and'
                             ' is always terminated by a dimensionality reduction stage')
    parser.add_argument('--num_blocks_per_stage', nargs="?", type=int, default=5,
                        help='Number of convolutional blocks in each stage, not including the reduction stage.'
                             ' A convolutional block is made up of two convolutional layers activated using the '
                             ' leaky-relu non-linearity')
    parser.add_argument('--num_filters', nargs="?", type=int, default=16,
                        help='Number of convolutional filters per convolutional layer in the network (excluding '
                             'dimensionality reduction layers)')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='Total number of epochs for model training')
    parser.add_argument('--num_classes', nargs="?", type=int, default=100, help='Number of classes in the dataset')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0,
                        help='Weight decay to use for Adam')
    parser.add_argument('--block_type', type=str, default='conv_block',
                        help='Type of convolutional blocks to use in our network '
                             '(This argument will be useful in running experiments to debug your network)')
    args = parser.parse_args()
    print(args)
    return args
