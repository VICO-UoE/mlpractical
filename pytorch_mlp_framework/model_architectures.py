import torch
import torch.nn as nn
import torch.nn.functional as F


class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, use_bias=False):
        """
        Initializes a fully connected network similar to the ones implemented previously in the MLP package.
        :param input_shape: The shape of the inputs going in to the network.
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every fcc layer.
        :param num_layers: Number of fcc layers (excluding dim reduction stages)
        :param use_bias: Whether our fcc layers will use a bias.
        """
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            self.layer_dict['fcc_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # initialize a fcc layer
                                                            out_features=self.num_filters,
                                                            bias=self.use_bias)

            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        self.logits_linear_layer = nn.Linear(in_features=out.shape[1],  # initialize the prediction output linear layer
                                             out_features=self.num_output_classes,
                                             bias=self.use_bias)
        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)
        # flatten inputs to shape (b, -1) where -1 is the dim resulting from multiplying the
        # shapes of all dimensions after the 0th dim

        for i in range(self.num_layers):
            out = self.layer_dict['fcc_{}'.format(i)](out)  # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        out = self.logits_linear_layer(out)  # apply the layer to the previous layer's outputs
        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()


class EmptyBlock(nn.Module):
    def __init__(self, input_shape=None, num_filters=None, kernel_size=None, padding=None, bias=None, dilation=None,
                 reduction_factor=None):
        super(EmptyBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        self.layer_dict['Identity'] = nn.Identity()

    def forward(self, x):
        out = x

        out = self.layer_dict['Identity'].forward(out)

        return out


class EntryConvolutionalBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(EntryConvolutionalBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        self.layer_dict['bn_0'] = nn.BatchNorm2d(num_features=out.shape[1])
        out = F.leaky_relu(self.layer_dict['bn_0'].forward(out))

        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(self.layer_dict['bn_0'].forward(out))

        return out


class ConvolutionalProcessingBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation):
        super(ConvolutionalProcessingBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        return out


class ConvolutionalDimensionalityReductionBlock(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, padding, bias, dilation, reduction_factor):
        super(ConvolutionalDimensionalityReductionBlock, self).__init__()

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.reduction_factor = reduction_factor
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        self.layer_dict['conv_0'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        self.layer_dict['conv_1'] = nn.Conv2d(in_channels=out.shape[1], out_channels=self.num_filters, bias=self.bias,
                                              kernel_size=self.kernel_size, dilation=self.dilation,
                                              padding=self.padding, stride=1)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x):
        out = x

        out = self.layer_dict['conv_0'].forward(out)
        out = F.leaky_relu(out)

        out = F.avg_pool2d(out, self.reduction_factor)

        out = self.layer_dict['conv_1'].forward(out)
        out = F.leaky_relu(out)

        return out


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters,
                 num_blocks_per_stage, num_stages, use_bias=False, processing_block_type=ConvolutionalProcessingBlock,
                 dimensionality_reduction_block_type=ConvolutionalDimensionalityReductionBlock):
        """
        Initializes a convolutional network module
        :param input_shape: The shape of the tensor to be passed into this network
        :param num_output_classes: Number of output classes
        :param num_filters: Number of filters per convolutional layer
        :param num_blocks_per_stage: Number of blocks per "stage". Each block is composed of 2 convolutional layers.
        :param num_stages: Number of stages in a network. A stage is defined as a sequence of layers within which the
        data dimensionality remains constant in the spacial axis (h, w) and can change in the channel axis. After each stage
        there exists a dimensionality reduction stage, composed of two convolutional layers and an avg pooling layer.
        :param use_bias: Whether to use biases in our convolutional layers
        :param processing_block_type: Type of processing block to use within our stages
        :param dimensionality_reduction_block_type: Type of dimensionality reduction block to use after each stage in our network
        """
        super(ConvolutionalNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_stages = num_stages
        self.processing_block_type = processing_block_type
        self.dimensionality_reduction_block_type = dimensionality_reduction_block_type

        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        self.layer_dict = nn.ModuleDict()
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        print("Building basic block of ConvolutionalNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers

        out = x
        self.layer_dict['input_conv'] = EntryConvolutionalBlock(input_shape=out.shape, num_filters=self.num_filters,
                                                                kernel_size=3, padding=1, bias=self.use_bias,
                                                                dilation=1)
        out = self.layer_dict['input_conv'].forward(out)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        for i in range(self.num_stages):  # for number of layers times
            for j in range(self.num_blocks_per_stage):
                self.layer_dict['block_{}_{}'.format(i, j)] = self.processing_block_type(input_shape=out.shape,
                                                                                         num_filters=self.num_filters,
                                                                                         bias=self.use_bias,
                                                                                         kernel_size=3, dilation=1,
                                                                                         padding=1)
                out = self.layer_dict['block_{}_{}'.format(i, j)].forward(out)
            self.layer_dict['reduction_block_{}'.format(i)] = self.dimensionality_reduction_block_type(
                input_shape=out.shape,
                num_filters=self.num_filters, bias=True,
                kernel_size=3, dilation=1,
                padding=1,
                reduction_factor=2)
            out = self.layer_dict['reduction_block_{}'.format(i)].forward(out)

        out = F.avg_pool2d(out, out.shape[-1])
        print('shape before final linear layer', out.shape)
        out = out.view(out.shape[0], -1)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=True)
        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        out = self.layer_dict['input_conv'].forward(out)
        for i in range(self.num_stages):  # for number of layers times
            for j in range(self.num_blocks_per_stage):
                out = self.layer_dict['block_{}_{}'.format(i, j)].forward(out)
            out = self.layer_dict['reduction_block_{}'.format(i)].forward(out)

        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

        self.logit_linear_layer.reset_parameters()
