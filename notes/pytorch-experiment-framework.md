# PyTorch Experiment Framework

## What does this framework do?
The PyTorch experiment framework located in ```mlp/pytorch_mlp_framework``` includes tooling for building an array of deep neural networks,
including fully connected and convolutional networks. In addition, it also includes tooling for experiment running, 
metric handling and storage, model weight storage, checkpointing (allowing continuation from previous saved point), as 
well as taking care of keeping track of the best validation model which is then used as the end to produce test set evaluation metrics.

## Why do we need it?
It serves two main purposes. The first, is to allow you an easy, worry-free transition into using PyTorch for experiments
 in your coursework. The second, is to teach you good coding practices for building and running deep learning experiments
  using PyTorch. The framework comes fully loaded with tooling that can keep track of relevant metrics, save models, resume from previous saved states and 
  even automatically choose the best validation model for test set evaluation. We include documentation and comments in almost 
  every single line of code in the framework, to help you maximize your learning. The code style itself, can be used for
   learning good programming practices in structuring your code in a modular, readable and computationally efficient manner that minimizes chances of user-error.

## Installation

First thing you have to do is activate your conda MLP environment. 

### GPU version on Google Compute Engine

For usage on google cloud, the disk image we provide comes pre-loaded with all the packages you need to run the PyTorch
experiment framework, including PyTorch itself.  Thus when you created an instance and setup your environment, everything you need for this framework was installed, thus removing the need for you to install PyTorch.

### CPU version on DICE (or other local machine)

If you do not have your MLP conda environment installed on your current machine please follow the instructions in the [MLP environment installation guide](notes/environment-set-up.md). It includes an explanation on how to install a CPU version of PyTorch, or a GPU version if you have a GPU available on your local machine.

Once PyTorch is installed in your MLP conda enviroment, you can start using the framework. The framework has been built to allow you to control your experiment hyperparameters directly from the command line, by using command line argument parsing.

## Using the framework

You can get a list of all available hyperparameters and arguments by using:
```
python pytorch_mlp_framework/train_evaluate_image_classification_system.py -h
```

The -h at the end is short for --help, which presents a list with all possible arguments next to a description of what they modify in the setup.
Once you execute that command, you should be able to see the following list:

```
Welcome to the MLP course's PyTorch training and inference helper script

optional arguments:
  -h, --help            show this help message and exit
  --batch_size [BATCH_SIZE]
                        Batch_size for experiment
  --continue_from_epoch [CONTINUE_FROM_EPOCH]
                        Which epoch to continue from. 
                        If -2, continues from where it left off
                        If -1, starts from scratch
                        if >=0, continues from given epoch
  --seed [SEED]         Seed to use for random number generator for experiment
  --image_num_channels [IMAGE_NUM_CHANNELS]
                        The channel dimensionality of our image-data
  --image_height [IMAGE_HEIGHT]
                        Height of image data
  --image_width [IMAGE_WIDTH]
                        Width of image data
  --num_stages [NUM_STAGES]
                        Number of convolutional stages in the network. A stage
                        is considered a sequence of convolutional layers where
                        the input volume remains the same in the spacial
                        dimension and is always terminated by a dimensionality
                        reduction stage
  --num_blocks_per_stage [NUM_BLOCKS_PER_STAGE]
                        Number of convolutional blocks in each stage, not
                        including the reduction stage. A convolutional block
                        is made up of two convolutional layers activated using
                        the leaky-relu non-linearity
  --num_filters [NUM_FILTERS]
                        Number of convolutional filters per convolutional
                        layer in the network (excluding dimensionality
                        reduction layers)
  --num_epochs [NUM_EPOCHS]
                        The experiment's epoch budget
  --num_classes [NUM_CLASSES]
                        The experiment's epoch budget
  --experiment_name [EXPERIMENT_NAME]
                        Experiment name - to be used for building the
                        experiment folder
  --use_gpu [USE_GPU]   A flag indicating whether we will use GPU acceleration
                        or not
  --weight_decay_coefficient [WEIGHT_DECAY_COEFFICIENT]
                        Weight decay to use for Adam
  --block_type BLOCK_TYPE
                        Type of convolutional blocks to use in our network
                        (This argument will be useful in running experiments
                        to debug your network)

```

For example, to run a simple experiment using a 7-layer convolutional network on the CPU you can run:

```
python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 0 --experiment_name VGG_07 --num_classes 100 --block_type 'conv_block' --weight_decay_coefficient 0.00000 --use_gpu False
```

Your experiment should begin running.

Your experiments statistics and model weights are saved in the directory tutorial_exp_1/ under tutorial_exp_1/logs and 
tutorial_exp_1/saved_models.


To run on a GPU on Google Compute Engine the command would be:
```
python pytorch_mlp_framework/train_evaluate_image_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --num_stages 3 --num_blocks_per_stage 0 --experiment_name VGG_07 --num_classes 100 --block_type 'conv_block' --weight_decay_coefficient 0.00000 --use_gpu True

```

We have also provided the exact scripts we used to run the experiments of VGG07 and VGG37 as shown in the coursework spec inside the files:
- run_vgg_08_default.sh
- run_vgg_38_default.sh

**However, remember, if you want to reuse those scripts for your own investigations, change the experiment name and seed.
If you do not change the name, the old folders will be overwritten.**

## So, where can I ask more questions and find more information on PyTorch and what it can do?

First course of action should be to search the web and then to refer to the PyTorch [documentation](https://pytorch.org/docs/stable/index.html),
 [tutorials](https://pytorch.org/tutorials/) and [github](https://github.com/pytorch/pytorch) sites.
 
 If you still can't get an answer to your question then as always, post on Piazza and/or come to the lab sessions.
 
