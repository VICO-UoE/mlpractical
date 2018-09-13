# Machine Learning Practical

This repository contains the code for the University of Edinburgh [School of Informatics](http://www.inf.ed.ac.uk) course [Machine Learning Practical](http://www.inf.ed.ac.uk/teaching/courses/mlp/).

This assignment-based course is focused on the implementation and evaluation of machine learning systems. Students who do this course will have experience in the design, implementation, training, and evaluation of machine learning systems.

The code in this repository is split into:
1. notebooks: 
    1. Introduction_to_tensorflow: Introduces students to the basics of tensorflow and lower level operations.
    2. Introduction_to_tf_mlp_repo: Introduces students to the high level functionality of this repo and how one 
    could run an experiment. The code is full of comments and documentation so you should spend more time 
    reading and understanding the code by running simple experiments and changing pieces of code to see the impact 
    on the system.
2. utils: 
    1. network_summary: Provides utilities with which one can get network summaries, such as the number of parameters and names of layers.
    2. parser_utils which are used to parse arguments passed to the training scripts.
    3. storage, which is responsible for storing network statistics.
3. data_providers.py : Provides the data providers for training, validation and testing.
4. network_architectures.py: Defines the network architectures. We provide VGGNet as an example.
5. network_builder.py: Builds the tensorflow computation graph. In more detail, it builds the losses, tensorflow summaries and training operations.
6. network_trainer.py: Runs an experiment, composed of training, validation and testing. It is setup to use arguments such that one can easily write multiple bash scripts with different hyperparameters and run experiments very quickly with minimal code changes.
    
    
## Getting set up

Detailed instructions for setting up a development environment for the course are given in [this file](notes/environment-set-up.md). Students doing the course will spend part of the first lab getting their own environment set up.
Once you have setup the basic environment then to install the requirements for the tf_mlp repo simply run:
```
pip install -r requirements.txt
```
For CPU tensorflow and
```
pip install -r requirements_gpu.txt
```
for GPU tensorflow.

If you install the wrong version of tensorflow simply run

```
pip uninstall $tensorflow_to_uninstall
```
replacing $tensorflow_to_uninstall with the tensorflow you want to install and then install the correct one 
using pip install as normally done.

## Additional Packages

For the tf_mlp you are required to install either the tensorflow-1.4.1 package for CPU users or the tensorflow_gpu-1.4.1 for GPU users. Both of these can easily be installed via pip using:

```
pip install tensorflow
```

or 

```
pip install tensorflow_gpu
```
