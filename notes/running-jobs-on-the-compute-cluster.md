# Running jobs on the compute cluster

There is a [compute cluster](http://computing.help.inf.ed.ac.uk/msc-teaching-cluster) now available for submitting jobs to for your MLP assignments. Jobs on the cluster are scheduled using an open source variant of [Oracle Grid Engine](https://en.wikipedia.org/wiki/Oracle_Grid_Engine) which manages allocation of resources to jobs.

## Accessing the cluster

Jobs are submitted to a cluster and managed via one or more *head nodes*. The *head nodes* interface with a cluster of *compute nodes* where the actual jobs are run. The *head nodes* of the cluster you will be using have the aliases `msccluster` and `msccluster1` - from within the `inf.ed.ac.uk` domain you can log in to the head node by running

```
ssh [username]@msccluster
```

or 

```
ssh [username]@msccluster1
```

were `[username]` is your DICE username (student number).

## Cluster file system

The cluster has a separate user file system from your normal AFS homespace on DICE. Each user has a home directory they can read and write from under `/home/[username]` which will be your current working directory by default when you log in.

When you are logged on to the head node you can access your AFS homespace under the appropriate directory in `/afs/inf.ed.ac.uk/user/` (you can check the appropriate path by running `pwd` from your home directory when logged in to a DICE computer). This allows you to for example copy files between your homespace on the cluster and your AFS homespace (e.g. copy a script to run from AFS to the cluster or copy output files from a job run on the cluster to AFS for visualising on a DICE computer).

**Jobs running on the cluster cannot access your AFS homespace however**. This means you cannot run scripts in the Python `mlp` Conda environment you created in the first lab last semester and you also cannot write any outputs from a job directly to your AFS home directory.

### Miniconda install

To deal with the first issue, a shared read-only `miniconda2` install has been made available on each of the nodes at `/disk/scratch/mlp/miniconda2/`. The root environment in this `miniconda2` install has been set up with the required versions of all the libraries needed for training TensorFlow models on the cluster. In particular the environment has NumPy 1.11.3, SciPy 0.18.1, Matplotlib 2.0.0, TensorFlow 0.12.1 installed and the current version of the `mlp` module from Github. 

The Conda environment *does not have any of the Jupyter modules installed*. This is intentional as you should submit `.py` Python script files to run jobs on the cluster rather than running jobs interactively using the Jupyter notebook interface. You can export all of the code in a Jupyter notebook to a `.py` file by selecting `File > Download as > Python (.py)` from the main menu in the notebook interface.

### Data files

The data files for the course (i.e. all of the files available at `/afs/inf.ed.ac.uk/group/teaching/mlp/data`) are also available on all of the nodes under `/disk/scratch/mlp/data`. You will need to make sure the environment variable `MLP_DATA_DIR` is set to this path in any job you submit to run on one of the cluster compute nodes.

## Submitting a job

To submit a job to the cluster from the head node you need to use the `qsub` command. This has many optional arguments - we will only cover the most basic usage here. To see a full description you can view the manual page for the command by running `man qsub` or search for tutorials on line for `grid engine qsub`.

The main argument to `qsub` needs to be a script that can executed directly in a shell. One option is to create a wrapper `.sh` shell script which set ups the requisite environment variables and then executes a second Python script using the Python binary in `/disk/scratch/mlp/miniconda2/bin`. For example this could be done by creating a file `mlp-job.sh` in your home directory on the cluster file system with the following contents

```
#!/bin/sh

MLP_DATA_DIR='/disk/scratch/mlp/data'
/disk/scratch/mlp/miniconda2/bin/python [path-to-python-script]
```

where `[path-to-python-script]` is the path to the Python script you wish to submit as a job e.g. `$HOME/train-model.py`. The script can then be submitted to the cluster using

```
qsub -q cpu $HOME/mlp-job.sh
```

assuming the `mlp-job.sh` script is in your home directory on the cluster file system. The `-q` option specifies which queue list to submit the job to; for MLP you should run jobs on the `cpu` queue.

The scheduler will allocate the job to one of the CPU nodes. You can check on the status of submitted jobs using `qstat` - again `man qstat` can be used to give details of the output of this command and various optional arguments.

An alternative to creating a separate bash script file to run the job is to make your Python script directly executable by adding an appropriate [`shebang`](https://en.wikipedia.org/wiki/Shebang_(Unix)) as the first line in the script. The shebang indicates which interpreter to use to run a script file. If the following line is added to the top of a Python script

```
#!/disk/scratch/mlp/miniconda2/bin/python
```

then if the script is directly executed in a shell it will be run using the Python binary at `/disk/scratch/mlp/miniconda2/bin/python` (i.e. using the `miniconda2` Python install on the cluster nodes rather than the default system Python binary).

The resulting script can then be submitted to the cluster by running

```
qsub -q cpu -v MLP_DATA_DIR='/disk/scratch/mlp/data' [path-to-python-script]
```

where here `[path-to-python-script]` is the path to the Python script *with shebang line* you wish to run. The optional `-v` argument to `qsub` here is used to set an environment variable `MLP_DATA_DIR` on the compute node the job is run on that will be accessible to the Python script.

## Example: Training a MNIST model on cluster

To give you an example of how you might structure a Python script for running a job on the cluster we have added a [`example-tf-mnist-train-job.py`](../scripts/example-tf-mnist-train-job.py) script file to the Github repository under the `scripts` subdirectory.

### Saving model output

To enable you to analyse any model you train on the cluster, you will probably want to save the model state during training to allow you to restore the model for example in Jupyter notebook running on a DICE or personal computer. You could optionally have your Python script also do all the model analysis and just the save the numeric results (e.g. final training / validation set performance) and any generated plot outputs (e.g. training curves) to the cluster file system while the job is running, however even in this case it will usually be worthwhile when running longer jobs to checkpoint your model state during training. This allows you to restore from the last saved state in runs which you manually abort or error out due to an exception or job timeout.

The easiest option for saving a model state is to use the in-built [`Saver`](https://www.tensorflow.org/api_docs/python/state_ops/saving_and_restoring_variables#Saver) class in TensorFlow, which allows the values of variables which define a model's state to be checkpointed to a file on disk during training. The `example-tf-mnist-train-job.py` script gives an example of setting up a `Saver` instance and using it to checkpoint the model after every training epoch.

The example script also uses the `SummaryWriter` class described in the [`08_Introduction_to_TensorFlow`](../notebooks/08_Introduction_to_TensorFlow.ipynb) notebook to log summaries of the training and validation set accuracy and error values to log files which can be loaded in TensorBoard to visualise training runs. The script also gives an example of manually accumulating these statistics into NumPy arrays and saving these to a `.npz` file which may be useful if you wish to create plots from these values using Matplotlib.

The script writes all outputted files (model checkpoints, train and validation summaries, training run statistics `.npz` file) to a timestamped subdirectory of a path on the cluster filesystem specified by a `OUTPUT_DIR` environment variable. You will need to make sure this environment variable is set on the cluster node before the script is run e.g. by adding it to a wrapper `.sh` script used to submit the job, or using the `-v` argument of `qsub`. For example you may wish to set `OUTPUT_DIR=$HOME/experiments` to have all outputs written to an `experiments` subdirectory in your cluster home space.

### Walk through of submitting example script to cluster

The complete series of commands you would need to run in a DICE terminal to submit the example script as a job on the compute cluster are as follows:

  1. Log in to the head node by running  
     ```
     ssh [username]@msccluster
     ```
  2. Download the script file from Github to your cluster homespace  
     ```
     wget  https://raw.githubusercontent.com/CSTR-Edinburgh/mlpractical/mlp2016-7/master/scripts/example-tf-mnist-train-job.py
     ```
  3. Create an `experiments` directory  
     ```
     mkdir experiments
     ```
  4. Submit the job to the cluster  
     ```
     qsub -q cpu -v MLP_DATA_DIR='/disk/scratch/mlp/data',OUTPUT_DIR='$HOME/experiments' example-tf-mnist-train-job.py
     ``` 
     
If the job is successfully submitted you should see a message

```
Your job [job-id] ("example-tf-mnist-train-job.py") has been submitted
```

printed to the terminal, where `[job-id]` is an integer ID which identifies the job to the scheduler (this can be used for example to delete one of your running jobs using `qdel [job-id]`). You can use `qstat` to view the status of all of your currently submitted jobs. Typically straight after a job is submitted it will show its state as `qw` which means the job is waiting in the queue to be run. Once the job is in progress on one of the nodes this will change to `r` which indicates job is runnning. 

An `E` in the job state indicates there has been an error. Running `qstat -j [job-id] | grep error` may help diagnose the issue. You should also `qdel [job-id]` after you have read the error message if any.

By default the stdout and sterr output from your script will be written to files `example-tf-mnist-train-job.py.o[job-id]` and `example-tf-mnist-train-job.py.e[job-id]` respectively in your cluster home directory while the job is running (this default behaviour can be changed using the optional `-o` and `-e` options in `qsub`). If you display the contents of the `example-tf-mnist-train-job.py.o[job-id]` file by running e.g.

```
cat example-tf-mnist-train-job.py.o[job-id]
```

you should see a series of messages printing the training and validation performance over training (i.e. the output of the `print` statements in the example script).

### Restoring a model checkpoint

If you wished to load the final checkpoint of the trained model in to Jupyter notebook running on a DICE machine, you should first copy across the relevant experiment output directory to your AFS home space e.g.

```
cp experiments/2017-02-10_12-30-00 /afs/inf.ed.ac.uk/user/s12/s123456/experiments
```

(replacing the timestamp and directories appropriately) and then run the following in a Python interpreter / Jupyter notebook cell *on a DICE computer* (i.e. not on your session logged in to the cluster head node)

```python
import os
import tensorflow as tf

ckpt_dir = os.path.join(
    os.environ['HOME'], 'experiments', '2017-02-10_12-30-00', 'checkpoints')
sess = tf.Session()
saver = tf.train.import_meta_graph(
    os.path.join(ckpt_dir, 'model.ckpt-5000.meta'))
saver.restore(sess, os.path.join(ckpt_dir, 'model.ckpt-5000'))
```

again replacing the example timestamp with the appropriate value.

The TensorFlow session `sess` will then contain a restored version of the checkpointed graph and the associated states of the variables (e.g. model parameters) in the graph at the point the model was checkpointed in training.
