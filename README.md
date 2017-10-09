# Machine Learning Practical

This repository contains the code for the University of Edinburgh [School of Informatics](http://www.inf.ed.ac.uk) course [Machine Learning Practical](http://www.inf.ed.ac.uk/teaching/courses/mlp/).

This assignment-based course is focused on the implementation and evaluation of machine learning systems. Students who do this course will have experience in the design, implementation, training, and evaluation of machine learning systems.

The code in this repository is split into:

  *  a Python package `mlp`, a [NumPy](http://www.numpy.org/) based neural network package designed specifically for the course that students will implement parts of and extend during the course labs and assignments,
  *  a series of [Jupyter](http://jupyter.org/) notebooks in the `notebooks` directory containing explanatory material and coding exercises to be completed during the course labs.

## Getting set up

Detailed instructions for setting up a development environment for the course are given in [this file](notes/environment-set-up.md). Students doing the course will spend part of the first lab getting their own environment set up.

## Frequent Issues/Solutions

Don’t forget that from your /mlpractica/l folder you should first do 
```
git status #to check whether there are any changes in your local branch. If there are, you need to do: 
git add “path /to/file”
git commit -m “some message”
```

Only if this is OK, you can run 
```
git checkout mlp2017-8/lab[n]
```
Related to MLP module not found error:
Another thing is to make sure you have you MLP_DATA_DIR path correctly set. You can check this by typing 
```echo $MLP_DATA_DIR```
in the command line. If this is not set up, you need to follow the instructions on the set-up-environment to get going. 

Finally, please make sure you have run 
```python setup.py develop```
