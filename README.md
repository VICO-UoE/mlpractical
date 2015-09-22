# mlpractical
## Machine Learning Practical (INFR11119)

To run the notebooks (and later the code you are going to write within this course)
you are expected to have installed the following packages:

python 2.7+
numpy (anything above 1.6, 1.9+ recommended, optimally
       compiled with some BLAS library [MKL, OpenBLAS, ATLAS, etc.)
scipy (optional, but may be useful to do some tests)
matplotlib (for plotting)
ipython (v3.0+, 4.0 recommended)
notebook (notebooks are in version 4.0)

You can install them straight away on your personal computer, 
there is also a notebook tutorial (00_Introduction) on how to
do this on DICE, and what configuration you are expected to have. 
For now, it suffices if you get the software working on your 
personal computers so you can start ipython notebook server 
and open the inital introductory tutorial (which will be make
publicitly available next Monday).

I) Installing the software on personal computers

a) On Windows: download and install the Anaconda package 
   (https://store.continuum.io/cshop/anaconda/)

b) On Mac (use macports): 

Install macports following instructions at https://www.macports.org/install.php
Install the relevant python packages in macports
sudo port install py27-scipy +openblas
sudo port install py27-ipython +notebook
sudo port install py27-notebook
sudo port install py27-matplotlib
sudo port select --set python python27
sudo port select --set ipython2 py27-ipython
sudo port select --set ipython py27-ipython

Also, make sure that your $PATH has /opt/local/bin before /usr/bin 
so you pick up the version of python you just installed

c) On DICE (we will do this during the first lab)

II) Setting up the repository

Assuming ~/mlpractical is a target workspace you want to use during
this course (where ~ denotes your home path, i.e. /home/user1). 
To start, open the terminal and clone the github mlpractical 
repository to your local disk:

git clone https://github.com/CSTR-Edinburgh/mlpractical.git

(Note: you can do it from your git account if you have one as the
above just clone the repo as anonymous user, though it does not 
matter at this point, as you will not submit pull requests)

Naviagate to the checked out directory by typing cd ~/mlpractical and type:

ipython notebook

This should start notebook server and open the browser with the page
listing files/subdirs in the current directory.

To update the repository (for example, on Monday), enter ~/mlpractical
and type git pull.




