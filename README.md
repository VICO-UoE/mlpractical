# mlpractical
## Machine Learning Practical (INFR11119)

**Note:** At this point, you can go straight to 00_Introduction notebook - which contains more information.

To run the notebooks (and later the code you are going to write within this course)
you are expected to have installed the following packages:

<ul>
<li>python 2.7+</li>
<li>numpy (anything above 1.6, 1.9+ recommended, optimally compiled with some BLAS library [MKL, OpenBLAS, ATLAS, etc.)</li>
<li>scipy (optional, but may be useful to do some tests)</li>
<li>matplotlib (for plotting)</li>
<li>ipython (v3.0+, 4.0 recommended)</li>
<li>notebook (notebooks are in version 4.0)</li>
</ul>

You can install them straight away on your personal computer, 
there is also a notebook tutorial (00_Introduction) on how to
do this (particularly) on DICE, and what configuration you 
are expected to have installed. For now, it suffices if you 
get the software working on your personal computers so you can
 start ipython notebook server and open the inital introductory 
tutorial (which will be made publicitly available next Monday).

### Installing the software on personal computers

#### On Windows: 

Download and install the Anaconda package 
(https://store.continuum.io/cshop/anaconda/)

#### On Mac (use macports): 

<ul>
<li>Install macports following instructions at https://www.macports.org/install.php</li>
<li>Install the relevant python packages in macports
<ul>
<li>  sudo port install py27-scipy +openblas </li>
<li>  sudo port install py27-ipython +notebook </li>
<li>  sudo port install py27-notebook </li>
<li>  sudo port install py27-matplotlib </li>
<li>  sudo port select --set python python27 </li>
<li>  sudo port select --set ipython2 py27-ipython </li>
<li>  sudo port select --set ipython py27-ipython </li>
</ul>
</ul>

Also, make sure that your $PATH has /opt/local/bin before /usr/bin 
so you pick up the version of python you just installed

#### On DICE (we will do this during the first lab)

### Getting the mlpractical repository

Assuming ~/mlpractical is a target workspace you want to use during
this course (where ~ denotes your home path, i.e. /home/user1). 
To start, open the terminal and clone the github mlpractical 
repository to your local disk:

git clone https://github.com/CSTR-Edinburgh/mlpractical.git

(Note: you can do it from your git account if you have one as the
above just clone the repo as anonymous user, though it does not 
matter at this point, as you are not required to submit pull requests, but you are **welcomed** to do so if you think some aspects of the notebooks can be improved!)

Naviagate to the checked out directory (cd ~/mlpractical) and type:

ipython notebook

This should start ipython notebook server and open the browser with the page
listing files/subdirs in the current directory.

To update the repository (for example, on Monday), 
enter ~/mlpractical and type git pull.




