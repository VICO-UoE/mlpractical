
# How to fix notebook's "kernel issues" on DICE

Some of the people in mlpractical have been affected by a recent update to the numpy and numercial 
library pushed to DICE last week. The symptom was restarting notebook kernel when someone
tried to run the exercise involving numpy usage.

The reason of this most likely affected people who either 1) ended up with 
default atlas libraries (which has been updated in the meantime) or 2) re-compiled 
numpy with new DICE OpenBLAS already available, but LD_LIBRARY_PATH pointed to the 
version compiled last time - which could introduce some unexepcted behaviours.

## Fix

Follow the below setps **before** you activate the old virtual environment. The fix 
basically involves rebuilding the virtual environments. But the whole process is now
much simpler (due to the fact OpenBLAS is now a deafult numerical library on DICE).

1) Comment out (or remove) `export=$LD_LIBRARY_PATH...` line in your ~/.bashrc script. Then type 
`unset LD_LIBRARY_PATH` in the terminal. To make sure this variable is not
set, type `export` and check visually in the printed list of variables

2) Go to `~/mlpractical/repos-3rd/virtualenv` and install the new virtual
environment by typing: 

```
./virtualenv.py --python /usr/bin/python2.7 --no-site-packages $MLP_WDIR/venv2
```

3) Activate it by typing: source $MLP_WDIR/venv2/bin/activate and install the usual for the course packages using pip:

   * pip install pip --upgrade
   * pip install numpy
   * pip install ipython
   * pip install notebook
   * pip install matplotlib

4) Now enter `~/mlpractical/repo-mlp` and see whether numpy has been
linked to DICE-standard OpenBLAS (and works) by starting python notebook:
```
ipython notebook
```
and running two first interactive examples from 00_Introduction.py. 
If they run, you can simply modify `activate_mlp` alias in `./bashrc`to point to
`venv2` instead of `venv`

5) You can also remove both old `venv` amd other not needed anymore
directories with numpy and OpenBLAS sources in `~/mlpractical/repos-3rd` directory.