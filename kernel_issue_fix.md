
# How to fix notebook's "kernel issues" on DICE

Some people in MLP have been affected by a recent update to the `numpy` and `numerical` 
libraries on DICE on 3 October.  The problem affects you if you get a message stating that the kernel was restarted when you run code involving `numpy`.

If you have experienced these issues you have either:
1. ended up using the default `atlas` libraries with `numpy` (which have been updated in the meantime) 
2. or re-compiled `numpy` with the new DICE `OpenBLAS` that is available, but the `LD_LIBRARY_PATH` that you set during the first lab last week gave priority to load the `OpenBLAS` libraries compiled last time - which could introduce some unexepcted behaviour at runtime.

## The Fix

Follow the below steps **before** you activate the old virtual environment (or deactivate it if it is activated). The fix basically involves rebuilding the virtual environments. But the whole process is now much simpler due to the fact `OpenBLAS` is now a default numerical library on DICE.

1.	Comment out (or remove) the `export=$LD_LIBRARY_PATH...` line in your ~/.bashrc file. Then type
	```
	unset LD_LIBRARY_PATH
	``` 
	in the terminal. To make sure this variable is not set, type `export` and check visually in the printed list of variables

2.	Go to `~/mlpractical/repos-3rd/virtualenv` and install the new virtual environment (`venv2`) by typing: 
	```
	./virtualenv.py --python /usr/bin/python2.7 --no-site-packages $MLP_WDIR/venv2
	```

3.	Activate your new virtual environment by typing: 
	```
	source $MLP_WDIR/venv2/bin/activate 
	```
	and install the usual packages required by MLP using pip:
	```
	pip install pip --upgrade
	pip install numpy
	pip install ipython
	pip install notebook
	pip install matplotlib
	```

4.	Change directory to `~/mlpractical/repo-mlp` and check that `numpy` is linked to the DICE-standard `OpenBLAS` (and works) by starting ipython notebook:
	```
	ipython notebook
	```
	then run the first two interactive examples from `00_Introduction.py.`   If they run, then you can simply modify the `activate_mlp` alias in `./bashrc` to point to `venv2` instead of `venv`.

5.	You can also remove both the old `venv` and the other unrequired directories that contain `numpy` and `OpenBLAS` sources in the `~/mlpractical/repos-3rd` directory.


