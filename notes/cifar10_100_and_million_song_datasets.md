## Datasets Available on AFS

For your convinience we provided data providers for cifar10/100 and million song dataset. Below you can find 
information on the datasets and the AFS paths where one can find them.

## CIFAR-10 and CIFAR-100 datasets

[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) are a pair of image classification datasets collected by collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. They are labelled subsets of the much larger [80 million tiny images](dataset). They are a common benchmark task for image classification - a list of current accuracy benchmarks for both data sets are maintained by Rodrigo Benenson [here](http://rodrigob.github.io/are_we_there_yet/build/).

As the name suggests, CIFAR-10 has images in 10 classes:

    airplane
    automobile
    bird 
    cat
    deer
    dog
    frog
    horse
    ship
    truck

with 6000 images per class for an overall dataset size of 60000. Each image has three (RGB) colour channels and pixel dimension 32×32, corresponding to a total dimension per input image of 3×32×32=3072. For each colour channel the input values have been normalised to the range [0, 1].

CIFAR-100 has images of identical dimensions to CIFAR-10 but rather than 10 classes they are instead split across 100 fine-grained classes (and 20 coarser 'superclasses' comprising multiple finer classes):

<table style='border: none;'>
    <tbody><tr style='font-weight: bold;'>
        <td>Superclass</td>
        <td>Classes</td>
    </tr>
    <tr>
        <td>aquatic mammals</td>
        <td>beaver, dolphin, otter, seal, whale</td>
    </tr>
    <tr>
        <td>fish</td>
        <td>aquarium fish, flatfish, ray, shark, trout</td>
    </tr>
    <tr>
        <td>flowers</td>
        <td>orchids, poppies, roses, sunflowers, tulips</td>
    </tr>
    <tr>
        <td>food containers</td>
        <td>bottles, bowls, cans, cups, plates</td>
    </tr>
    <tr>
        <td>fruit and vegetables</td>
        <td>apples, mushrooms, oranges, pears, sweet peppers</td>
    </tr>
    <tr>
        <td>household electrical devices</td>
        <td>clock, computer keyboard, lamp, telephone, television</td>
    </tr>
    <tr>
        <td>household furniture</td>
        <td>bed, chair, couch, table, wardrobe</td>
    </tr>
    <tr>
        <td>insects</td>
        <td>bee, beetle, butterfly, caterpillar, cockroach</td>
    </tr>
    <tr>
        <td>large carnivores</td>
        <td>bear, leopard, lion, tiger, wolf</td>
    </tr>
    <tr>
        <td>large man-made outdoor things</td>
        <td>bridge, castle, house, road, skyscraper</td>
    </tr>
    <tr>
        <td>large natural outdoor scenes</td>
        <td>cloud, forest, mountain, plain, sea</td>
    </tr>
    <tr>
        <td>large omnivores and herbivores</td>
        <td>camel, cattle, chimpanzee, elephant, kangaroo</td>
    </tr>
    <tr>
        <td>medium-sized mammals</td>
        <td>fox, porcupine, possum, raccoon, skunk</td>
    </tr>
    <tr>
        <td>non-insect invertebrates</td>
        <td>crab, lobster, snail, spider, worm</td>
    </tr>
    <tr>
        <td>people</td>
        <td>baby, boy, girl, man, woman</td>
    </tr>
    <tr>
        <td>reptiles</td>
        <td>crocodile, dinosaur, lizard, snake, turtle</td>
    </tr>
    <tr>
        <td>small mammals</td>
        <td>hamster, mouse, rabbit, shrew, squirrel</td>
    </tr>
    <tr>
        <td>trees</td>
        <td>maple, oak, palm, pine, willow</td>
    </tr>
    <tr>
        <td>vehicles 1</td>
        <td>bicycle, bus, motorcycle, pickup truck, train</td>
    </tr>
    <tr>
        <td>vehicles 2</td>
        <td>lawn-mower, rocket, streetcar, tank, tractor</td>
    </tr>
</tbody></table>

Each class has 600 examples in it, giving an overall dataset size of 60000 i.e. the same as CIFAR-10.

Both CIFAR-10 and CIFAR-100 have standard splits into 50000 training examples and 10000 test examples. For CIFAR-100 as there is an optional Kaggle competition (see below) scored on predictions on the test set, we have used a non-standard assignation of examples to test and training set and only provided the inputs (and not target labels) for the 10000 examples chosen for the test set. 

For CIFAR-10 the 10000 test set examples have labels provided: to avoid any accidental over-fitting to the test set **you should only use these for the final evaluation of your model(s)**. If you repeatedly evaluate models on the test set during model development it is easy to end up indirectly fitting to the test labels - for those who have not already read it see this [excellent cautionary note from the MLPR notes by Iain Murray](http://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w2a_train_test_val.html#warning-dont-fool-yourself-or-make-a-fool-of-yourself). 

For both CIFAR-10 and CIFAR-100, the remaining 50000 non-test examples have been split in to a 40000 example training dataset and a 10000 example validation dataset, each with target labels provided. If you wish to use a more complex cross-fold validation scheme you may want to combine these two portions of the dataset and define your own functions for separating out a validation set.

Data provider classes for both CIFAR-10 and CIFAR-100 are available in the `mlp.data_providers` module. Both have similar behaviour to the `MNISTDataProvider` used extensively last semester. A `which_set` argument can be used to specify whether to return a data provided for the training dataset (`which_set='train'`) or validation dataset (`which_set='valid'`).

The CIFAR-100 data provider also takes an optional `use_coarse_targets` argument in its constructor. By default this is set to `False` and the targets returned by the data provider correspond to 1-of-K encoded binary vectors for the 100 fine-grained object classes. If `use_coarse_targets=True` then instead the data provider will return 1-of-K encoded binary vector targets for the 20 coarse-grained superclasses associated with each input instead.

Both data provider classes provide a `label_map` attribute which is a list of strings which are the class labels corresponding to the integer targets (i.e. prior to conversion to a 1-of-K encoded binary vector).

### Accessing the CIFAR-10 and CIFAR-100 data

Before using the data provider objects you will need to make sure the data files are accessible to the `mlp` package by existing under the directory specified by the `MLP_DATA_DIR` path.

The data is available as compressed NumPy `.npz` files in the AFS directory `/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/`.

If you are working on DICE one option is to redefine your `MLP_DATA_DIR` to directly point to the shared AFS data directory by editing the `env_vars.sh` start up file for your environment. This will avoid using up your DICE quota by storing the data files in your homespace but may involve slower initial loading of the data on initialising the data providers if many people are trying access the same files at once. The environment variable can be redefined by running

```
gedit ~/miniconda3/envs/mlp/etc/conda/activate.d/env_vars.sh
```

in a terminal window (assuming you installed `miniconda3` to your home directory), and changing the line

```
export MLP_DATA_DIR=$HOME/mlpractical/data
```

to

```
export MLP_DATA_DIR="`/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/`"
```

and then saving and closing the editor. You will need reload the `mlp` environment using `source activate mlp` and restart the Jupyter notebook server in the reloaded environment for the new environment variable definition to be available.

For those working on DICE who have sufficient quota remaining or those using there own machine, an alternative option is to copy the data files in to your local `mlp/data` directory (or wherever your `MLP_DATA_DIR` environment variable currently points to if different). 


Assuming your local `mlpractical` repository is in your home directory you should be able to copy the required files on DICE by running

```
cp `/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/cifar*.npz ~/mlpractical/data
```

On a non-DICE machine, you will need to either [set up local access to AFS](http://computing.help.inf.ed.ac.uk/informatics-filesystem), use a remote file transfer client like `scp` or you can alternatively download the files using the iFile web interface [here](https://ifile.inf.ed.ac.uk/?path=%2Fafs%2Finf.ed.ac.uk%2Fgroup%2Fteaching%2Fmlp%2Fdata&goChange=Go) (requires DICE credentials).

As some of the files are quite large you may wish to copy only those you are using currently (e.g. only the files for one of the two tasks) to your local filespace to avoid filling up your quota. The `cifar-100-test-inputs.npz` file will only be needed by those intending to enter the associated optional Kaggle competition.

## Genre classification with the Million Song Dataset

The [Million Song Dataset](http://labrosa.ee.columbia.edu/millionsong/) is a 

>  freely-available collection of audio features and metadata for a million contemporary popular music tracks

originally collected and compiled by Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere.

The dataset is intended to encourage development of algorithms in the field of [music information retrieval](https://en.wikipedia.org/wiki/Music_information_retrieval). The [data for each track](http://labrosa.ee.columbia.edu/millionsong/pages/example-track-description) includes both textual features such as artist and album names, numerical descriptors such as duration and various audio features derived using a music analysis platform provided by [The Echo Nest](https://en.wikipedia.org/wiki/The_Echo_Nest) (since acquired by Spotify). Of the various audio features and segmentations included in the full dataset, the most detailed information is included at a 'segment' level: each segment corresponds to an automatically identified 'quasi-stable music event' - roughly contiguous sections of the audio with similar perceptual quality. The number of segments per track is variable and each segment can itself be of variable length - typically they seem to be around 0.2 - 0.4 seconds but can be as long as 10 seconds or more. 

For each segment of the track various extracted audio features are available - a 12 dimensional vector of [chroma features](https://en.wikipedia.org/wiki/Chroma_feature), a 12 dimensional vector of ['MFCC-like'](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) timbre features and various measures of the loudness of the segment, including loudness at the segment start and maximum loudness. In the version of the data we provide, we include a 25 dimensional vector for each included segment, consisting of the 12 timbre features, 12 chroma features and loudness at start of segment concatenated in that order. To allow easier integration in to standard feedforward models, the basic version of the data we provide includes features only for a fixed length crop of the central 120 segments of each track (with tracks with less than 120 segments therefore not being included). This gives an overall input dimension per track of 120×25=3000. Each of the 3000 input dimensions has been been preprocessed by subtracting the per-dimension mean across the training data and dividing by the per-dimension standard deviation across the training data.

We provide data providers for the fixed length crops versions of the input features, with the inputs being returned in batches of 3000 dimensional vectors (these can be reshaped to (120, 25) to get the per-segment features). To allow for more complex variable-length sequence modelling with for example recurrent neural networks, we also provide a variable length version of the data. This is only provided as compressed NumPy (`.npz`) data files rather than data provider objects - you will need to write your own data provider if you wish to use this version of the data. As the inputs are of variable number of segments they have been ['bucketed'](https://www.tensorflow.org/tutorials/seq2seq/#bucketing_and_padding) into groups of similar maximum length, with the following binning scheme used:

     120 - 250  segments
     251 - 500  segments
     501 - 650  segments
     651 - 800  segments
     801 - 950  segments
     951 - 1200 segments
    1201 - 2000 segments
    2000 - 4000 segments
    
For each bucket the NumPy data files include inputs and targets arrays with second dimension equal to the maximum sgement size in the bucket (e.g. 250 for the bucket) and first dimension equal to the number of tracks with number of segments in that bucket. These are named `inputs_{n}` and `targets_{n}` in the data file where `{n}` is the maximal number of segments in the bucket e.g. `inputs_250` and `targets_250` for the first bucket. For tracks with less segments than the maximum size in the bucket, the features for the track have been padded with `NaN` values. For tracks with more segments than the maximum bucket size of 4000, only the first 4000 segments have been included.

To allow you to match tracks between the fixed length and variable length datasets, the data files also include an array for each bucket giving the indices of the corresponding track in the fixed length input arrays. For example the array `indices_250` will be an array of the same size as the first dimension of `inputs_250` and `targets_250` with the first element of `indices_250` giving the index into the `inputs` and `targets` array of the fixed length data corresponding to first element of `inputs_250` and `targets_250`.

The Million Song Dataset in its original form does not provide any genre labels, however various external groups have proposed genre labels for portions of the data by cross-referencing the track IDs against external music tagging databases. Analagously to the provision of both simpler and more complex classifications tasks for the CIFAR-10 / CIFAR-100 datasets, we provide two classification task datasets derived from the Million Song Dataset - one with 10 coarser level genre classes, and another with 25 finer-grained genre / style classifications.

The 10-genre classification task uses the [*CD2C tagtraum genre annotations*](http://www.tagtraum.com/msd_genre_datasets.html) derived from multiple source databases (beaTunes genre dataset, Last.fm dataset, Top-MAGD dataset), with the *CD2C* variant using only non-ambiguous annotations (i.e. not including tracks with multiple genre labels). Of the 15 genre labels provided in the CD2C annotations, 5 (World, Latin, Punk, Folk and New Age) were not included due to having fewer than 5000 examples available. This left 10 remaining genre classes:

    Rap
    Rock
    RnB
    Electronic
    Metal
    Blues
    Pop
    Jazz
    Country
    Reggae

For each of these 10 classes, 5000 labelled examples have been collected for training / validation (i.e. 50000 example in total) and a further 1000 example per class for testing, with the exception of the `Blues` class for which only 991 testing examples are provided due to there being insufficient labelled tracks of the minimum required length (i.e. a total of 9991 test examples). 

The 9991 test set examples have labels provided: however to avoid any accidental over-fitting to the test set **you should only use these for the final evaluation of your model(s)**. If you repeatedly evaluate models on the test set during model development it is easy to end up indirectly fitting to the test labels - for those who have not already read it see this [excellent cautionary note int the MLPR notes by Iain Murray](http://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w2a_train_test_val.html#warning-dont-fool-yourself-or-make-a-fool-of-yourself). 


The 25-genre classification tasks uses the [*MSD Allmusic Style Dataset*](http://www.ifs.tuwien.ac.at/mir/msd/MASD.html) labels derived from the [AllMusic.com](http://www.allmusic.com/) database by [Alexander Schindler, Rudolf Mayer and Andreas Rauber of Vienna University of Technology](http://www.ifs.tuwien.ac.at/~schindler/pubs/ISMIR2012.pdf). The 25 genre / style labels used are:

    Big Band
    Blues Contemporary
    Country Traditional
    Dance
    Electronica
    Experimental
    Folk International
    Gospel
    Grunge Emo
    Hip Hop Rap
    Jazz Classic
    Metal Alternative
    Metal Death
    Metal Heavy
    Pop Contemporary
    Pop Indie
    Pop Latin
    Punk
    Reggae
    RnB Soul
    Rock Alternative
    Rock College
    Rock Contemporary
    Rock Hard
    Rock Neo Psychedelia
    
For each of these 25 classes, 2000 labelled examples have been collected for training / validation (i.e. 50000 example in total). A further 400 example per class have been collected for testing (i.e. 10000 examples in total), which you are provided inputs but not targets for. The optional Kaggle competition being run for this dataset (see email) is scored based on the 25-genre class label predictions on these unlabelled test inputs. 

The tracks used for the 25-genre classification task only partially overlap with those used for the 10-genre classification task and we do not provide any mapping between the two.

For each of the two tasks, the 50000 examples collected for training have been pre-split in to a 40000 example training dataset and a 10000 example validation dataset. If you wish to use a more complex cross-fold validation scheme you may want to combine these two portions of the dataset and define your own functions / classes for separating out a validation set.

Data provider classes for both fixed length input data for the 10 and 25 genre classification tasks in the `mlp.data_providers` module as `MSD10GenreDataProvider` and `MSD25GenreDataProvider`. Both have similar behaviour to the `MNISTDataProvider` used extensively last semester. A `which_set` argument can be used to specify whether to return a data provided for the training dataset (`which_set='train'`) or validation dataset (`which_set='valid'`).  Both data provider classes provide a `label_map` attribute which is a list of strings which are the class labels corresponding to the integer targets (i.e. prior to conversion to a 1-of-K encoded binary vector).

The test dataset files for the 10 genre classification task are provided as two separate NumPy data files `msd-10-genre-test-inputs.npz` and `msd-10-genre-test-targets.npz`. These can be loaded using [`np.load`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html) function. The inputs are stored as a $10000\times3000$ array under the key `inputs` in the file `msd-10-genre-test-inputs.npz` and the targets in a 10000 element array of integer labels under the key `targets` in `msd-10-genre-test-targets.npz`. A corresponding `msd-25-genre-test-inputs.npz` file is provided for the 25 genre task inputs.

### Accessing the Million Song Dataset data

Before using the data provider objects you will need to make sure the data files are accessible to the `mlp` package by existing under the directory specified by the `MLP_DATA_DIR` path.

The fixed length input data and associated targets is available as compressed NumPy `.npz` files in the AFS directory ``/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/``.

If you are working on DICE one option is to redefine your `MLP_DATA_DIR` to directly point to the shared AFS data directory by editing the `env_vars.sh` start up file for your environment. This will avoid using up your DICE quota by storing the data files in your homespace but may involve slower initial loading of the data on initialising the data providers if many people are trying access the same files at once. The environment variable can be redefined by running

```
gedit ~/miniconda3/envs/mlp/etc/conda/activate.d/env_vars.sh
```

in a terminal window (assuming you installed `miniconda3` to your home directory), and changing the line

```
export MLP_DATA_DIR=$HOME/mlpractical/data
```

to

```
export MLP_DATA_DIR="/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/"
```

and then saving and closing the editor. You will need reload the `mlp` environment using `source activate mlp` and restart the Jupyter notebook server in the reloaded environment for the new environment variable definition to be available.

Assuming your local `mlpractical` repository is in your home directory you should be able to copy the required files on DICE by running

```
cp `/afs/inf.ed.ac.uk/group/teaching/mlp/data/2017-18/msd*.npz ~/mlpractical/data
```

On a non-DICE machine, you will need to either [set up local access to AFS](http://computing.help.inf.ed.ac.uk/informatics-filesystem), use a remote file transfer client like `scp` or you can alternatively download the files using the iFile web interface [here](https://ifile.inf.ed.ac.uk/?path=%2Fafs%2Finf.ed.ac.uk%2Fgroup%2Fteaching%2Fmlp%2Fdata&goChange=Go) (requires DICE credentials).

As some of the files are quite large you may wish to copy only those you are using currently (e.g. only the files for one of the two tasks) to your local filespace to avoid filling up your quota. The `cifar-100-test-inputs.npz` file will only be needed by those intending to enter the associated optional Kaggle competition.
