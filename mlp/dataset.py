
# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import cPickle
import gzip
import numpy
import os


class DataProvider(object):
    """
    Data provider defines an interface for our
    generic data-independent readers.
    """
    def __init__(self, batch_size, randomize=True):
        """
        :param batch_size: int, specifies the number
               of elements returned at each step
        :param randomize: bool, shuffles examples prior
               to iteration, so they are presented in random
               order for stochastic gradient descent training
        :return:
        """
        self.batch_size = batch_size
        self.randomize = randomize
        self._curr_idx = 0

    def reset(self):
        """
        Resets the provider to the initial state to
        use in another epoch
        :return: None
        """
        self._curr_idx = 0

    def __randomize(self):
        """
        Data-specific implementation of shuffling mechanism
        :return:
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        This method says an object is iterable.
        """
        return self

    def next(self):
        """
        Data-specific iteration mechanism. Called each step 
        (i.e. each iteration in a loop)
        unitl StopIteration() exception is raised.
        :return:
        """
        raise NotImplementedError()


class MNISTDataProvider(DataProvider):
    """
    The class iterates over MNIST digits dataset, in possibly
    random order.
    """
    def __init__(self, dset,
                 batch_size=10,
                 max_num_batches=-1,
                 randomize=True):

        super(MNISTDataProvider, self).\
            __init__(batch_size, randomize)

        assert dset in ['train', 'valid', 'eval'], (
            "Expected dset to be either 'train', "
            "'valid' or 'eval' got %s" % dset
        )
        
        assert max_num_batches != 0, (
            "max_num_batches should be != 0"
        )

        dset_path = './data/mnist_%s.pkl.gz' % dset
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        with gzip.open(dset_path) as f:
            x, t = cPickle.load(f)

        self._max_num_batches = max_num_batches
        self.x = x
        self.t = t
        self.num_classes = 10

        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MNISTDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        return numpy.random.permutation(numpy.arange(0, self.x.shape[0]))

    def next(self):
        
        has_enough = (self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches < (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        rval_x = self.x[range_idx]
        rval_t = self.t[range_idx]

        self._curr_idx += self.batch_size

        return rval_x, self.__to_one_of_k(rval_t)

    def __to_one_of_k(self, y):
        rval = numpy.zeros((y.shape[0], self.num_classes), dtype=numpy.float32)
        for i in xrange(y.shape[0]):
            rval[i, y[i]] = 1
        return rval


class MetOfficeDataProvider(DataProvider):
    """
    The class iterates over South Scotland Weather, in possibly
    random order.
    """
    def __init__(self, window_size,
                 batch_size=10,
                 max_num_batches=-1,
                 randomize=True):

        super(MetOfficeDataProvider, self).\
            __init__(batch_size, randomize)

        dset_path = './data/HadSSP_daily_qc.txt'
        assert os.path.isfile(dset_path), (
            "File %s was expected to exist!." % dset_path
        )

        raw = numpy.loadtxt(dset_path, skiprows=3, usecols=range(2, 32))
        
        self.window_size = window_size
        self._max_num_batches = max_num_batches
        #filter out all missing datapoints and
        #flatten a matrix to a vector, so we will get
        #a time preserving representation of measurments
        #with self.x[0] being the first day and self.x[-1] the last
        self.x = raw[raw >= 0].flatten()
        
        #normalise data to zero mean, unit variance
        mean = numpy.mean(self.x)
        var = numpy.var(self.x)
        assert var >= 0.01, (
            "Variance too small %f " % var
        )
        self.x = (self.x-mean)/var
        
        self._rand_idx = None
        if self.randomize:
            self._rand_idx = self.__randomize()

    def reset(self):
        super(MetOfficeDataProvider, self).reset()
        if self.randomize:
            self._rand_idx = self.__randomize()

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        # we generate random indexes starting from window_size, i.e. 10th absolute element
        # in the self.x vector, as we later during mini-batch preparation slice
        # the self.x container backwards, i.e. given we want to get a training 
        # data-point for 11th day, we look at 10 preeceding days. 
        # Note, we cannot do this, for example, for the 5th day as
        # we do not have enough observations to make an input (10 days) to the model
        return numpy.random.permutation(numpy.arange(self.window_size, self.x.shape[0]))

    def next(self):
        
        has_enough = (self._curr_idx + self.batch_size) <= self.x.shape[0]
        presented_max = (0 < self._max_num_batches < (self._curr_idx / self.batch_size))

        if not has_enough or presented_max:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = \
                self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = \
                numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        #build slicing matrix of size minibatch, which will contain batch_size
        #rows, each keeping indexes that selects windows_size+1 [for (x,t)] elements
        #from data vector (self.x) that itself stays always sorted w.r.t time
        range_slices = numpy.zeros((self.batch_size, self.window_size + 1), dtype=numpy.int32)
       
        for i in xrange(0, self.batch_size):
            range_slices[i, :] = \
                numpy.arange(range_idx[i], 
                             range_idx[i] - self.window_size - 1, 
                             -1,
                             dtype=numpy.int32)[::-1]
        
        #here we use advanced indexing to select slices from observation vector
        #last column of rval_x makes our targets t (as we splice window_size + 1
        tmp_x = self.x[range_slices]
        rval_x = tmp_x[:,:-1]
        rval_t = tmp_x[:,-1].reshape(self.batch_size, -1)
        
        self._curr_idx += self.batch_size

        return rval_x, rval_t

    
class FuncDataProvider(DataProvider):
    """
    Function gets as an argument a list of functions defining the means
    of a normal distribution to sample from.
    """
    def __init__(self,
                 fn_list=[lambda x: x ** 2, lambda x: numpy.sin(x)],
                 std_list=[0.1, 0.1],
                 x_from = 0.0,
                 x_to = 1.0,
                 points_per_fn=200,
                 batch_size=10,
                 randomize=True):
        """
        """

        super(FuncDataProvider, self).__init__(batch_size, randomize)

        def sample_points(y, std):
            ys = numpy.zeros_like(y)
            for i in xrange(y.shape[0]):
                ys[i] = numpy.random.normal(y[i], std)
            return ys

        x = numpy.linspace(x_from, x_to, points_per_fn, dtype=numpy.float32)
        means = [fn(x) for fn in fn_list]
        y = [sample_points(mean, std) for mean, std in zip(means, std_list)]

        self.x_orig = x
        self.y_class = y

        self.x = numpy.concatenate([x for ys in y])
        self.y = numpy.concatenate([ys for ys in y])

        if self.randomize:
            self._rand_idx = self.__randomize()
        else:
            self._rand_idx = None

    def __randomize(self):
        assert isinstance(self.x, numpy.ndarray)
        return numpy.random.permutation(numpy.arange(0, self.x.shape[0]))

    def __iter__(self):
        return self

    def next(self):
        if (self._curr_idx + self.batch_size) >= self.x.shape[0]:
            raise StopIteration()

        if self._rand_idx is not None:
            range_idx = self._rand_idx[self._curr_idx:self._curr_idx + self.batch_size]
        else:
            range_idx = numpy.arange(self._curr_idx, self._curr_idx + self.batch_size)

        x = self.x[range_idx]
        y = self.y[range_idx]

        self._curr_idx += self.batch_size

        return x, y

