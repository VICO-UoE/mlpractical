# -*- coding: utf-8 -*-
"""Data providers."""

import cPickle
import gzip
import numpy as np
import os
from mlp import DEFAULT_SEED


class DataProvider(object):
    """Interface for generic data-independent readers."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """
        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        assert max_num_batches != 0 and not max_num_batches < -1, (
            'max_num_batches should be -1 or > 0')
        self.max_num_batches = max_num_batches
        possible_num_batches = self.inputs.shape[0] // batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        """Resets the provider to the initial state to use in a new epoch."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def shuffle(self):
        """Shuffles order of data."""
        new_order = self.rng.permutation(self.inputs.shape[0])
        self.inputs = self.inputs[new_order]
        self.targets = self.targets[new_order]

    def next(self):
        if self._curr_batch + 1 > self.num_batches:
            self.reset()
            raise StopIteration()
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class MNISTDataProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        assert which_set in ['train', 'valid', 'eval'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'mnist_{0}.pkl.gz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        with gzip.open(data_path) as f:
            inputs, targets = cPickle.load(f)
        super(MNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        inputs_batch, targets_batch = super(MNISTDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, int_targets):
        one_of_k_targets = np.zeros((int_targets.shape[0], self.num_classes))
        one_of_k_targets[range(int_targets.shape[0]), int_targets] = 1
        return one_of_k_targets


class MetOfficeDataProvider(DataProvider):
    """South Scotland Met Office weather data provider."""

    def __init__(self, window_size, batch_size=10, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'HadSSP_daily_qc.txt')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        raw = np.loadtxt(data_path, skiprows=3, usecols=range(2, 32))
        assert window_size > 1, 'window_size must be at least 2.'
        self.window_size = window_size
        #filter out all missing datapoints and flatten to a vector
        filtered = raw[raw >= 0].flatten()
        #normalise data to zero mean, unit standard deviation
        mean = np.mean(filtered)
        std = np.std(filtered)
        normalised = (filtered - mean) / std
        # create a view on to array corresponding to a rolling window
        shape = (normalised.shape[-1] - self.window_size + 1, self.window_size)
        strides = normalised.strides + (normalised.strides[-1],)
        windowed = np.lib.stride_tricks.as_strided(
            normalised, shape=shape, strides=strides)
        # inputs are first (window_size - 1) entries in windows
        inputs = windowed[:, :-1]
        # targets are last entry in windows
        targets = windowed[:, -1]
        super(MetOfficeDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)
