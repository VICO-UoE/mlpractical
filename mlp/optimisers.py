# -*- coding: utf-8 -*-
"""Model optimisers.

This module contains objects implementing (batched) stochastic gradient descent
based optimisation of models.
"""

import time
import logging
from collections import OrderedDict
import numpy as np


logger = logging.getLogger(__name__)


class Optimiser(object):
    """Basic model optimiser."""

    def __init__(self, model, error, learning_rule, train_dataset,
                 valid_dataset=None, data_monitors=None, schedulers=[],
                 use_stochastic_eval=False):
        """Create a new optimiser instance.

        Args:
            model: The model to optimise.
            error: The scalar error function to minimise.
            learning_rule: Gradient based learning rule to use to minimise
                error.
            train_dataset: Data provider for training set data batches.
            valid_dataset: Data provider for validation set data batches.
            data_monitors: Dictionary of functions evaluated on targets and
                model outputs (averaged across both full training and
                validation data sets) to monitor during training in addition
                to the error. Keys should correspond to a string label for
                the statistic being evaluated.
            schedulers: List of learning rule scheduler objects for adjusting
                learning rule hyperparameters over training. Can be empty.
            use_stochastic_eval: Whether to use `stochastic=True` flag in
                `model.fprop` for evaluating model performance during training.
        """
        self.model = model
        self.error = error
        self.learning_rule = learning_rule
        self.learning_rule.initialise(self.model.params)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.data_monitors = OrderedDict([('error', error)])
        if data_monitors is not None:
            self.data_monitors.update(data_monitors)
        self.schedulers = schedulers
        self.use_stochastic_eval = use_stochastic_eval

    def do_training_epoch(self):
        """Do a single training epoch.

        This iterates through all batches in training dataset, for each
        calculating the gradient of the estimated error given the batch with
        respect to all the model parameters and then updates the model
        parameters according to the learning rule.
        """
        for inputs_batch, targets_batch in self.train_dataset:
            activations = self.model.fprop(inputs_batch)
            grads_wrt_outputs = self.error.grad(activations[-1], targets_batch)
            grads_wrt_params = self.model.grads_wrt_params(
                activations, grads_wrt_outputs)
            self.learning_rule.update_params(grads_wrt_params)

    def eval_monitors(self, dataset, label):
        """Evaluates the monitors for the given dataset.

        Args:
            dataset: Dataset to perform evaluation with.
            label: Tag to add to end of monitor keys to identify dataset.

        Returns:
            OrderedDict of monitor values evaluated on dataset.
        """
        data_mon_vals = OrderedDict([(key + label, 0.) for key
                                     in self.data_monitors.keys()])
        for inputs_batch, targets_batch in dataset:
            activations = self.model.fprop(
                inputs_batch, stochastic=self.use_stochastic_eval)
            for key, data_monitor in self.data_monitors.items():
                data_mon_vals[key + label] += data_monitor(
                    activations[-1], targets_batch)
        for key, data_monitor in self.data_monitors.items():
            data_mon_vals[key + label] /= dataset.num_batches
        return data_mon_vals

    def get_epoch_stats(self):
        """Computes training statistics for an epoch.

        Returns:
            An OrderedDict with keys corresponding to the statistic labels and
            values corresponding to the value of the statistic.
        """
        epoch_stats = OrderedDict()
        epoch_stats.update(self.eval_monitors(self.train_dataset, '(train)'))
        if self.valid_dataset is not None:
            epoch_stats.update(self.eval_monitors(
                self.valid_dataset, '(valid)'))
        epoch_stats['params_penalty'] = self.model.params_penalty()
        return epoch_stats

    def log_stats(self, epoch, epoch_time, stats):
        """Outputs stats for a training epoch to a logger.

        Args:
            epoch (int): Epoch counter.
            epoch_time: Time taken in seconds for the epoch to complete.
            stats: Monitored stats for the epoch.
        """
        logger.info('Epoch {0}: {1:.2f}s to complete\n  {2}'.format(
            epoch, epoch_time,
            ', '.join(['{0}={1:.2e}'.format(k, v) for (k, v) in stats.items()])
        ))

    def train(self, num_epochs, stats_interval=5):
        """Trains a model for a set number of epochs.

        Args:
            num_epochs: Number of epochs (complete passes through trainin
                dataset) to train for.
            stats_interval: Training statistics will be recorded and logged
                every `stats_interval` epochs.

        Returns:
            Tuple with first value being an array of training run statistics,
            the second being a dict mapping the labels for the statistics
            recorded to their column index in the array and the final value
            being the total time elapsed in seconds during the training run.
        """
        stats = self.get_epoch_stats()
        logger.info(
            'Epoch 0:\n  ' +
            ', '.join(['{0}={1:.2e}'.format(k, v) for (k, v) in stats.items()])
        )
        run_stats = [stats.values()]
        run_start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            for scheduler in self.schedulers:
                scheduler.update_learning_rule(self.learning_rule, epoch - 1)
            start_time = time.time()
            self.do_training_epoch()
            epoch_time = time.time() - start_time
            if epoch % stats_interval == 0:
                stats = self.get_epoch_stats()
                self.log_stats(epoch, epoch_time, stats)
                run_stats.append(stats.values())
        run_time = time.time() - run_start_time
        return (
            np.array(run_stats),
            {k: i for i, k in enumerate(stats.keys())},
            run_time
        )
