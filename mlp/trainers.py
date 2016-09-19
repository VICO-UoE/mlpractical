# -*- coding: utf-8 -*-
"""Model trainers."""

import time
import logging
from collections import OrderedDict
import numpy as np


logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, model, cost, learning_rule, train_dataset,
                 valid_dataset=None, data_monitors=None):
        self.model = model
        self.cost = cost
        self.learning_rule = learning_rule
        self.learning_rule.initialise(self.model.params)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.data_monitors = OrderedDict([('cost', cost)])
        if data_monitors is not None:
            self.data_monitors.update(data_monitors)

    def do_training_epoch(self):
        for inputs_batch, targets_batch in self.train_dataset:
            activations = self.model.fprop(inputs_batch)
            grads_wrt_outputs = self.cost.grad(activations[-1], targets_batch)
            grads_wrt_params = self.model.grads_wrt_params(
                activations, grads_wrt_outputs)
            self.learning_rule.update_params(grads_wrt_params)

    def monitors(self, dataset, label):
        data_mon_vals = OrderedDict([(key + label, 0.) for key
                                     in self.data_monitors.keys()])
        for inputs_batch, targets_batch in dataset:
            activations = self.model.fprop(inputs_batch)
            for key, data_monitor in self.data_monitors.items():
                data_mon_vals[key + label] += data_monitor(
                    activations[-1], targets_batch)
        for key, data_monitor in self.data_monitors.items():
            data_mon_vals[key + label] /= dataset.num_batches
        return data_mon_vals

    def get_epoch_stats(self):
        epoch_stats = OrderedDict()
        epoch_stats.update(self.monitors(self.train_dataset, '(train)'))
        if self.valid_dataset is not None:
            epoch_stats.update(self.monitors(self.valid_dataset, '(valid)'))
        epoch_stats['cost(param)'] = self.model.params_cost()
        return epoch_stats

    def log_stats(self, epoch, epoch_time, stats):
        logger.info('Epoch {0}: {1:.1f}s to complete\n    {2}'.format(
            epoch, epoch_time,
            ', '.join(['{0}={1:.2e}'.format(k, v) for (k, v) in stats.items()])
        ))

    def train(self, n_epochs, stats_interval=5):
        run_stats = []
        for epoch in range(1, n_epochs + 1):
            start_time = time.clock()
            self.do_training_epoch()
            epoch_time = time.clock() - start_time
            if epoch % stats_interval == 0:
                stats = self.get_epoch_stats()
                self.log_stats(epoch, epoch_time, stats)
                run_stats.append(stats.values())
        return np.array(run_stats), {k: i for i, k in enumerate(stats.keys())}
