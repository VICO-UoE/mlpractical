"""Model trainers."""

import time
import logging
from collections import OrderedDict


logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, model, cost, learning_rule, train_dataset,
                 valid_dataset=None):
        self.model = model
        self.cost = cost
        self.learning_rule = learning_rule
        self.learning_rule.initialise(self.model.params)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def do_training_epoch(self):
        for inputs_batch, targets_batch in self.train_dataset:
            activations = self.model.fprop(inputs_batch)
            grads_wrt_outputs = self.cost.grad(activations[-1], targets_batch)
            grads_wrt_params = self.model.grads_wrt_params(
                activations, grads_wrt_outputs)
            self.learning_rule.update_params(grads_wrt_params)
        self.train_dataset.reset()

    def data_cost(self, dataset):
        cost = 0.
        for inputs_batch, targets_batch in dataset:
            activations = self.model.fprop(inputs_batch)
            cost += self.cost(activations[-1], targets_batch)
        dataset.reset()
        return cost

    def get_epoch_stats(self):
        epoch_stats = OrderedDict()
        epoch_stats['cost(train)'] = self.data_cost(
            self.train_dataset)
        epoch_stats['cost(valid)'] = self.data_cost(
            self.valid_dataset)
        epoch_stats['cost(param)'] = self.model.params_cost()
        return epoch_stats

    def log_stats(self, epoch, stats):
        logger.info('Epoch {0}: {1}'.format(
            epoch,
            ', '.join(['{0}={1:.3f}'.format(k, v) for (k, v) in stats.items()])
        ))

    def train(self, n_epochs, stats_interval=5):
        run_stats = []
        for epoch in range(n_epochs):
            start_time = time.clock()
            self.do_training_epoch()
            epoch_time = time.clock() - start_time
            if epoch % stats_interval == 0:
                stats = self.get_epoch_stats()
                stats['time'] = epoch_time
                self.log_stats(epoch, stats)
                run_stats.append(stats.items())
        return np.array(run_stats), stats.keys()
