from mlp.learning_rules import AdamLearningRuleWithWeightDecay
from mlp.schedulers import CosineAnnealingWithWarmRestarts
import numpy as np

def test_adam_with_weight_decay():
    weights = np.arange(0., 1000., ).reshape(20, 50)
    grads = np.arange(-1., 1., step=(2./1000)).reshape(20, 50)

    optimizer = AdamLearningRuleWithWeightDecay()
    optimizer.initialise(params=weights)
    optimizer.update_params(grads_wrt_params=grads)

    correct_params = np.load("../data/weight_decay_correct_results.npz")
    correct_params = correct_params["updated_weights"]
    check_functionality = np.mean(np.allclose(optimizer.params, correct_params))

    return check_functionality, correct_params, optimizer.params


def test_cosine_scheduler():
    from mlp.learning_rules import AdamLearningRule
    loaded = np.load("../data/cosine_scheduler_correct_test_results.npz")
    learning_rule = AdamLearningRule()
    correct_learning_rates, epoch_idx = loaded['learning_rates'], loaded['epoch_idx']
    cosine_scheduler = CosineAnnealingWithWarmRestarts(min_learning_rate=0.0001, max_learning_rate=0.01,
                                                       total_iters_per_period=100.,
                                                       max_learning_rate_discount_factor=0.9,
                                                       period_iteration_expansion_factor=1.0)



    check_scheduler_functionality_epoch_idx_array = np.array([i for i in range(1000)])
    check_experiment_continued_functionality_epoch_idx_array = np.array([i for i in range(2000, 3000)])

    learning_rate_array = np.empty(shape=check_scheduler_functionality_epoch_idx_array.shape)

    for idx, epoch in enumerate(check_scheduler_functionality_epoch_idx_array):
        cur_learning_rate = cosine_scheduler.update_learning_rule(learning_rule=learning_rule, epoch_number=epoch)
        learning_rate_array[idx] = cur_learning_rate

    check_functionality = np.mean(np.allclose(correct_learning_rates[check_scheduler_functionality_epoch_idx_array],
                                              learning_rate_array))

    functionality_output = np.copy(learning_rate_array)
    functionality_correct = np.copy(correct_learning_rates[check_scheduler_functionality_epoch_idx_array])

    learning_rate_array = np.empty(shape=check_scheduler_functionality_epoch_idx_array.shape)

    for idx, epoch in enumerate(check_experiment_continued_functionality_epoch_idx_array):
        cur_learning_rate = cosine_scheduler.update_learning_rule(learning_rule=learning_rule, epoch_number=epoch)
        learning_rate_array[idx] = cur_learning_rate
    
    check_continuation_feature = np.mean(np.allclose(correct_learning_rates[1000:],
                                                     learning_rate_array))

    continuation_from_previous_state_output = np.copy(learning_rate_array)
    continuation_from_previous_state_correct = np.copy(correct_learning_rates[1000:])

    return check_functionality, functionality_correct, functionality_output, check_continuation_feature, continuation_from_previous_state_correct, continuation_from_previous_state_output

