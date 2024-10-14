# from mlp.learning_rules import AdamLearningRuleWithWeightDecay
# from mlp.schedulers import CosineAnnealingWithWarmRestarts
from mlp.layers import DropoutLayer
from mlp.penalties import L1Penalty, L2Penalty
import numpy as np
import os



def test_dropout_layer():
    # loaded = np.load("../data/correct_results.npz")
    rng = np.random.RandomState(92019)
    
    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'regularization_debug_pack.npy'), allow_pickle=True).item()
    
    rng = np.random.RandomState(92019)
    layer = DropoutLayer(rng=rng)

    out = layer.fprop(x)

    grads = layer.bprop(inputs=x, outputs=out, grads_wrt_outputs=np.ones(x.shape))

#     correct_outputs = correct_outputs['dropout']

    fprop_test = np.allclose(correct_outputs['DropoutLayer_fprop'], out)

    bprop_test = np.allclose(correct_outputs['DropoutLayer_bprop'], grads)

    return fprop_test, out, correct_outputs['DropoutLayer_fprop'], bprop_test, grads, correct_outputs['DropoutLayer_bprop']


def test_L1_Penalty():
    

    rng = np.random.RandomState(92019)
    
    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'regularization_debug_pack.npy'), allow_pickle=True).item()
    
    layer = L1Penalty(1e-4)

    out = layer(x)

    grads = layer.grad(x)

#     correct_outputs = correct_outputs['l1penalty']

    __call__test = np.allclose(correct_outputs['L1Penalty___call__correct'], out)

    grad_test = np.allclose(correct_outputs['L1Penalty_grad_correct'], grads)

    return __call__test, out, correct_outputs['L1Penalty___call__correct'], grad_test, grads, correct_outputs['L1Penalty_grad_correct']


def test_L2_Penalty():
    

    rng = np.random.RandomState(92019)
    
    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(os.path.join(os.environ['MLP_DATA_DIR'], 'regularization_debug_pack.npy'), allow_pickle=True).item()
    
    layer = L2Penalty(1e-4)

    out = layer(x)

    grads = layer.grad(x)

#     correct_outputs = correct_outputs['l2penalty']

    __call__test = np.allclose(correct_outputs['L2Penalty___call__correct'], out)

    grad_test = np.allclose(correct_outputs['L2Penalty_grad_correct'], grads)

    return __call__test, out, correct_outputs['L2Penalty___call__correct'], grad_test, grads, correct_outputs['L2Penalty_grad_correct']

