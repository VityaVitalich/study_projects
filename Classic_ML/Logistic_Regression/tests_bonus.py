import numpy as np
import numpy.testing as npt
from modules.losses_solution import MultinomialLoss


def test_function_multiclass():
    loss_function = MultinomialLoss(l2_coef=0.5)
    X = np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ])
    y = np.array([0, 2, 1])
    w = np.array([
        [1, -1, 1],
        [2, -1.5, 0],
        [0.5, 0, 1.2],
    ])
    npt.assert_almost_equal(loss_function.func(X, y, w), 5.28054, decimal=5)


def test_gradient_multiclass():
    loss_function = MultinomialLoss(l2_coef=0.5)
    X = np.array([
        [1, 1, 2],
        [1, 3, 4],
        [1, -5, 6]
    ])
    y = np.array([0, 2, 1])
    w = np.array([
        [1, -1, 1],
        [2, -1.5, 0],
        [0.5, 0, 1.2],
    ])
    right_gradient = np.array([
       [-0.328565  , -0.67679358, -0.3137294 ],
       [ 2.02528761, -2.63980797, -0.17268757],
       [ 0.80327739, -2.51673179,  6.01975031],
    ])
    npt.assert_almost_equal(loss_function.grad(X, y, w), right_gradient, decimal=5)
    