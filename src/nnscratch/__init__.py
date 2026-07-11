"""nnscratch: a feedforward neural network built from first principles with NumPy."""

from nnscratch.gradient_check import gradient_check
from nnscratch.initializers import he_init, xavier_init
from nnscratch.layers import Dense, Dropout, ReLU, Sigmoid
from nnscratch.losses import SoftmaxCrossEntropy
from nnscratch.model import Sequential
from nnscratch.optimizers import SGD, Adam, Momentum, get_optimizer
from nnscratch.training import History, accuracy, train

__all__ = [
    "SGD",
    "Adam",
    "Dense",
    "Dropout",
    "History",
    "Momentum",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "SoftmaxCrossEntropy",
    "accuracy",
    "get_optimizer",
    "gradient_check",
    "he_init",
    "train",
    "xavier_init",
]

__version__ = "0.1.0"
