import numpy as np


def gen_toy_data(samples, dim):
    rng = np.random.RandomState(37)
    x = rng.rand(samples, dim)
    y = 10 * x.dot(rng.random(dim)) + 3 * rng.rand(samples)
    return x, y
