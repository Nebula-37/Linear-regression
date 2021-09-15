import numpy as np


class Linear_reg:
    def __init__(self):
        self.beta = np.array([])

    def find_beta(self, x, y):
        """
        return the the values of b_0,b_1,b_2, ... ,b_m
        y_hat =  b_0 + x_1*b_1 + x_2*b_2 + ... + x_n *b_n
        """
        x = np.concatenate( (np.ones((len(x),1)),x), axis=1) # Adding a column of ones for the intercept b_0
        x2 = x.T.dot(x)
        b = np.linalg.linalg.inv(x2).dot(x.T)
        self.beta = b.dot(y)
        return self.beta

    def RSS(self, x, y):
        """
        returns the square error of the model
        """
        mat = y - x.dot(self.beta)
        return mat.T.dot(mat)
