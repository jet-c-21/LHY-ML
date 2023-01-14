import numpy as np


class LinearRegressionGD_ADA(object):

    def __init__(self, eta=1, n_iter=2000, random_state=1, shuffle=True, alpha=0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.alpha = alpha

    def fit(self, X, y):
        print(X.shape)
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        lr_b = 0
        lr_w = np.zeros(X.shape[1])
        for i in range(self.n_iter):

            b_grad = 0.0
            w_grad = np.zeros(X.shape[1])

            if self.shuffle:
                X, y = self._shuffle(X, y)

            for xi, target in zip(X, y):  # iterate on single sample
                cost = []  # record cost for each sample
                output = self.net_input(xi)
                error = (target - output)

                w_grad = w_grad - 2 * xi.dot(error)
                b_grad = b_grad - 2 * error
            #                 self.w_[1:] += 2* self.eta * xi.dot(error)
            #                 self.w_[0] += 2*self.eta * error

            lr_b = lr_b + b_grad ** 2
            lr_w = lr_w + w_grad ** 2

            self.w_[1:] = self.w_[1:] - self.eta / np.sqrt(lr_w) * w_grad + self.alpha * self.w_[1:]
            self.w_[0] = self.w_[0] - self.eta / np.sqrt(lr_b) * b_grad

            # calculate RMSE for an epoch
            errors = (sum((y - (self.net_input(X))) ** 2) / len(y)) ** 0.5
            self.cost_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
