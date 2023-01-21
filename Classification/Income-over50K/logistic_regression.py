import math
import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, iteration, lr, verbose=True, random_state=369):
        self.iteration = iteration
        self.lr = lr
        self.verbose = verbose

        self.w = None
        self.b = 0
        self.history = list()
        self.rand_gen = np.random.default_rng(seed=random_state)

    @staticmethod
    def sigmoid(z, high_bound=600, low_bound=1e-9) -> np.ndarray:
        z = np.clip(z, -high_bound, high_bound)
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, low_bound, 1 - low_bound)

    @staticmethod
    def parse_sigmoid(s_arr):
        return np.around(s_arr)

    def fit(self, X: np.ndarray, Y: np.ndarray, init_w=None, init_b=None):
        m, n = X.shape
        Y = Y.reshape((-1,))  # ensure Y is 1d (m, )

        if init_w is not None:
            self.w = init_w
        else:
            self.w = self.rand_gen.normal(loc=0.0, scale=0.01, size=n)

        if init_b is not None:
            self.b = init_b
        else:
            self.b = 0

        for i in range(self.iteration):
            y_hat = self.sigmoid(X @ self.w + self.b)  # (m, )
            loss = -(Y @ np.log(y_hat) + (1 - Y) @ np.log(1 - y_hat)) / m
            self.history.append(loss)

            error = y_hat - Y
            w_grad = X.T @ error / m
            b_grad = np.sum(error) / m

            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                pred_y = self.predict(X)
                acc = accuracy_score(Y, pred_y)
                print(f"Iteration {i:4d}: Cost {self.history[-1]:8.3f}   Acc: {acc:8.3f}")

    def predict(self, X):
        return self.parse_sigmoid(self.sigmoid(X @ self.w + self.b))


class LogisticRegressionAdaGrad(LogisticRegression):
    def fit(self, X: np.ndarray, Y: np.ndarray, init_w=None, init_b=None):
        m, n = X.shape
        Y = Y.reshape((-1,))  # ensure Y is 1d (m, )

        if init_w is not None:
            self.w = init_w
        else:
            self.w = self.rand_gen.normal(loc=0.0, scale=0.01, size=n)
            # self.w = np.ones(n)

        if init_b is not None:
            self.b = init_b
        else:
            self.b = 0

        sum_of_w_grad = np.zeros(n)
        sum_of_b_grad = 0
        for i in range(self.iteration):
            y_hat = self.sigmoid(X @ self.w + self.b)  # (m, )
            loss = -(Y @ np.log(y_hat) + (1 - Y) @ np.log(1 - y_hat))
            self.history.append(loss)

            error = y_hat - Y
            w_grad = X.T @ error
            b_grad = np.sum(error)

            sum_of_w_grad += w_grad ** 2
            sum_of_b_grad += b_grad ** 2

            self.w -= self.lr * w_grad / np.sqrt(sum_of_w_grad)
            self.b -= self.lr * b_grad / np.sqrt(sum_of_b_grad)

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                pred_y = self.predict(X)
                acc = accuracy_score(Y, pred_y)
                print(f"Iteration {i:4d}: Cost {self.history[-1]:8.3f}   Acc: {acc:8.3f}")
