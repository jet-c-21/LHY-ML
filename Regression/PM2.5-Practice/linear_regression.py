import math
import numpy as np


class CloseFormSol:
    def __init__(self):
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        m, n = X.shape
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        assert X.ndim == 2
        m, n = X.shape
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        return X @ self.w


class GradientDescent:
    def __init__(self, iteration, lr, verbose=True):
        self.iteration = iteration
        self.lr = lr
        self.verbose = verbose

        self.w = None
        self.history = list()

    @staticmethod
    def add_bias(X: np.ndarray):
        assert X.ndim == 2
        m, n = X.shape
        return np.concatenate((np.ones((m, 1)), X), axis=1)

    def fit(self, X, Y, init_w=None):
        X = self.add_bias(X)
        m, n = X.shape
        if init_w is not None:
            self.w = init_w
        else:
            self.w = np.zeros(n)

        for i in range(self.iteration):
            y_hat = X @ self.w
            error = y_hat - Y
            loss = np.sum(error ** 2) / m
            self.history.append(loss)

            w_grad = 2 * X.T @ error / m
            self.w -= self.lr * w_grad

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.history[-1]:8.3f}   ")

    def predict(self, X):
        X = self.add_bias(X)
        return X @ self.w


class AdaGrad(GradientDescent):
    def fit(self, X, Y, init_w=None):
        X = self.add_bias(X)
        m, n = X.shape
        if init_w is not None:
            self.w = init_w
        else:
            self.w = np.zeros(n)

        sum_of_grad = np.zeros(n)
        for i in range(self.iteration):
            y_hat = X @ self.w
            error = y_hat - Y
            loss = np.sum(error ** 2) / m
            self.history.append(loss)

            w_grad = 2 * X.T @ error / m
            sum_of_grad += w_grad ** 2
            self.w -= self.lr * w_grad / np.sqrt(sum_of_grad)

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.history[-1]:8.3f}   ")


class GradientDescentBiasSep:
    def __init__(self, iteration, lr, verbose=True):
        self.iteration = iteration
        self.lr = lr
        self.verbose = verbose

        self.w = None
        self.b = 0
        self.history = list()

    def fit(self, X, Y, init_w=None, init_b=None):
        m, n = X.shape

        if init_w is not None:
            self.w = init_w
        else:
            self.w = np.zeros(n)

        if init_b is not None:
            self.b = init_b
        else:
            self.b = 0

        for i in range(self.iteration):
            y_hat = X @ self.w + self.b
            error = y_hat - Y
            loss = np.sum(error ** 2) / m
            self.history.append(loss)

            w_grad = 2 * X.T @ error / m
            b_grad = 2 * np.sum(error) / m

            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.history[-1]:8.3f}   ")

    def predict(self, X):
        return X @ self.w + self.b


class AdaGradBiasSep(GradientDescentBiasSep):
    def fit(self, X, Y, init_w=None, init_b=None):
        m, n = X.shape

        if init_w is not None:
            self.w = init_w
        else:
            self.w = np.zeros(n)

        if init_b is not None:
            self.b = init_b
        else:
            self.b = 0

        sum_of_w_grad = np.zeros(n)
        sum_of_b_grad = 0
        for i in range(self.iteration):
            y_hat = X @ self.w + self.b
            error = y_hat - Y
            loss = np.sum(error ** 2) / m
            self.history.append(loss)

            w_grad = 2 * X.T @ error / m
            b_grad = 2 * np.sum(error) / m

            sum_of_w_grad += w_grad ** 2
            sum_of_b_grad += b_grad ** 2

            self.w -= self.lr * w_grad / np.sqrt(sum_of_w_grad)
            self.b -= self.lr * b_grad / np.sqrt(sum_of_b_grad)

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.history[-1]:8.3f}   ")
