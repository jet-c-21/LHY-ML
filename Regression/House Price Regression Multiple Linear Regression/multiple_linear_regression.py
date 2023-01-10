import math
import numpy as np


class MultiLinearRegression:
    def __init__(self, X, y, iteration, lr):
        self.X = X
        self.y = y
        self.iteration = iteration
        self.lr = lr

        self.n = self.X[0].shape[0]
        self.w = np.zeros(self.n)
        self.b = 0

        self.loss_history = list()

    def predict(self, x, w, b):
        return x @ w + b

    def get_loss(self, X, y, w, b):
        m = X.shape[0]
        loss = np.sum((X.dot(w) + b - y) ** 2) / (2 * m)
        return loss

    def get_gradient(self, X, y, w, b):
        m = X.shape[0]
        y_hat = X @ w + b
        e = y_hat - y
        w_grad = X.T @ e / m
        b_grad = np.sum(e) / m

        return w_grad, b_grad

    def gradient_descent(self):
        for i in range(self.iteration):
            w_grad, b_grad = self.get_gradient(self.X, self.y, self.w, self.b)
            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            self.loss_history.append(self.get_loss(self.X, self.y, self.w, self.b))

            if i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.loss_history[-1]:8.2f}   ")

        return self.w, self.b, self.loss_history
