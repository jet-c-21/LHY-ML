import numpy as np
import math


class UnivarGD:
    def __init__(self, X, Y, w_init, b_init, iteration, lr=1e-2):
        self.X = X
        self.Y = Y
        self.w = w_init
        self.b = b_init
        self.iteration = iteration
        self.lr = lr

        self.param_history = list()
        self.loss_history = list()

    def predict(self, x, w, b):
        return w * x + b

    def get_loss(self, x: np.ndarray, y: np.ndarray, w, b):
        assert len(x) == len(y)
        loss = 0
        n = len(x)
        for i in range(n):
            y_hat = self.predict(x[i], w, b)
            loss += (y_hat - y[i]) ** 2
        loss = (1 / (2 * n)) * loss
        return loss

    def get_gradient(self, x: np.ndarray, y: np.ndarray, w, b):
        w_grad = 0
        b_grad = 0
        n = len(x)
        for i in range(n):
            y_hat = self.predict(x[i], w, b)
            w_grad += (y_hat - y[i]) * x[i]
            b_grad += (y_hat - y[i])

        w_grad /= n
        b_grad /= n
        return w_grad, b_grad

    def gradient_descent(self):
        for i in range(self.iteration):
            w_grad, b_grad = self.get_gradient(self.X, self.Y, self.w, self.b)
            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            self.param_history.append((self.w, self.b))
            self.loss_history.append(self.get_loss(self.X, self.Y, self.w, self.b))

            if i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4}: Loss {self.loss_history[-1]:0.2e} ",
                      f"w_grad: {w_grad: 0.3e}, b_grad: {b_grad: 0.3e}  ",
                      f"w: {self.w: 0.3e}, b:{self.b: 0.5e}")

        return self.w, self.b, self.param_history, self.loss_history
