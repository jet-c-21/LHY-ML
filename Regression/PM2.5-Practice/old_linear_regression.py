import math
import numpy as np
from sklearn.model_selection import KFold
from more_itertools import ilen
from tqdm import tqdm


def get_z_score_norm(X: np.ndarray):
    m, row, col = X.shape
    feat_m = np.concatenate(X, axis=1)  # (row, col * m) (18, 50868)
    mean = np.mean(feat_m, axis=1)  # (18,)
    std = np.std(feat_m, axis=1)  # (18,)

    mean_2d = np.repeat(mean[:, np.newaxis], col, axis=1)  # (18, 9)
    mean_3d = np.repeat(mean_2d[np.newaxis, :, :], m, axis=0)  # (m, 18, 9)

    std_2d = np.repeat(std[:, np.newaxis], col, axis=1)  # (18, 9)
    std_3d = np.repeat(std_2d[np.newaxis, :, :], m, axis=0)  # (m, 18, 9)

    x_norm = (X - mean_3d) / std_3d

    return x_norm, mean, std


def z_score_norm(X, mean, std):
    row, col = X.shape
    mean_2d = np.repeat(mean[:, np.newaxis], col, axis=1)  # (18, 9)
    std_2d = np.repeat(std[:, np.newaxis], col, axis=1)  # (18, 9)
    return (X - mean_2d) / std_2d


def get_lr_grid(low_bond=1e-6, high_bond=1e-2, intv=3, rd=6):
    lr_ls = [low_bond]
    while low_bond < high_bond:
        low_bond *= intv
        lr_ls.append(low_bond)

    return sorted(lr_ls, reverse=True)


def k_fold_cross_validation(X, y, k, iteration, lr, random_state=369):
    score_ls = list()
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    for f_idx, (train_idx, valid_idx) in enumerate(tqdm(kf.split(X=X), total=k, desc='Training Fold')):
        train_x, train_y = X[train_idx], y[train_idx]
        valid_x, valid_y = X[valid_idx], y[valid_idx]
        lin_reg = LinearRegression(train_x, train_y, iteration, lr, verbose=False)
        lin_reg.gradient_descent()
        loss = lin_reg.get_loss(valid_x, valid_y)
        score_ls.append(loss)

    return np.array(score_ls)


class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray, iteration, lr, epsilon=1e-6, validation=(None, None),
                 verbose=True):
        self.X = X
        self.y = y
        self.iteration = iteration
        self.lr = lr
        self.epsilon = epsilon
        self.verbose = verbose
        self.X_valid, self.y_valid = validation

        self.n = self.X[0].shape[0]
        self.w = np.zeros(self.n)
        self.b = 0

        self.history = {
            'train_loss': list(),
            'valid_loss': list(),
        }

    def predict(self, x):
        return x @ self.w + self.b

    def get_loss(self, X, y):
        m = X.shape[0]
        loss = np.sum((X @ self.w + self.b - y) ** 2) / (2 * m)
        return loss

    def get_gradient(self, X, y, w, b):
        m = X.shape[0]
        y_hat = X @ w + b
        e = y_hat - y
        w_grad = X.T @ e / m
        b_grad = np.sum(e) / m

        return w_grad, b_grad

    def gradient_descent(self):
        # last_cost = float('inf')
        for i in range(self.iteration):
            w_grad, b_grad = self.get_gradient(self.X, self.y, self.w, self.b)
            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            curr_train_loss = self.get_loss(self.X, self.y)
            self.history['train_loss'].append(curr_train_loss)

            if self.X_valid is not None and self.y_valid is not None:
                curr_valid_loss = self.get_loss(self.X_valid, self.y_valid)
                self.history['valid_loss'].append(curr_valid_loss)

            # if (last_cost - curr_loss) < self.epsilon:
            #     msg = f"Converge at iteration = {i}, cost = {curr_loss}"
            #     print(msg)
            #     return self.w, self.b, self.loss_history
            # else:
            #     last_cost = curr_loss

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.history['train_loss'][-1]:8.3f}   ")

        return self.w, self.b, self.history

    def ada_grad(self):
        lr_w = np.zeros(self.n)
        lr_b = 0
        for i in range(self.iteration):
            w_grad, b_grad = self.get_gradient(self.X, self.y, self.w, self.b)
            lr_w += w_grad ** 2
            lr_b += b_grad ** 2

            self.w -= (self.lr / np.sqrt(lr_w)) * w_grad
            self.b -= (self.lr / np.sqrt(lr_b)) * b_grad

            curr_train_loss = self.get_loss(self.X, self.y)
            self.history['train_loss'].append(curr_train_loss)

            if self.X_valid is not None and self.y_valid is not None:
                curr_valid_loss = self.get_loss(self.X_valid, self.y_valid)
                self.history['valid_loss'].append(curr_valid_loss)

            if self.verbose and i % math.ceil(self.iteration / 10) == 0:
                print(f"Iteration {i:4d}: Cost {self.history['train_loss'][-1]:8.3f}   ")

        return self.w, self.b, self.history
