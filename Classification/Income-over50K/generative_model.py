import numpy as np


class GenerativeModel:
    def __init__(self):
        self.c1 = None
        self.c2 = None
        self.m_c1 = None
        self.m_c2 = None
        self.mu_c1 = None
        self.mu_c2 = None
        self.sigma_c1 = None
        self.sigma_c2 = None
        self.sigma = None
        self.sigma_inv = None

    @staticmethod
    def sigmoid_ntu_ta(z, limit=1e-9) -> np.ndarray:
        """
        I think this function has some bug
        :param z:
        :param limit:
        :return:
        """
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, limit, 1 - limit)

    @staticmethod
    def sigmoid(z, high_bound=600, low_bound=1e-9) -> np.ndarray:
        z = np.clip(z, -high_bound, high_bound)
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, low_bound, 1 - low_bound)

    @staticmethod
    def parse_sigmoid(s_arr: np.ndarray):
        # return np.where(s_arr >= 0.5, 1, 0)
        return np.around(s_arr)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Y is binary class (0 or 1)
        class 1: y == 0
        class 2: y == 1
        :param X: 2d array
        :param Y: 2d array
        :return:
        """
        assert X.ndim == Y.ndim == 2
        c1_idx = np.where(Y == 1)[0]
        c2_idx = np.where(Y == 0)[0]
        self.m_c1, self.m_c2 = len(c1_idx), len(c2_idx)
        m = self.m_c1 + self.m_c2

        self.c1, self.c2 = X[c1_idx], X[c2_idx]
        self.mu_c1 = np.mean(self.c1, axis=0)
        self.mu_c2 = np.mean(self.c2, axis=0)
        # each col represents a variable, while the rows contain observations.
        self.sigma_c1 = np.cov(self.c1, rowvar=False)
        self.sigma_c2 = np.cov(self.c2, rowvar=False)
        self.sigma = (self.m_c1 / m) * self.sigma_c1 + (self.m_c2 / m) * self.sigma_c2
        self.sigma_inv = np.linalg.pinv(self.sigma)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Shape of variables:
        X (m, n)
        mu (n,)
        sigma and sigma_inv (n, n)

        :param X:
        :return:
        """
        w = (self.mu_c1 - self.mu_c2) @ self.sigma_inv
        b = (-1 / 2) * self.mu_c1 @ self.sigma_inv @ self.mu_c1 + \
            (1 / 2) * self.mu_c2 @ self.sigma_inv @ self.mu_c2 + \
            np.log(self.m_c1 / self.m_c2)
        # print(f"w shape = {w.shape}, b shape = {b.shape}, X shape = {X.shape}")

        z = X @ w + b  # (m, n) @ (n) + ()

        return self.parse_sigmoid(self.sigmoid(z))


if __name__ == '__main__':
    import pandas as pd

    train_x_csv_path = 'data/format/X_train'
    train_x_df = pd.read_csv(train_x_csv_path)
    train_y_csv_path = 'data/format/Y_train'
    train_y_df = pd.read_csv(train_y_csv_path)
    test_x_csv_path = 'data/format/X_test'
    test_x_df = pd.read_csv(test_x_csv_path)
    train_x = train_x_df.values
    train_y = train_y_df.values
    test_x = test_x_df.values
    # print(f"train_x shape: {train_x.shape}")
    # print(f"train_y shape: {train_y.shape}")
    # print(f"test_x shape: {test_x.shape}")

    gm = GenerativeModel()
    gm.fit(train_x, train_y)
    pred_y = gm.predict(test_x)
    print(np.unique(pred_y, return_counts=True))
