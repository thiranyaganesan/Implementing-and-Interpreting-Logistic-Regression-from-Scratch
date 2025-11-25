import numpy as np
from sklearn.datasets import make_classification

class LogisticRegressionScratch:
    def __init__(self, lr=0.05, epochs=8000, tol=1e-6):
        self.lr = lr
        self.epochs = epochs
        self.tol = tol

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        eps = 1e-9
        return -np.mean(y*np.log(y_pred+eps) + (1-y)*np.log(1-y_pred+eps))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        prev_loss = float('inf')

        for _ in range(self.epochs):
            linear = X @ self.weights + self.bias
            y_pred = self.sigmoid(linear)

            loss = self.binary_cross_entropy(y, y_pred)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            dw = (1/m) * (X.T @ (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = X @ self.weights + self.bias
        return (self.sigmoid(linear) >= 0.5).astype(int)

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=600,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        random_state=42
    )

    model = LogisticRegressionScratch()
    model.fit(X, y)

    print("Weights:", model.weights)
    print("Bias:", model.bias)
    preds = model.predict(X)
    print("Accuracy:", np.mean(preds == y))
