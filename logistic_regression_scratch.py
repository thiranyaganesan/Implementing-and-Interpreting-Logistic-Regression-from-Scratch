import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)

            dw = (1/self.m) * np.dot(X.T, (y_pred - y))
            db = (1/self.m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        return np.round(y_pred)

# Example synthetic data test
if __name__ == "__main__":
    X = np.array([[0,1],[1,1],[2,2],[3,3]])
    y = np.array([0,0,1,1])

    model = LogisticRegressionScratch(lr=0.1, epochs=2000)
    model.fit(X, y)

    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Predictions:", model.predict(X))
