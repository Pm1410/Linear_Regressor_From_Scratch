import numpy as np


class LinearRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = None

        self.loss_history = []

    def initialize_parameters(self, n_features):
        self.w = np.random.randn(n_features)*0.01
        self.b = 0.0

    def predict(self, X):
        """
        Predict using:
        y_hat = X @ w + b
        """
        return X @ self.w + self.b

    def compute_loss(self, y_true, y_pred):
        """
        Mean Squared Error
        """
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X, y):
        """
        Training loop.
        Gradient descent will be added later.
        """
        n_samples, n_features = X.shape

        self.initialize_parameters(n_features)

        for epoch in range(self.epochs):

            # Prediction
            y_pred = self.predict(X)

            error=y_pred-y
            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            # Prediction
            y_pred = self.predict(X)

            # Loss
            loss = self.compute_loss(y, y_pred)

            # Save loss
            self.loss_history.append(loss)
            if (epoch+1)%100==0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss:.6f}")

    def get_weights(self):
        return self.w, self.b