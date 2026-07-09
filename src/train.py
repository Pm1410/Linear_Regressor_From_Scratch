import numpy as np
from model import LinearRegressionScratch
from metrics import Error_Metrics
import matplotlib
matplotlib.use("TkAgg")

X_train = np.load("data/train_data/X_train.npy")
y_train = np.load("data/train_data/y_train.npy")

X_test = np.load("data/train_data/X_test.npy")
y_test = np.load("data/train_data/y_test.npy")

model = LinearRegressionScratch(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X_train, y_train)

# predictions = model.predict(X_train)
y_pred = model.predict(X_test)
er=Error_Metrics()

print("MSE :", Error_Metrics.mse(y_test, y_pred))
print("RMSE:", Error_Metrics.rmse(y_test, y_pred))
print("MAE :", Error_Metrics.mae(y_test, y_pred))
print("R2  :", Error_Metrics.r2_score(y_test, y_pred))
# print(predictions[:5])
# print(model.loss_history)

import matplotlib.pyplot as plt

plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
