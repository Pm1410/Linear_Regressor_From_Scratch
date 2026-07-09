import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from model import LinearRegressionScratch
from metrics import Error_Metrics
import matplotlib
matplotlib.use("TkAgg")
X_train=np.load("data/train_data/X_train.npy")
y_train=np.load("data/train_data/y_train.npy")
X_test=np.load("data/train_data/X_test.npy")
y_test=np.load("data/train_data/y_test.npy")

def standardize(Xtr,Xte):
    mean=Xtr.mean(axis=0)
    std=Xtr.std(axis=0)
    std[std==0]=1
    return (Xtr-mean)/std,(Xte-mean)/std

def evaluate(name,y_true,y_pred):
    print(f"\n{name}")
    print("-"*40)
    print(f"MSE : {Error_Metrics.mse(y_true,y_pred):.8f}")
    print(f"RMSE: {Error_Metrics.rmse(y_true,y_pred):.8f}")
    print(f"MAE : {Error_Metrics.mae(y_true,y_pred):.8f}")
    print(f"R2  : {Error_Metrics.r2_score(y_true,y_pred):.8f}")

#======================
# Scratch Model
#======================

scratch=LinearRegressionScratch(learning_rate=0.01,epochs=1000)
scratch.fit(X_train,y_train)
pred_scratch=scratch.predict(X_test)
evaluate("Scratch Linear Regression",y_test,pred_scratch)

#======================
# Sklearn Model
#======================

sk=LinearRegression()
sk.fit(X_train,y_train)
pred_sk=sk.predict(X_test)
evaluate("Sklearn Linear Regression",y_test,pred_sk)

print("\nWeights Comparison")
print("-"*40)
print("Scratch W :",scratch.w)
print("Scratch b :",scratch.b)
print()
print("Sklearn W :",sk.coef_)
print("Sklearn b :",sk.intercept_)

#======================
# Learning Rate Comparison
#======================

plt.figure(figsize=(8,5))
lrs=[0.0001,0.001,0.01,0.1]

for lr in lrs:
    model=LinearRegressionScratch(learning_rate=lr,epochs=1000)
    model.fit(X_train,y_train)
    plt.plot(model.loss_history,label=f"lr={lr}")
    print(f"LR={lr} Final Loss={model.loss_history[-1]:.8f}")

plt.title("Learning Rate Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/learning_rate.png",dpi=300,bbox_inches="tight")
plt.show()

#======================
# Feature Scaling
#======================

Xtr_scaled,Xte_scaled=standardize(X_train,X_test)

model1=LinearRegressionScratch(learning_rate=0.01,epochs=1000)
model1.fit(X_train,y_train)

model2=LinearRegressionScratch(learning_rate=0.01,epochs=1000)
model2.fit(Xtr_scaled,y_train)

pred1=model1.predict(X_test)
pred2=model2.predict(Xte_scaled)

evaluate("Without Scaling",y_test,pred1)
evaluate("With Scaling",y_test,pred2)

plt.figure(figsize=(8,5))
plt.plot(model1.loss_history,label="Without Scaling")
plt.plot(model2.loss_history,label="With Scaling")
plt.title("Scaling Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/scaling_comparison.png",dpi=300,bbox_inches="tight")
plt.show()

#======================
# Prediction Comparison
#======================

plt.figure(figsize=(6,6))
plt.scatter(y_test,pred_scratch,s=10,label="Scratch")
plt.scatter(y_test,pred_sk,s=10,label="Sklearn")
mn=min(y_test.min(),pred_sk.min(),pred_scratch.min())
mx=max(y_test.max(),pred_sk.max(),pred_scratch.max())
plt.plot([mn,mx],[mn,mx],'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Prediction Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/prediction_comparison.png",dpi=300,bbox_inches="tight")
plt.show()

Xb=np.c_[np.ones((X_train.shape[0],1)),X_train]
theta=np.linalg.solve(Xb.T @ Xb, Xb.T @ y_train)
print("Xb :",Xb)
print("Theta :",theta)