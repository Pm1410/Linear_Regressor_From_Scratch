# Linear Regression From Scratch using NumPy

## Overview

This project implements **Linear Regression completely from scratch** using only **NumPy**. No machine learning libraries such as `scikit-learn`, `statsmodels`, or `TensorFlow` are used for the implementation.

Every component of the algorithm is implemented manually, including:

- Weight initialization
- Forward propagation
- Mean Squared Error (MSE) loss
- Gradient computation
- Gradient Descent optimization
- Prediction
- Performance metrics
- Feature Scaling
- Learning Rate comparison
- Comparison with Scikit-Learn
- Comparison with the Normal Equation

The objective of this project is to understand the mathematical foundations behind one of the most important machine learning algorithms rather than relying on existing libraries.

---

# Mathematical Background

Suppose we have

- n training samples
- m features

Our dataset is

\[
X \in \mathbb{R}^{n \times m}
\]

Target values

\[
y \in \mathbb{R}^{n}
\]

Weights

\[
w \in \mathbb{R}^{m}
\]

Bias

\[
b \in \mathbb{R}
\]

---

# Linear Regression

Linear Regression assumes that the target variable can be approximated as a linear combination of the input features.

The hypothesis is

\[
\hat y = Xw+b
\]

where

- X = Feature Matrix
- w = Weight Vector
- b = Bias
- ŷ = Predicted Output

In NumPy

```python
y_hat = X @ w + b
```

---

# Why Bias?

Without bias

\[
\hat y=Xw
\]

the regression line must always pass through the origin.

Bias shifts the regression line upward or downward.

Final equation

\[
\boxed{\hat y=Xw+b}
\]

---

# Cost Function

The objective of Linear Regression is to minimize prediction error.

The most common loss function is Mean Squared Error (MSE).

\[
MSE=\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

NumPy implementation

```python
loss=np.mean((y-y_hat)**2)
```

Properties

- Always non-negative
- Convex
- Differentiable
- Gives larger penalty to larger errors

---

# Why Square the Error?

Instead of

\[
y-\hat y
\]

we use

\[
(y-\hat y)^2
\]

because

- Positive and negative errors do not cancel
- Large errors are penalized more
- The function becomes differentiable

---

# Optimization Objective

We want to find

\[
w^*,b^*
\]

such that

\[
\boxed{\arg\min_{w,b} MSE(w,b)}
\]

This optimization problem is solved using Gradient Descent.

---

# Gradient Descent

Gradient Descent is an optimization algorithm that iteratively updates parameters in the direction of decreasing loss.

General update rule

\[
\theta=\theta-\alpha\nabla J(\theta)
\]

where

- θ = parameters
- α = learning rate
- J = cost function

---

# Gradient of MSE

Prediction

\[
\hat y=Xw+b
\]

Error

\[
e=\hat y-y
\]

Weight gradient

\[
\boxed{
\frac{\partial L}{\partial w}
=
\frac{2}{n}X^Te
}
\]

Bias gradient

\[
\boxed{
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum e
}
\]

NumPy

```python
error=y_hat-y

dw=(2/n)*(X.T@error)

db=(2/n)*np.sum(error)
```

---

# Weight Update

Weights are updated every epoch.

\[
w=w-\alpha dw
\]

Bias

\[
b=b-\alpha db
\]

NumPy

```python
self.w-=self.learning_rate*dw

self.b-=self.learning_rate*db
```

---

# Complete Training Algorithm

1. Initialize weights randomly.
2. Compute predictions.
3. Compute loss.
4. Compute gradients.
5. Update weights.
6. Repeat until convergence.

Pseudo Code

```
Initialize w,b

Repeat:

    Predict

    Compute Loss

    Compute Gradient

    Update Parameters

End
```

---

# Why Gradient Descent Works

Gradient points in the direction of maximum increase.

Moving in the opposite direction decreases the loss.

Eventually the algorithm reaches the minimum of the cost function where

\[
\nabla J=0
\]

which means no further improvement is possible.

---

# Learning Rate

Learning Rate controls the size of each update.

Small Learning Rate

- Slow convergence
- Stable training

Large Learning Rate

- Faster convergence
- Can overshoot the minimum
- May diverge

Update equation

\[
w=w-\alpha dw
\]

where

\[
\alpha
\]

is the learning rate.

# Feature Scaling

## Why Feature Scaling?

Gradient Descent updates each weight according to the magnitude of its corresponding feature.

Suppose one feature ranges between

```
0 - 1
```

while another ranges between

```
0 - 1,000,000
```

The larger feature dominates the gradient, causing:

- Slow convergence
- Oscillations
- Numerical instability
- Poor optimization

Scaling places every feature on approximately the same scale.

---

# Standardization (Z-Score Scaling)

This project uses **Standardization**.

Formula

\[
x'=\frac{x-\mu}{\sigma}
\]

where

- μ = mean of training feature
- σ = standard deviation of training feature

NumPy implementation

```python
mean=X_train.mean(axis=0)
std=X_train.std(axis=0)

std[std==0]=1

X_train=(X_train-mean)/std
X_test=(X_test-mean)/std
```

Notice that

- Mean and standard deviation are computed **only on the training set**.
- The same values are used to transform the test set.

This prevents **data leakage**.

---

# Data Leakage

Data leakage occurs when information from the test data influences the training process.

Incorrect

```python
mean=np.mean(np.concatenate([X_train,X_test]))
```

Correct

```python
mean=X_train.mean(axis=0)
```

The model should never access future or unseen data during training.

---

# Performance Metrics

Model performance is evaluated using four metrics.

---

## Mean Squared Error (MSE)

Formula

\[
MSE=\frac1n\sum(y-\hat y)^2
\]

Implementation

```python
np.mean((y-y_pred)**2)
```

Properties

- Lower is better
- Penalizes large errors heavily
- Differentiable
- Used during training

---

## Root Mean Squared Error (RMSE)

Formula

\[
RMSE=\sqrt{MSE}
\]

Implementation

```python
np.sqrt(np.mean((y-y_pred)**2))
```

Properties

- Same units as target
- Easier to interpret than MSE

---

## Mean Absolute Error (MAE)

Formula

\[
MAE=\frac1n\sum|y-\hat y|
\]

Implementation

```python
np.mean(np.abs(y-y_pred))
```

Properties

- Less sensitive to outliers
- Direct interpretation of prediction error

---

## R² Score

Formula

\[
R^2=
1-
\frac{\sum(y-\hat y)^2}
{\sum(y-\bar y)^2}
\]

Implementation

```python
ss_res=np.sum((y-y_pred)**2)
ss_tot=np.sum((y-np.mean(y))**2)

r2=1-(ss_res/ss_tot)
```

Interpretation

| R² | Meaning |
|----|----------|
|1|Perfect prediction|
|0|Same as predicting the mean|
|<0|Worse than predicting the mean|

---

# Normal Equation

Gradient Descent is an iterative optimization algorithm.

Linear Regression also has an analytical solution called the Normal Equation.

\[
\theta=(X^TX)^{-1}X^Ty
\]

Instead of explicitly computing the inverse, this project uses

```python
theta=np.linalg.solve(X.T@X,X.T@y)
```

which is more numerically stable.

---

# Why Compare With Normal Equation?

Gradient Descent should converge toward the same solution obtained analytically.

The comparison verifies that the implementation is mathematically correct.

---

# Comparison with Scikit-Learn

Scikit-Learn's `LinearRegression` computes the optimal parameters using least squares (typically via SVD or related numerical methods), not gradient descent.

Comparing our implementation against Scikit-Learn verifies that:

- Gradient computation is correct.
- Weight updates are correct.
- Prediction equation is correct.
- Loss function is correct.

Very similar metrics indicate a correct implementation.

---

# Learning Rate Experiment

The model was trained using four different learning rates.

```
0.0001
0.001
0.01
0.1
```

Observation

- Very small learning rate converges slowly.
- Moderate learning rate converges efficiently.
- Large learning rate converges faster but may become unstable on different datasets.

The optimal learning rate depends on the dataset and feature scaling.

---

# Feature Scaling Experiment

Two experiments were performed.

### Without Scaling

- Original features
- Slower optimization
- Slightly higher final loss

### With Standardization

- Zero mean
- Unit variance
- Faster convergence
- Slightly better performance

On this dataset the improvement is small because stock returns are already on a similar scale.

---

# Why Stock Return Prediction is Difficult

Daily stock returns contain very little linear information.

Using only the previous five daily returns to predict tomorrow's return is a difficult problem.

As a result

- R² remains close to zero.
- Predictions concentrate near the average return.
- This behavior is expected for financial return forecasting.

Poor predictive performance does **not** imply an incorrect implementation.

---

# Time Complexity

Let

- n = number of samples
- m = number of features

Prediction

\[
O(nm)
\]

Gradient computation

\[
O(nm)
\]

One training epoch

\[
O(nm)
\]

Total training

\[
O(E\times n\times m)
\]

where E is the number of epochs.

---

# Space Complexity

Feature matrix

\[
O(nm)
\]

Weights

\[
O(m)
\]

Predictions

\[
O(n)
\]

Loss history

\[
O(E)
\]

Overall

\[
O(nm)
\]

---

# Project Structure

```
Linear_Regression_From_Scratch/

│

├── data/

│   ├── raw/

│   └── train_data/

│

├── src/

│   ├── model.py

│   ├── metrics.py

│   ├── train.py

│   └── experiments.py

│

├── results/

│

├── README.md

│

└── requirements.txt
```

---

# Training Workflow

```
Load Dataset

↓

Create Feature Matrix

↓

Initialize Weights

↓

Forward Pass

↓

Compute MSE

↓

Compute Gradients

↓

Update Parameters

↓

Repeat Until Convergence

↓

Evaluate Model

↓

Compare Results

↓

Generate Plots
```
# Experimental Results

The following experiments were conducted to verify the correctness of the implementation.

---

# Experiment 1 : Scratch Linear Regression

Objective

- Verify that the model learns using Gradient Descent.
- Observe the decrease in training loss.

Observation

- Loss decreases monotonically.
- Model converges after sufficient epochs.
- Gradient Descent successfully minimizes MSE.

---

# Experiment 2 : Learning Rate Comparison

Learning Rates Tested

```

0.0001
0.001
0.01
0.1

```

Observation

| Learning Rate | Observation |
|---------------|-------------|
|0.0001|Very slow convergence|
|0.001|Stable convergence|
|0.01|Fast and stable|
|0.1|Fastest convergence on this dataset|

Conclusion

Learning rate directly controls the speed of optimization.

A very small learning rate requires many epochs.

A very large learning rate may overshoot the optimum or diverge on difficult datasets.

---

# Experiment 3 : Feature Scaling

Two experiments were performed.

Without Standardization

- Original features
- Slightly slower convergence

With Standardization

- Mean = 0
- Standard Deviation = 1
- Faster optimization
- Slightly lower loss

Conclusion

Feature scaling improves Gradient Descent because all features contribute on a similar numerical scale.

---

# Experiment 4 : Comparison with Scikit-Learn

The scratch implementation was compared against

```

sklearn.linear_model.LinearRegression

```

Comparison

- Predictions are very similar.
- Error metrics are almost identical.
- Confirms the correctness of the implementation.

The difference in weights occurs because

- Scratch model uses Gradient Descent.
- Scikit-Learn computes the exact least-squares solution.

---

# Experiment 5 : Comparison with Normal Equation

The analytical solution was computed using

```python
theta=np.linalg.solve(X.T@X,X.T@y)
```

The obtained parameters match the Scikit-Learn solution.

This verifies the mathematical correctness of the implementation.

---

# Observations

1.

Gradient Descent successfully minimizes the Mean Squared Error.

2.

Learning rate is the most important hyperparameter in Gradient Descent.

3.

Feature Scaling improves convergence speed.

4.

The analytical solution and Scikit-Learn produce identical coefficients.

5.

The scratch implementation achieves nearly identical prediction performance.

6.

Stock return prediction using only five previous returns is extremely difficult.

Therefore

- R² remains close to zero.
- Predictions remain close to the average return.

This is expected behaviour.

---

# Advantages

- Complete understanding of Gradient Descent.
- No dependency on ML libraries.
- Fully vectorized NumPy implementation.
- Easily extendable.
- Efficient matrix operations.
- Good educational implementation.

---

# Limitations

- Supports only Linear Regression.
- Uses Batch Gradient Descent.
- No Mini-Batch Gradient Descent.
- No Stochastic Gradient Descent.
- No Regularization.
- No Polynomial Features.
- No Cross Validation.
- No Early Stopping.
- Sensitive to Learning Rate.
- Cannot model nonlinear relationships.

---

# Possible Improvements

Future improvements include

- Ridge Regression
- Lasso Regression
- ElasticNet
- Polynomial Regression
- Mini Batch Gradient Descent
- Stochastic Gradient Descent
- Momentum
- RMSProp
- Adam Optimizer
- Feature Selection
- Cross Validation
- Early Stopping
- Regularization
- Multiple Output Regression

---

# Common Mistakes

### Using the test data for scaling

Incorrect

```python
mean=np.mean(np.concatenate([X_train,X_test]))
```

Correct

```python
mean=X_train.mean(axis=0)
```

---

### Forgetting the bias

Incorrect

```python
y=X@w
```

Correct

```python
y=X@w+b
```

---

### Wrong gradient

Incorrect

```python
dw=X.T@(y-y_pred)
```

Correct

```python
dw=(2/n)*(X.T@(y_pred-y))
```

---

### Wrong update direction

Incorrect

```python
w=w+lr*dw
```

Correct

```python
w=w-lr*dw
```

---

### Computing loss before prediction

Correct order

```

Prediction

↓

Loss

↓

Gradient

↓

Update

```

---

### Scaling the target variable unnecessarily

Only features should be standardized in this project.

---

# Key Takeaways

- Linear Regression is a linear model.
- MSE measures prediction error.
- Gradient Descent minimizes MSE.
- Learning Rate controls optimization speed.
- Feature Scaling improves convergence.
- Normal Equation gives the analytical optimum.
- Scikit-Learn verifies the implementation.
- Daily stock returns have very weak linear predictability.

---

# References

Christopher Bishop

Pattern Recognition and Machine Learning

---

Trevor Hastie

The Elements of Statistical Learning

---

Andrew Ng

Machine Learning Specialization

---

Ian Goodfellow

Deep Learning

---

NumPy Documentation

https://numpy.org/doc/

---

Scikit-Learn Documentation

https://scikit-learn.org/

---

MIT OpenCourseWare

Linear Algebra

---

Stanford CS229

Machine Learning
# Interview Questions

## 1. What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to predict a continuous target variable by fitting a linear relationship between the input features and the target.

---

## 2. What is the equation of Linear Regression?

\[
\hat y=Xw+b
\]

where

- X = Feature Matrix
- w = Weight Vector
- b = Bias
- ŷ = Prediction

---

## 3. Why do we use Linear Regression?

- Continuous value prediction
- Trend estimation
- Forecasting
- Baseline model for regression tasks

---

## 4. What is Supervised Learning?

Learning from labeled examples.

Input

```
X
```

Output

```
y
```

---

## 5. What is the objective of Linear Regression?

To minimize the prediction error between

```
Actual Values
```

and

```
Predicted Values
```

---

## 6. What is the Cost Function?

A function that measures how wrong the predictions are.

For Linear Regression

\[
MSE=\frac1n\sum(y-\hat y)^2
\]

---

## 7. Why is MSE used?

- Differentiable
- Convex
- Penalizes large errors
- Easy to optimize

---

## 8. Why square the error?

- Removes negative signs
- Gives larger penalties to larger mistakes
- Makes optimization smooth

---

## 9. What is Gradient Descent?

An optimization algorithm that minimizes the loss function by iteratively updating model parameters.

---

## 10. What is a Gradient?

The gradient is the vector of partial derivatives.

It indicates the direction of maximum increase of a function.

---

## 11. Why move opposite to the gradient?

Because the gradient points toward increasing loss.

Moving in the opposite direction decreases the loss.

---

## 12. What is Learning Rate?

The step size used during parameter updates.

\[
w=w-\alpha dw
\]

---

## 13. What happens if the learning rate is too small?

- Slow convergence
- More epochs required

---

## 14. What happens if the learning rate is too large?

- Overshooting
- Oscillation
- Divergence

---

## 15. What is an Epoch?

One complete pass through the entire training dataset.

---

## 16. What is Batch Gradient Descent?

Gradient is computed using the entire dataset before updating parameters.

---

## 17. What is Stochastic Gradient Descent?

Parameters are updated after every training sample.

---

## 18. What is Mini-Batch Gradient Descent?

Parameters are updated after a small batch of samples.

---

## 19. Why initialize weights randomly?

To avoid identical updates and ensure proper optimization.

---

## 20. Can weights be initialized to zero?

Yes, for Linear Regression.

Unlike neural networks, zero initialization does not cause symmetry problems.

---

## 21. What is Bias?

A constant term that shifts the regression line.

---

## 22. Why is Bias important?

Without bias

```
y=Xw
```

the regression line must pass through the origin.

---

## 23. What is Feature Scaling?

Transforming features so they have similar numerical ranges.

---

## 24. Why perform Feature Scaling?

- Faster convergence
- Better numerical stability
- Balanced gradients

---

## 25. What is Standardization?

\[
x'=\frac{x-\mu}{\sigma}
\]

Mean becomes

```
0
```

Standard deviation becomes

```
1
```

---

## 26. Should we scale the test set independently?

No.

Use the training mean and standard deviation.

---

## 27. What is Data Leakage?

Using information from the test data during training.

---

## 28. What is Overfitting?

Model memorizes training data but performs poorly on unseen data.

---

## 29. What is Underfitting?

Model is too simple to learn the underlying relationship.

---

## 30. What is the Normal Equation?

Analytical solution

\[
\theta=(X^TX)^{-1}X^Ty
\]

or

```python
theta=np.linalg.solve(X.T@X,X.T@y)
```

---

## 31. Why use Gradient Descent if Normal Equation exists?

Normal Equation becomes computationally expensive for large feature spaces.

Gradient Descent scales much better.

---

## 32. What is RMSE?

Square root of MSE.

It has the same units as the target.

---

## 33. What is MAE?

Average absolute prediction error.

Less sensitive to outliers than MSE.

---

## 34. What is R² Score?

Measures how much variance is explained by the model.

---

## 35. What does R² = 1 mean?

Perfect prediction.

---

## 36. What does R² = 0 mean?

Equivalent to predicting the mean every time.

---

## 37. What does R² < 0 mean?

Model performs worse than predicting the mean.

---

## 38. Why is R² negative in stock prediction?

Daily stock returns contain very weak linear relationships.

Previous five returns are insufficient to predict the next return accurately.

---

## 39. Why compare with Scikit-Learn?

To verify that the scratch implementation is correct.

---

## 40. Why do Gradient Descent and Scikit-Learn have different weights?

Scikit-Learn computes the exact least-squares solution.

Gradient Descent approximates the optimum iteratively.

---

## 41. Why are the prediction metrics nearly identical despite different weights?

The loss surface is relatively flat near the optimum.

Different parameter values can produce almost identical predictions.

---

## 42. Why use NumPy matrix multiplication instead of loops?

- Faster
- Cleaner
- Uses optimized BLAS/LAPACK libraries
- More memory efficient

---

## 43. Why store the loss history?

To visualize convergence and detect optimization problems.

---

## 44. How do you know Gradient Descent has converged?

- Loss stops decreasing
- Parameter updates become very small
- Gradient approaches zero

---

## 45. Why compare different learning rates?

To understand the trade-off between convergence speed and stability.

---

# Conclusion

This project demonstrates a complete implementation of Linear Regression from first principles using only NumPy. Every stage—from data preparation and forward propagation to gradient computation, optimization, evaluation, and comparison with analytical and library-based solutions—was implemented and analyzed manually. Through experiments on learning rate selection, feature scaling, and comparisons with the Normal Equation and Scikit-Learn, the project validates both the mathematical correctness of the implementation and the practical behavior of Gradient Descent. It provides a strong foundation for understanding more advanced machine learning algorithms that build upon these same optimization principles.
