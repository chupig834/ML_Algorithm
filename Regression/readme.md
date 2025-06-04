# Regression

### Part 1.1 Mean Square Error 

Given a linear model parameter $w$ and a data set specified by $X$ and $y$, compute the mean square error.

- `TODO 1` Complete `def mean_square_error(w, X, y)` in `linear_regression.py`

### Part 1.2 Linear Regression 

Based on what we discussed in the lectures, implement linear regression with no regularization using a training data set $(X,y)$ and return the model parameter $w$. You do not need to worry about non-invertible matrices for this part. You should use numpy inverse function directly (the whole implementation can in fact be as simple as one or two lines of code).

- `TODO 2` Complete `def linear_regression_noreg(X, y)` in `linear_regression.py`

Once you finish Part 1.1 and Part 1.2, you should be able to run `linear_regression_test.py` and test these two parts. Read the output, and check your dimension of $w$ (should be 12 in this case) and MSE for training, evaluation and testing datasets (should all be between 0.5\~0.6).

### Part 1.3 Regularized Linear Regression 

To prevent overfitting, we now add L2 regularization with a regularization parameter $\\lambda$.

- `TODO 3` Complete `def regularized_linear_regression(X, y, lambd)` in `linear_regression.py`

Once you finish this part, run linear\_regression\_test.py again. You should see a better MSE for the test data.

### Part 1.4 Tune the regularization parameter

Now try to tune the regularization parameter among the following 15 values: $2^{-14}, 2^{-13}, \\ldots, 2^{-1}, 2^{0}=1$. More specifically, for each value, use the given training set and the `regularized_linear_regression` function you implemented in Part 1.3 to train a model, then use the given validation set and the `mean_square_error` function you implemented in Part 1.1 to evaluate the model. Finally return the best value corresponding to the model with the lowest mean square error.

- `TODO 4` Complete `def tune_lambda(Xtrain, ytrain, Xval, yval)` in `linear_regression.py`

Once you finish this part, run linear\_regression\_test.py again. The best lambda happens to be $2^{-14}$ in this case.

### Part 1.5 Polynomial regression

In the lectures, we discussed polynomial regression for the one-dimensional case. Here, you will implement a simplified version of the polynomial regression for high-dimensional data, by only raising each feature to some power and ignoring "crossed" features. For example, if we have a two-dimensional feature $(x\_1, x\_2)$, then for a 2-degree polynomial regression, we will map this feature to $(x\_1, x\_2, x\_1^2, x\_2^2)$ (note that there is no "crossed" feature $x\_1x\_2$).

To reuse previous code for linear regression, you task is simply to take a dataset $X$ and an integer $p$, and return the augmented data set $\[X, X^2, \\ldots, X^p]$ where $X^i$ stands for element-wise power.

- `TODO 5` Complete `def mapping_data(X, power)` in `linear_regression.py`

Once you finish this part, run linear\_regression\_test.py again. You should see that the training MSE is getting smaller as we use a larger degree, but at the same time the testing MSE is increasing due to overfitting.
