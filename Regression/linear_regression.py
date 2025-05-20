import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    err = None

    '''
    1. y_predicted -> X @ w. This will return a num_samples x 1 numpy array. 
    2. sum the difference between y and y_predicted
    3. divide the sum by num_samples 
    '''

    y_predict = X @ w 
    diff = y_predict - y 
    squared_diff = diff ** 2
    total_diff = np.sum(squared_diff)

    err = total_diff / len(y)
    
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		

  '''
  The formula for w ->  w = (X_T * X) ^ -1 * X_T * y 
    How we derived this is -> We are trying to minimizing the MSE. Taking the derivative of the MSE equation, w will have a closed form like this 

  1. Add an intercet terms into X  
  2. First, calculate the X_T 
  3. Apply X_T with the parameters to calculate w
  '''

  w = None

  #X_withIntercept = np.hstack([np.ones((X.shape[0], 1)), X])

  X_transpose = X.T

  XtX = X_transpose @ X 

  # print("X_tranpose shape", X_transpose.shape)
  # print("X_shape", X_withIntercept.shape)
  # print("XtX", XtX.shape)

  XtX_inverse = np.linalg.inv(XtX)

  w = XtX_inverse @ X_transpose @ y

  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################	

    '''
    The formula for w ->  w = (X_T * X + lambd * I) ^ -1 * X_T * y  
    '''	
    w = None

    lambd = float(lambd)

    X_transpose = X.T
    XtX = X_transpose @ X

    I = np.eye(X.shape[1])

    insideInv = XtX + lambd * I
    XtX_inverse = np.linalg.inv(insideInv)

    w = XtX_inverse @ X_transpose @ y

    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    bestlambda = None

    result = []

    #Start from -14 to -1 (not including - 1)
    for num in range(-14, 1):
      lambd = 2 ** num
      w = regularized_linear_regression(Xtrain, ytrain, lambd)
      score = mean_square_error(w, Xval, yval)

      result.append([lambd, score])
    
    result.sort(key=lambda x: x[1])

    print(result)
    bestlambda = result[0][0]

    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################

    mapped_X = X.copy()

    for p in range(2, p + 1):
      mapped_X = np.hstack((mapped_X, X ** p))
    
    return mapped_X

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

