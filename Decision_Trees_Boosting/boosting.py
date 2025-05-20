import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Part 2: Implementation of AdaBoost with decision trees as weak learners

class AdaBoost:
  def __init__(self, n_estimators=60, max_depth=10):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.betas = []
    self.models = []
    
  def fit(self, X, y):
    ###########################TODO#############################################
    # In this part, please implement the adaboost fitting process based on the 
    # lecture and update self.betas and self.models, using decision trees with 
    # the given max_depth as weak learners

    # Inputs: X, y are the training examples and corresponding (binary) labels
    
    # Hint 1: remember to convert labels from {0,1} to {-1,1}
    # Hint 2: DecisionTreeClassifier supports fitting with a weighted training set

    '''
    1. Convert y from {0,1} to {-1, 1}
    2. Create the w (creating sample weights)
    3. Create a weak learner using the current sample weight
    4. Predict labels on the training data using the stump model 
    5. Compute the weighted error 
    6. Computer the beta
      Smaller error, larger beta, more influence
    7. Update sample weight
      Misclassified samples -> weight increase 
      Correct classified -> weight decrease
    '''

    y_new = np.where(y==0, -1, 1)
    N = X.shape[0]


    w = np.ones(N) / N

    for t in range(self.n_estimators):
      #Train weak learner with current sample weights
      stump = DecisionTreeClassifier(max_depth=self.max_depth)
      stump.fit(X, y, sample_weight=w)

      y_pred = stump.predict(X)
      y_pred = np.where(y_pred == 0, -1, 1)

      error = np.sum(w * (y_pred != y_new))

      if error == 0:
        beta = 1e10
      else:
        beta = 0.5 * np.log((1-error) / error)

      w = w * np.exp(-beta * y_new * y_pred)
      w = w / np.sum(w)

      self.models.append(stump)
      self.betas.append(beta)

    return self
    
  def predict(self, X):
    ###########################TODO#############################################
    # In this part, make prediction on X using the learned ensemble
    # Note that the prediction needs to be binary, that is, 0 or 1.

    '''
    1. Get prediction from each weak learner
    2. Multiply each prediction by its weight beta
    3. sum all weighted prediction 
    4. Take the sign of the result
    5. Convert prediction from -1, 1 to 0, 1
    '''

    y_pred_sum = np.zeros(X.shape[0])

    for index in range(len(self.models)):
      model = self.models[index]
      y_pred = model.predict(X)
      y_pred = np.where(y_pred == 0, -1, 1)

      y_pred_sum += self.betas[index] * y_pred

    
    final_pred = np.sign(y_pred_sum)
    
    preds = np.where(final_pred == 1, 1, 0)

    return preds
    
  def score(self, X, y):
    accuracy = accuracy_score(y, self.predict(X))
    return accuracy

