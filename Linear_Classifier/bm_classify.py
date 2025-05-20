import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################

        '''
        Linear Classification: Prediction = sign(w_transpose * x + b)
            if w_transpose * x + b > 0, predict +1   
            if w_transpose * x + b < 0, predict -1   
        Perceptron Loss = max(0, -yi(w_transpose * xi + b))
            if wrong prediction, the perception loss > 0
            if right prediction, the perception loss = 0

        Step 1: Since y is 0 or 1, we will covert labels from {0, 1} -> {-1, 1}
            y_binary = 2 * y - 1
        
        Step 2: create a new array the same shape as w, but filled with zerios

        Step 3: Gradient Decent on Loss function. If -yi(w_transpose * xi + b) > 0, compute gradient

        Step 4: 
            Calculate the change in loss based on weight for each row of the data
            Calculate the change in loss based on bias for each row of the data

        '''

        y_binary = 2 * y - 1 #if y = 0, it will become -1, if y =1, it will become 1 

        for num in range(max_iterations):
            grad_w = np.zeros_like(w)
            grad_b = 0

            #y_binary = 2 * y - 1 #if y = 0, it will become -1, if y =1, it will become 1 
            for numRecord in range(N):
                margin = y_binary[numRecord] * (w.T @ X[numRecord] + b) # margin is the distance between the decision boundary

                if margin <= 0:
                    grad_w = grad_w + -y_binary[numRecord] * X[numRecord] #Derivative of loss function respect to w is negative
                    grad_b = grad_b + -y_binary[numRecord] #Derivative of loss function respect to b is negative
            
            grad_w = grad_w / N 
            grad_b = grad_b / N 

            w = w - step_size * grad_w
            b = b - step_size * grad_b  

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################

        '''
        Logistic Loss = ln(1 + exp(-y(w * x + b)))
            When y(w * x + b) is large positive, exp(-y(w * x + b)) becomes very small -> loss is near 0 
            When y(w * x + b) is large negative, exp(-y(w * x + b)) becomes very large -> loss is large

        Step 1: Since y is 0 or 1, we will covert labels from {0, 1} -> {-1, 1}
            y_binary = 2 * y - 1
        
        Step 2: create a new array the same shape as w, but filled with zerios

        Step 3: Gradient Decent on Loss function. If y(w * x + b) < 0, compute gradient

        Step 4: 
            Calculate the change in loss based on weight for each row of the data
            Calculate the change in loss based on bias for each row of the data

        '''
        
        y_binary = 2 * y - 1 

        for num in range(max_iterations):
            grad_w = np.zeros_like(w)
            grad_b = 0

            for numRecord in range(N):

                # margin = y_binary[numRecord] * (w @ X[numRecord] + b)

                # if margin <= 0:
                #     grad_w = grad_w + -(y_binary[numRecord] * X[numRecord]) / (1 + np.exp(y_binary[numRecord] * (w @ X[numRecord] + b)))
                #     grad_b = grad_b + -y_binary[numRecord] / (1 + np.exp(y_binary[numRecord] * (w @ X[numRecord] + b)))

                z = np.dot(w, X[numRecord]) + b
                sigmoid_z = sigmoid(z)

                grad_w = grad_w + (sigmoid_z - y[numRecord]) * X[numRecord]
                grad_b = grad_b + (sigmoid_z - y[numRecord])
            
            grad_w = grad_w / N
            grad_b = grad_b / N

            w = w - step_size * grad_w
            b = b - step_size * grad_b 
            
    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################

    '''
    Sigmoid is an activation function 
    Funcation - 1 / (1 + exp(-z))
    '''

    value = 1 / ( 1 + np.exp(-z))
    
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    
    '''
    1. Linear Score  z = w . x + b 
    2. predicted y = activation function (z)
    3. if predict y >= 0.5 predict 1 || if predicted y < 0.5, predict 0 
        sigmoid function threshold of 0.5 in binary classification
    '''
    z = X @ w + b

    pred_y = sigmoid(z)
    preds = np.where(pred_y < 0.5, 0, 1)
    

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################

            '''
            In multiclass classification, each class will have a score z (logit)
                1. Randomly pick an index (Starting point)
                2. Compute z scores
                3. Apply softmax
                    Find the predicted probabilities for classes
                4. Create one-hot label
                5. Compute gradient
                    y_onehot -> find the true class
                    prob - y = prediction error
                6. Update weights and bias
            '''			

            xi = X[n] #Shape (D,)
            yi = y[n]

            #print("xi shape", xi.shape)
            #print("w shape", w.shape)
            z = xi @ w.T + b # Z score for each class (sahpe (C,))

            z_stable = z - np.max(z)
            prob = np.exp(z_stable) / np.sum(np.exp(z_stable))

            y_onehot = np.zeros(C)
            y_onehot[yi] = 1 #yi is the class index. yi -> (0, 1, ... C-1)

            grad_w = np.outer(xi, (prob - y_onehot)) 
            grad_b = prob - y_onehot

            #print("grad_w shape", grad_w.shape)

            w -= step_size * grad_w.T
            b -= step_size * grad_b

      
    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
       
        '''
        1. Compute logits z = x @ W + b  || Shape (N X C)
            z is the score for by feature for each class
        
        2. Softmax 
            find z_stable (by row)
            calculate the prob (by row)
        
        3. one-hot
            y will be a shape of N X C (Each record should have a result for a class)
        
        4. Calculate gradient
        5. Update parameters

        '''

        for iteration in range(max_iterations):

            z = X @ w.T + b

            z_stable = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_stable)
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            y_onehot = np.zeros((N,C))
            y_onehot[np.arange(N), y] = 1 #np.arange(N) create an array from 0 to N-1. Using this to assign 1

            error = probs - y_onehot
            grad_w = (X.T @ error) / N 
            grad_b = np.mean(error, axis=0)

            w -= step_size * grad_w.T
            b -= step_size * grad_b

        
    else:
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################

    '''
    1. Compute the raw score z (a score for each feature)

    2. Apply softmax to get class probability
    '''
    
    z = X @ w.T + b 

    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    preds = np.argmax(prob, axis=1)

    assert preds.shape == (N,)
    return preds




        