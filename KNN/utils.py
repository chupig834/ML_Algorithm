import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    assert len(real_labels) == len(predicted_labels)

    #F1 score = 2 * (Precision * Recall) / (Precision + Recall)
    #Precision = TP / (TP + FP)
    #Recall = TP / (TP + FN)
    
    '''
    1. I will set 3 variables for TP, FP, and FN. 
    2. Loop thru both lists and add the count for each variables 
    3. Use the formula above to calculate the F1 score
    '''

    TP = 0
    FP = 0
    FN = 0 

    for i in range (len(predicted_labels)):
        if predicted_labels[i] == 1 and real_labels[i] == 1:
            TP += 1
        elif predicted_labels[i] == 1 and real_labels[i] == 0:
            FP += 1
        elif predicted_labels[i] == 0 and real_labels[i] == 1:
            FN += 1
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return F1

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #A generalized version of Euclidean Distance. In this exercise, we will set p = 3
        finalD = 0

        for i in range (len(point1)):
            dist = abs(point1[i] - point2[i])
            dist3 = dist ** 3
            finalD = finalD + dist3

        minkD = finalD ** (1/3)

        return minkD

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        finalD = 0 

        for i in range (len(point1)):
            dist = point1[i] - point2[i]
            dist2 = dist ** 2
            finalD = finalD + dist2

        eucliD = finalD ** 0.5

        return eucliD

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        '''
        cosin_sim formula = A . B / magnitude (A) . magnitude (B)
        . -> Dot Product 
        mag(A) = (point1(1)^2 + point1(2)^2...) ^ 0.5 

        1. loop thru the points to calculate the dot product of the two Lists (point1, point2)
        2. Calculate the magnitude of A and B
        3. Use the formula to find cosin_sim
        '''
        magA = 0
        magB = 0
        finalDot = 0
        for i in range (len(point1)):
            num = point1[i] * point2[i]
            finalDot = finalDot + num

            magA = magA + point1[i] ** 2
            magB = magB + point2[i] ** 2
        
        if magA == 0 or magB == 0:
            cosSim = 0
        else:
            cosSim = finalDot / ((magA ** 0.5) * (magB ** 0.5))
        cosDist = 1 - cosSim
        
        return cosDist


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        score = []
        results = []

        
        for k in range(1, 30, 2):
            for name, func in distance_funcs.items():
                model = KNN(k=k, distance_function=func)
                model.train(x_train, y_train)
                predictions = model.predict(x_val)
                #print("y_val:", len(y_val))
                #print("predictions:", len(predictions))
                score = f1_score(y_val, predictions)
                results.append([k, name, score, func])
                #print("k w.o scaling", k)
                #print("name", name)
                #print("score", score)
        
        result_array = np.array(results, dtype=object)
        #print("result array", result_array)
        scores = result_array[:, 2].astype(float)

        max_value = np.max(scores)
        all_max_indices = np.where(scores == max_value)[0]
        tied_rows = result_array[all_max_indices]

        distance_priority = {'euclidean': 0, 'minkowski': 1, 'cosine_dist': 2}
        tied_rows_sorted = sorted(tied_rows, key=lambda row: (distance_priority[row[1]], row[0]))

        best_row = tied_rows_sorted[0]
        
        self.best_k = best_row[0]

        finalModel = KNN(self.best_k, best_row[3])
        finalModel.train(x_train, y_train)

        self.best_distance_function = best_row[1]
        self.best_model = finalModel
        self.best_scaler = None
    


    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        score = []
        results = []

        for k in range(1, 30, 2):
            for scaleName, scaleFunc in scaling_classes.items():
                scaler = scaleFunc()
                #print("Scale Name", scaleName)
                #print("x_train inside w. scaler", x_train[:,1])
                normalize_x_train = scaler(x_train)
                #print("First column:", [row[1] for row in normalize_x_train])
                normalize_x_val = scaler(x_val)
                for name, func in distance_funcs.items():
                    model = KNN(k=k, distance_function=func)
                    model.train(normalize_x_train, y_train)
                    predictions = model.predict(normalize_x_val)
                    #print("Prediction inside Tune w. Scaling", len(predictions), len(y_val))
                    score = f1_score(y_val, predictions)
                    results.append([k, name, score, func, scaleName, scaleFunc])

        result_array = np.array(results, dtype=object)
        #print("Result Array", result_array)     
        scores = result_array[:, 2].astype(float)

        max_value = np.max(scores)
        all_max_indices = np.where(scores == max_value)[0]
        tied_rows = result_array[all_max_indices]

        scaler_priority = {'min_max_scale': 0, 'normalize': 1}
        distance_priority = {'euclidean': 0, 'minkowski': 1, 'cosine_dist': 2}

        tied_rows_sorted = sorted(
            tied_rows,
            key=lambda row: (
                scaler_priority[row[4]],         # scaler name
                distance_priority[row[1]],       # distance function name
                row[0]                            # k
            )
        )
        best_row = tied_rows_sorted[0]

        self.best_k = best_row[0]
        self.best_distance_function = best_row[1]
        self.best_scaler = best_row[4]

        scaler_class = best_row[5]
        scaler = scaler_class()
        scaled_x_train = scaler(x_train)
        self.best_model = KNN(k=best_row[0], distance_function=best_row[3])
        self.best_model.train(scaled_x_train, y_train)

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        result = []

        for row in features:
            norm = sum(x ** 2 for x in row) ** 0.5
            if norm > 0:
                normalized_row = [x / norm for x in row]
            else:
                normalized_row = [0 for _ in row]  # Handle zero-vector
            result.append(normalized_row)

        return result



class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        # result = []
        # colArray = []

        # for col in range(len(features[0])):
        #     for row in range(len(features)):
        #         colArray.append(features[row][col])
            
        #     maxValue = max(colArray)
        #     minValue = min(colArray)

        #     for row in range(len(features)):
        #         features[row][col] = (features[row][col] - minValue) / (maxValue - minValue) if (maxValue - minValue > 0) else 0
            
        #     colArray = []
        
        # return features

        features = np.array(features)
        result = np.zeros_like(features, dtype=float)

        for col in range(features.shape[1]):
            col_values = features[:, col]
            max_val = np.max(col_values)
            min_val = np.min(col_values)
            denom = max_val - min_val
            if denom > 0:
                result[:, col] = (col_values - min_val) / denom
            # elif denom == 0:
            #     result[:, col] = 0  # or leave as 0
        
        return result.tolist()


        

