import numpy as np
from collections import Counter

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################

class KNN:
    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        
        '''
        1. Change the 2D list into numpy array -> np.array(features)
        2. Change the label into numpy array
        '''

        self.train_features = np.array(features)
        self.train_labels = np.array(labels)


    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds the k nearest neighbours in the training set.
        It needs to return a list of labels of these k neighbours. When there is a tie in distance, 
		prioritize examples with a smaller index.
        :param point: List[float]
        :return:  List[int]
        """
        distance = []
        index = []
        maxD = 0
        final = []

        training_set = self.train_features
        label_y = self.train_labels

        #print("k:", self.k)

        for row_index in range(training_set.shape[0]):
            result = self.distance_function(point, training_set[row_index])
            distance.append((result, row_index))

        distance.sort()

        neighbor_labels = []
        for i in range(min(self.k, len(distance))):
            idx = distance[i][1]
            neighbor_labels.append(self.train_labels[idx])
            
        return neighbor_labels
            
        #     if len(distance) < self.k:
        #         index.append(row_index)
        #         distance.append(result)
        #     else:
        #         maxIndex = distance.index(max(distance))
        #         if max(distance) > result:
        #             distance.pop(maxIndex)
        #             distance.append(result)
        #             index.pop(maxIndex)
        #             index.append(row_index)
        
        # #print("Index:", index)
        
        # for num in index:
        #     final.append(int(label_y[num]))
        
        # #print("neighbor Result:", final)

        # #print("Length", len(index))
        
        # return final
            

	# TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need to process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predicted label for that testing data point (you can assume that k is always a odd number).
        Thus, you will get N predicted label for N test data point.
        This function needs to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """

        prediction = []
        features = np.array(features)
        
        for row_index in range(len(features)):
            result = self.get_k_neighbors(features[row_index])
            #print("prediction Result:", result)
            counter = Counter(result)

            most_common_label = counter.most_common(1)[0][0]

            prediction.append(int(most_common_label))

            #print("row index", row_index)
        
        return prediction
            

if __name__ == '__main__':
    print(np.__version__)
