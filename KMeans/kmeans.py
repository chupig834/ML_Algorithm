import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: a list of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    p = generator.randint(0, n) #this is the index of the first center
    #############################################################################
    # TODO: implement the rest of Kmeans++ initialization. To sample an example
	# according to some distribution, first generate a random number between 0 and
	# 1 using generator.rand(), then find the the smallest index n so that the 
	# cumulative probability from example 1 to example n is larger than r.
    #############################################################################

    '''
    1. Compute Distance with the centroidadd
        || x - c| || ^2
    2. Sum all the distance and find the probability propotion to the distance to find the next centroid
        K-Means++ random pick the next center where each point has a chance of being picked proportional to its squared distance
        Complete this process until we have all the n_clusters 
    '''

    center_index = [p]


    while len(center_index) < n_cluster:
        centers_so_far = x[center_index]
        dists = np.linalg.norm(x[:, np.newaxis] - centers_so_far, axis=2) ** 2
        min_distances = np.min(dists, axis=1)
        # min_distances = []
        # for point in x : 
        #     squared_distances = [np.sum((x[c] - point) ** 2) for c in center_index]
        #     min_distances.append(min(squared_distances))

        #min_distances = np.array(min_distances)
        sum_distance = np.sum(min_distances)
        prob = min_distances / sum_distance
        cumulative_probs = np.cumsum(prob)

        r = generator.rand()
        next_index = np.searchsorted(cumulative_probs, r)
        center_index.append(next_index)

    centers = center_index
    
    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################

        centers = x[self.centers]
        for iteration in range(self.max_iter):
            label = []
            dists = np.linalg.norm(x[:, np.newaxis] - centers, axis=2) ** 2
            label = np.argmin(dists, axis=1)
            # for point in x:
            #     squared_distances = [np.sum((c - point) ** 2 ) for c in centers]
            #     index = np.argmin(squared_distances)
            #     label.append(index)
            
            new_centers = np.zeros((self.n_cluster, x.shape[1]))  
            for num in range(self.n_cluster):
                assigned_points  = x[np.array(label) == num]

                if len(assigned_points) > 0:
                    new_centers[num] = np.mean(assigned_points, axis = 0)
                else:
                    new_centers[num] = x[self.generator.randint(0, len(x))]
            
            if np.allclose(centers, new_centers, atol=self.e):
                break

            centers = new_centers

        label = np.array(label)
        return centers, label, iteration + 1



class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################

        '''
        In Kmeans, we are training a classifier, so we want each cluster (from Kmean) to represent a class
        1. For each cluster k, look at which data points were assigned to it (label == k)
        2. Look at the true lables y of those points 
        3. Use majority vote to decide 
            Most points in this cluster are class 2. We will label this cluster as class 2

        '''

        kmean = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, label, num_iter = kmean.fit(x, centroid_func)

        centroid_labels = np.zeros(self.n_cluster, dtype=int)

        for num in range(self.n_cluster):
            cluster_y = y[label == num]
            freq = {}

            if len(cluster_y) == 0:
                centroid_labels[num] = -1
            
            else:
                for lbl in cluster_y:
                    if lbl in freq:
                        freq[lbl] += 1
                    else:
                        freq[lbl] = 1
                
                sorted_freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
                most_common_label = sorted_freq[0][0]

                centroid_labels[num] = most_common_label            
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################

        '''
        1. Calculate the distance between X and the centroids
        2. find the closest centroids 
        3. Have the closest centroids, find the label 
        '''
        label = []
        dists = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2) ** 2
        nearest = np.argmin(dists, axis=1)
        # for num in x:
        #     squared_distances = [np.sum((c - point) ** 2 ) for c in centers]
        #     index = np.argmin(squared_distances)
        #     label.append(self.centroid_labels[index])

        return self.centroid_labels[nearest]


def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    
    H, W, C = image.shape
    new_image = image.reshape(-1, 3) 

    dists = np.linalg.norm(new_image[:, np.newaxis] - code_vectors, axis=2) ** 2 

    nearest = np.argmin(dists, axis=1)  

    sq_diff = code_vectors[nearest]

    final_image = sq_diff.reshape(H, W, 3)

    quantized_image = final_image

    return quantized_image