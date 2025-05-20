import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    '''
        tags are the states, words are the observation
    '''
    dict1 = {}
    dict2 = {}

    word2idx = {word: index for index, word in enumerate(unique_words)}
    tag2idx = {tag: index for index, tag in enumerate(tags)}

    S = len(tags)
    word_len = len(unique_words)

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################

    '''
        Pi is the probability of the initial state 
        B is the emission. At state i what we observation do we see
    '''
    for sentence in train_data:
        first_tag = sentence.tags[0]
        pi[tag2idx[first_tag]] += 1

        for tag, word in zip(sentence.tags, sentence.words):
            if word in word2idx:
                B[tag2idx[tag], word2idx[word]] += 1 
        
        for time in range(1, len(sentence.tags)):
            prev_tag = sentence.tags[time-1]
            current_tag = sentence.tags[time]
            A[tag2idx[prev_tag], tag2idx[current_tag]] += 1 
        
    total_states = np.sum(pi)
    pi = pi / np.sum(pi)

    for state_index in range(S):
        row_sum = np.sum(A[state_index])
        if row_sum > 0:
            A[state_index] = A[state_index] / row_sum
    
    for state_index in range(S):
        row_sum = np.sum(B[state_index])
        if row_sum > 0:
            B[state_index] = B[state_index] / row_sum
       
        

    

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################

    for sentence in test_data:
        observation = sentence.words
        observation_index = []

        for word in observation:
            if word not in model.obs_dict:
                new_index = len(model.obs_dict)
                model.obs_dict[word] = new_index

                num_records = len(model.B)
                new_column = np.ones((num_records,1)) * 1e-6
                model.B = np.hstack((model.B, new_column))
            
            #observation_index.append(model.obs_dict[word])
        
        path_index = model.viterbi(observation)

        #path_tag = [model.find_key(model.state_dict, index) for index in path_index]

        tagging.append(path_index)

    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
