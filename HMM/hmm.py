from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################

        '''
        A = [
            [0.7, 0.3],
            [0.4, 0.6]
        ]
        x = current state, y = next state 
        A[0][0] = 0.7  p(next state is 0 | current state is 0 )
        alpha = giveing the obeservation from 1 to t and end up at state s at time t, this is the probability
        1. First, I need to calculate the initial state 
        2. I have to create a recursion 
        '''

        #pi has the initial state at t = 0
        #O[0] it has the first observation
        for stateIndex in range(S):
            alpha[stateIndex, 0] = self.pi[stateIndex] * self.B[stateIndex, O[0]]

        #Now I need to create a recursion

        for obsIndex in range(1, L):
            for nextStateIndex in range(S):
                sum_answer = 0
                for prevStateIndex in range(S):
                    sum_answer += alpha[prevStateIndex, obsIndex - 1] * self.A[prevStateIndex, nextStateIndex]
                alpha[nextStateIndex, obsIndex] = sum_answer * self.B[nextStateIndex, O[obsIndex]]

        return alpha


    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################

        '''
        If we are in state s, at time t, what is the probability of seeing all the future observations from t+1 to T
        '''

        beta = np.zeros((S, L))

        for state in range(S):
            beta[state, L-1] = 1
        
        for time in range(L - 2, -1, -1):
            for current_state in range(S):
                sum_answer = 0 
                for next_state in range(S):
                    transition_prob = self.A[current_state][next_state]
                    emission_prob = self.B[next_state][O[time+1]]
                    future_beta = beta[next_state][time+1]
                    sum_answer += transition_prob * emission_prob * future_beta
                beta[current_state][time] = sum_answer

        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################

        alpha = self.forward(Osequence)

        latest_time = len(alpha[1]) - 1
        total_prob = np.sum(alpha[:, latest_time])

        return total_prob


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        #gamma = np.zeros((len(alpha[0]), len(alpha[1])))
        S, L = alpha.shape
        gamma = np.zeros((S, L))

        #Summing the probablity of reaching each state at the final time step, having observed the whole sequence. 
        '''
        observation ['walk', 'shop', 'clean']
        alpha[1][1] = Prob. of seeing the full sequence and ending in rainy. 
            observed 'walk' to 'shop'
            and I am in state 1 
        '''
        obs_all_prob = np.sum(alpha[:, -1])

        if obs_all_prob == 0:
            return gamma

        for time in range(L):
            for state in range(S):
                joint_prob = alpha[state, time] * beta[state, time]
                gamma[state, time] = joint_prob / obs_all_prob
        
        return gamma
    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        Observation = self.find_item(Osequence)
        total_obs_prob = np.sum(alpha[:, -1])

        for time in range(L - 1):
            for current_state in range(S):
                for next_state in range(S):
                    transition_prob = self.A[current_state][next_state]
                    emission_prob = self.B[next_state][Observation[time + 1]]
                    alphaState = alpha[current_state][time]
                    betaState = beta[next_state][time+1]
                    joint_prob = alphaState * transition_prob * emission_prob * betaState
                    prob[current_state, next_state, time] = joint_prob / total_obs_prob
        
        return prob


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################

        observations = self.find_item(Osequence)
        num_states = len(self.pi)
        num_steps = len(observations)

        delta = np.zeros((num_states, num_steps))
        backtrack_index = np.zeros((num_states, num_steps), dtype=int)  #Making sure it is an int

        for state in range(num_states):
            backtrack_index[state, 0] = 0
            delta[state, 0] = self.pi[state] * self.B[state][observations[0]]

        for time in range(1, num_steps):
            for current_state in range(num_states):
                max_prob = 0
                best_prev = 0
                for prev_state in range(num_states):
                    prob = delta[prev_state, time - 1] * self.A[prev_state][current_state]
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_state
                
                delta[current_state, time] = max_prob * self.B[current_state][observations[time]]
                backtrack_index[current_state, time] = best_prev
        
        path_index = np.zeros(num_steps, dtype=int)
        last_state = np.argmax(delta[:,-1])
        path_index[-1] = last_state

        for time in range(num_steps -2, -1, -1):
            path_index[time] = backtrack_index[path_index[time + 1], time + 1]
        
        path = [self.find_key(self.state_dict, index) for index in path_index]

        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
