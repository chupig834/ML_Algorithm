# Machine Learning Algorithms Collection 

Welcome to my Machine Learning Algorithms Collection! This repository hosts a set of projects where I have implemented core machine learning algorithms from scratch and explored their behavior on real datasets

# Algorithms Implemented
- **K-Nearest Neighbors (KNN):**
  - **F1 Score**
  - **Distance Function (Euclidean_Distance, Minkowski_Distance, Cosine_Similarity_Distance)**
  - **KNN Class (Train, Get_k_neighbors, Predict)**
  - **Data Transformation (Normalization, Min-Max Scaling)**
  - **Hypterparameter Tuning**
    
- **Regression:**
  - **Mean Square Error**
  - **Linear Regression**
  - **Regularized Linear Regression**
  - **Tune Regularization Parameter**
  - **Polynomial Regression**
    
- **Linear Classifier:**
  - **Binary Classification (Perceptron and Logistic with gradient decent)**
  - **Multiclass Classification (Implement SGD and GD on multiclass logistic)**

- **Neural Nets:**
  - **Linear Layer: forward pass, backward pass**
  - **Activation Function - RELU: forward pass, backward pass**
  - **Activation Funcation - tanh: forward pass, backward pass**
  - **Dropout: forward pass, backward pass**
  - **Mini-batch Stochastic Gradient Descent**

- **Decision Trees Boosting**
  - **Decision Tress and Random Forest**
  - **AdaBoost**

- **K-Means**
  - **K-Menas++**
  - **K-Means**
  - **Classification with K-Means**
  - **Image Compression with K-Means**

- **PCA (Word Embedding)**
  - **Compute top-m eigenvalues/vectors, variance explained via PCA approximation**
  - **Build Embedding matrix, find and query similar words**
  - **Identify top-k influencing words per eigenvector**
  - **Project words along semantic direction**
  - **Solve analogies via vector arithmetic and nearest neighbors**
  - **Evaluate synonym vs antonym similarity accuracy**

- **Hidden Markov Models (HMM)**
  - **Forward: Compute forward messages**
  - **Backward: Compute backward messages**
  - **Sequence_prob: Probability of observing a sequence**
  - **Posterior_prob: State posterior at a given time step**
  - **Likelihood_prob: Transition likelihood between states at a time step**
  - **Viterbi: Find most likely hidden state path**
  - **Model Training: Estimate initial, transition, and emission probabilities from data**
  - **Speech Tagging: POS tagging with Viterbi and unknown-word handling**
 
- **Transformer (Decoder-Only)**
  - **Head.forward: Project inputs to Q/K/V, apply scaled dot-product attention with causal masking and softmax, compute weighted sums**
  - **MultiHeadAttention.forward: Run multiple attention heads in parallel and concatenate their outputs**
  - **Block.forward: Apply layer normalization, self-attention and feedforward sub-layers with residual connections**
  - **BigramLanguageModel.forward: Combine token and positional embeddings; pass through transformer blocks; apply final normalization and vocabulary projection; compute cross-entropy loss**
  - **Likelihood_prob: Transition likelihood between states at a time step**
  - **Viterbi: Find most likely hidden state path**
  - **Model Training: Estimate initial, transition, and emission probabilities from data**
  - **Speech Tagging: POS tagging with Viterbi and unknown-word handling**
 
- **Reinforcement Learning (Q-Learning)**
  - **ReplayBuffer: Store and sample experience with replay**
  - **QLearningAgent: Epsilon-greedy exploration, Q-value updates with sampled batches**
