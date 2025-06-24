# 🧠 Neural Network from Scratch

This project implements a basic **Multi-Layer Perceptron (MLP)** neural network from scratch using only **NumPy**, without relying on high-level libraries like TensorFlow or PyTorch. It is designed to classify handwritten digits from the **MNIST dataset**.

## 📌 Features

- Implements core neural network components:
  - Forward pass
  - Backpropagation
  - Loss computation (Cross-Entropy)
  - Weight updates via Stochastic Gradient Descent (SGD)
- Supports variable layer sizes and activation functions (e.g., ReLU, Tanh)
- Trains on MNIST dataset with basic data loading and preprocessing
- Tracks accuracy and loss over training epochs


## 📊 Results

The neural network was trained on the MNIST dataset for 10 epochs. Below are the training and validation results:

| Epoch | Training Loss | Training Accuracy | Validation Accuracy |
|-------|---------------|-------------------|---------------------|
| 1     | 330.71        | 89.78%            | 90.9%               |
| 2     | 229.45        | 93.32%            | 93.3%               |
| 3     | 178.94        | 94.76%            | 93.8%               |
| 4     | 143.60        | 96.08%            | 94.4%               |
| 5     | 129.89        | 96.46%            | 94.3%               |
| 6     | 106.29        | 97.12%            | 94.7%               |
| 7     | 92.55         | 97.68%            | 94.9%               |
| 8     | 76.73         | 98.20%            | 95.4%               |
| 9     | 65.05         | 98.74%            | 96.0%               |
| 10    | 56.36         | 98.84%            | 95.7%               |

**Final Test Accuracy**: **93.8%**

> ✅ **Note:** The highest validation accuracy (96.0%) was achieved at **epoch 9**. Training beyond this point resulted in a slight decrease in validation accuracy (95.7% at epoch 10), suggesting mild overfitting. Early stopping at epoch 9 may yield optimal generalization performance.


