# K-Nearest Neighbours for Binary Classification (50 points)

## 📚 Background

In this task, we will use three different distance functions to compare two points `x` and `x'` in `ℝⁿ`:

- **Euclidean Distance**  
  `d(x, x') = sqrt( Σ (xᵢ - xᵢ')² )`

- **Minkowski Distance (p=3)**  
  `d(x, x') = ( Σ |xᵢ - xᵢ'|³ )^(1/3)`

- **Cosine Distance**  


To evaluate the model’s performance, we will use the **F1-score** (not error rate). Refer to: https://en.wikipedia.org/wiki/F1_score

> Note: Label `1` is considered positive, and label `0` is negative.

---

## ✅ Part 1.1 — F1 Score and Distance Functions

Implement the following in `utils.py`:

- `f1_score`
- `class Distance`
- `euclidean_distance`
- `minkowski_distance`
- `cosine_similarity_distance`

---

## ✅ Part 1.2 — KNN Class

Implement the following in `knn.py`:

- `class KNN`
- `train`
- `get_k_neighbors`
- `predict`

---

## ✅ Part 1.3 — Data Transformation

We will explore how data transformation affects model performance.

### 1. Normalization of Feature Vectors

Given a vector `x`, the normalized vector is:


See: https://en.wikipedia.org/wiki/Feature_scaling

Implement in `utils.py`:

- `class NormalizationScaler`
  - `__call__`
- `class MinMaxScaler`
  - `__call__`

---

## ✅ Part 1.4 — Hyperparameter Tuning

Tune the following hyperparameters:

- The value of `k`
- The choice of distance function
- The data transformation scheme

Implement in `utils.py`:

- `class HyperparameterTuner`
  - `tuning_without_scaling`
  - `tuning_with_scaling`

---

## ✅ Part 1.5 — Testing with `test.py`

Nothing to implement here. Use `test.py` to validate your code.

If your implementation is correct, it will display confirmation messages.

You may also uncomment this line in `data.py` to shuffle data:

```python
np.random.shuffle(data)


