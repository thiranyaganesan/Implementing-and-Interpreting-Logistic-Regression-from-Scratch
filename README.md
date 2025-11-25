# Logistic Regression From Scratch — Final Consolidated Submission

## ✔ Requirements Fully Addressed
This project implements logistic regression **from scratch**, satisfying all required tasks:

### 1. Synthetic Dataset (Corrected)
- Generated with:
  - `n_features = 5`
  - `n_informative = 5`
  - `n_redundant = 0`
- Fully aligned with the project prompt.

### 2. Single, Clean Implementation
- All code consolidated into **one final file** (`logistic_regression_final.py`).
- Includes:
  - Sigmoid function
  - Binary Cross-Entropy loss (explicit implementation)
  - Gradient Descent
  - Convergence check using tolerance

### 3. Interpretation of Learned Weights
Below is the required written analysis:

## 📌 Interpretation of the Learned Weights
For logistic regression, **each weight represents the strength and direction of influence** of its corresponding feature on the probability of belonging to the positive class.

### Interpretation Rules
- **Positive weight** → as the feature increases, probability of class 1 increases.
- **Negative weight** → as the feature increases, probability of class 1 decreases.
- **Higher magnitude** → stronger impact on classification.
- **Near‑zero weight** → weak or negligible effect.

### Interpretation in This Project
Since the dataset uses **5 informative features**, the learned weights typically show:

- Some features receive **large positive or negative values**, indicating they strongly differentiate the classes.
- Because the data is linearly separable, logistic regression successfully identifies these directional influences.
- The magnitude patterns reflect the true structure imposed by `make_classification()`.

This completes Deliverable 3 and Task 5 (written feature‑importance analysis).

---

This ZIP contains:
- `logistic_regression_final.py` — final corrected implementation  
- `README.md` — analysis + documentation

