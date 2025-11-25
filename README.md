Final Deliverable — Logistic Regression from Scratch
=====================================================

Files included:
- logistic_regression_final.py : single, consolidated implementation.
- interpretation.md : numeric interpretation of learned weights and concrete examples.

Dataset generation
------------------
Used sklearn.datasets.make_classification with:
- n_samples=600, n_features=5, n_informative=5, n_redundant=0, random_state=42

Model training summary
----------------------
Final binary cross-entropy loss: 0.361563
Bias (intercept): 1.6045
Baseline probability at mean feature values: 0.5320

Learned weights (numeric)
-------------------------
Feature_1: weight=0.3984, approx marginal effect on P(class=1) at mean = 0.0992
Feature_2: weight=-0.3039, approx marginal effect on P(class=1) at mean = -0.0757
Feature_3: weight=0.6170, approx marginal effect on P(class=1) at mean = 0.1536
Feature_4: weight=-1.5402, approx marginal effect on P(class=1) at mean = -0.3835
Feature_5: weight=-1.2585, approx marginal effect on P(class=1) at mean = -0.3133

Interpretation (concrete)
-------------------------
- Each weight is the change in log-odds for a one-unit increase in the feature, holding others fixed.
- To convert a weight to an approximate change in predicted probability at the baseline (mean feature vector) use:
  marginal_effect ≈ p*(1-p)*weight, where p is baseline probability.
- Using baseline p=0.5320, the approximate marginal effects are shown above.

Concrete example (Feature_4)
----------------------------
- Weight for Feature_4 = -1.5402 (negative).
- A one-unit increase in Feature_4 decreases the log-odds by -1.5402.
- At baseline probability 0.5320, the predicted probability changes by approximately -0.3835 (i.e., a -38.35% absolute change).

Notes
-----
- The dataset is synthetic; feature names are Feature_1..Feature_5 with no external semantic labels.
- Interpretation links sign and magnitude to direction and strength; larger |weight| => larger effect.
- This README provides numeric mapping back to the actual final run (weights and marginal effects above).
