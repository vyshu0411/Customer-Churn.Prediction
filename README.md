# Customer Churn Prediction — ANN vs Random Forest

Predicting bank customer churn using a Neural Network and Random Forest,
with a focus on handling class imbalance correctly.

## Dataset
Churn_Modelling.csv — 10,000 bank customers, ~20% churn rate.

## Results

| Metric    | ANN (baseline) | ANN (class-weighted) | Random Forest |
|-----------|-----------------|------------------------|----------------|
| Accuracy  | 0.8535          | 0.8315                 | 0.8630         |
| Precision | 0.8032          | 0.5764                 | 0.7904         |
| Recall    | 0.3710          | 0.6486                 | 0.4447         |
| F1 Score  | 0.5076          | 0.6104                 | 0.5692         |
| ROC-AUC   | 0.8598          | 0.8591                 | 0.8553         |

## Key finding
Accuracy alone was misleading on this imbalanced dataset. Applying class
weighting to the ANN improved recall from 37% to 65% (F1: 0.51 → 0.61),
with almost no change in ROC-AUC — meaning the imbalance, not model
capacity, was the real issue. In a churn scenario, missing a churner
usually costs more than a false alarm, so the class-weighted ANN is
the preferred model here despite lower raw accuracy than Random Forest.

## Tech stack
Python, TensorFlow/Keras, scikit-learn, pandas, matplotlib
