import numpy as np
from sklearn.metrics import roc_curve, auc

def optimal_threshold(y_true, y_pred):
    y_true = np.concatenate(y_true)
    y_pred = y_pred.ravel()

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Calculate Youden's J statistic
    J = tpr - fpr

    # Find the optimal threshold
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold