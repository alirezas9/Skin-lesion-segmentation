import numpy as np
from torch.nn import BCEWithLogitsLoss
# loss

#jaccard
def jaccard_loss(y_pred, y_true, axis=None, smooth=1e-5):
    
    intersect = np.sum(y_pred * y_true, axis=axis)
    union = np.sum(y_pred, axis=axis) + np.sum(y_true, axis=axis) - intersect
    jaccard = 1 - (intersect + smooth) / (union + smooth)
    
    return np.mean(jaccard)

#MSE
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

#BCE
def bce(y_pred, y_true):
    return BCEWithLogitsLoss(y_pred, y_true)
