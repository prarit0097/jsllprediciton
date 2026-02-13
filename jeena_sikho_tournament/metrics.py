import numpy as np


def accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def f1_score(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def mae(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def coverage(y_true, lower, upper) -> float:
    y_true = np.array(y_true)
    lower = np.array(lower)
    upper = np.array(upper)
    return float(((y_true >= lower) & (y_true <= upper)).mean())
