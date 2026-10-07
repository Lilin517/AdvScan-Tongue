import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score

def compute_mAP(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ap_per_class = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            ap = average_precision_score(y_true[:, c], y_pred[:, c])
            ap_per_class.append(ap)
    if not ap_per_class:
        return 0.0
    return float(np.mean(ap_per_class))

def compute_f1_per_class(y_true: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    return {'f1_macro': float(f1_score(y_true, y_pred_bin, average='macro', zero_division=0)), 'f1_micro': float(f1_score(y_true, y_pred_bin, average='micro', zero_division=0))}

def compute_auc_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    auc_per_class = []
    for c in range(y_true.shape[1]):
        try:
            if y_true[:, c].sum() > 0 and (y_true[:, c] == 0).sum() > 0:
                auc = roc_auc_score(y_true[:, c], y_pred[:, c])
                auc_per_class.append(auc)
        except ValueError:
            pass
    if not auc_per_class:
        return {'auc_macro': 0.0}
    return {'auc_macro': float(np.mean(auc_per_class))}

def compute_exact_match(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    matches = np.all(y_true == y_pred_bin, axis=1)
    return float(matches.mean())

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray=None) -> dict:
    C = y_true.shape[1]
    if thresholds is None:
        thresholds = np.ones(C) * 0.5
    y_pred_bin = (y_pred >= thresholds).astype(np.float32)
    mAP = compute_mAP(y_true, y_pred)
    f1_dict = compute_f1_per_class(y_true, y_pred_bin)
    auc_dict = compute_auc_per_class(y_true, y_pred)
    emr = compute_exact_match(y_true, y_pred_bin)
    return {'mAP': mAP, 'f1_macro': f1_dict['f1_macro'], 'f1_micro': f1_dict['f1_micro'], 'auc_macro': auc_dict['auc_macro'], 'exact_match': emr, 'num_classes': C}
