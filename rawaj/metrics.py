import math
import numpy as np

def _validate_inputs(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have equal lengths.")
    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty.")
    if np.any(np.isnan(scores)):
        raise ValueError("Scores cannot contain NaN values.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true values must be either 0 or 1.")
        
    return y_true, scores

def confusion_at(y_true, scores, threshold: float) -> dict:
    yt, s = _validate_inputs(y_true, scores)
    preds = s >= threshold
    
    tp = int(np.sum((yt == 1) & preds))
    fp = int(np.sum((yt == 0) & preds))
    tn = int(np.sum((yt == 0) & (~preds)))
    fn = int(np.sum((yt == 1) & (~preds)))
    
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    cm = confusion_at(y_true, scores, threshold)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def precision_at_k(y_true, scores, k: int) -> float:
    yt, s = _validate_inputs(y_true, scores)
    n = len(yt)
    
    if k <= 0:
        raise ValueError("k must be greater than 0.")
        
    k = min(k, n)
    
    # Stable sort to preserve original order on score ties
    order = np.argsort(-s, kind="mergesort")
    top_k_labels = yt[order[:k]]
    
    tp = np.sum(top_k_labels == 1)
    return float(tp / k)

def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    cm = confusion_at(y_true, scores, threshold)
    return float(cm["fp"] * cost_fp + cm["fn"] * cost_fn)

def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    yt, s = _validate_inputs(y_true, scores)
    
    # Candidates are every distinct score plus math.inf
    distinct_scores = np.unique(s)
    candidates = np.append(distinct_scores, math.inf)
    
    best_thresh = None
    min_cost = math.inf
    
    for thresh in candidates:
        cost = expected_cost(yt, s, thresh, cost_fp=cost_fp, cost_fn=cost_fn)
        if cost < min_cost:
            min_cost = cost
            best_thresh = thresh
        elif cost == min_cost:
            # Tie breaker: choose the HIGHEST threshold
            if best_thresh is None or thresh > best_thresh:
                best_thresh = thresh
                
    return float(best_thresh)

def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    yt, s = _validate_inputs(y_true, scores)
    distinct_scores = np.unique(s)
    
    valid_thresholds = []
    for thresh in distinct_scores:
        prec, _ = precision_recall_at(yt, s, thresh)
        if prec >= min_precision:
            valid_thresholds.append(thresh)
            
    if not valid_thresholds:
        return None
        
    # Return the LOWEST threshold among candidates with precision >= min_precision
    return float(min(valid_thresholds))