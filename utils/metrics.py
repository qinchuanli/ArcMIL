"""
Evaluation Metrics for ArcMIL
"""

import numpy as np
from typing import Dict


def compute_metrics(y_true, y_pred, y_prob=None):
    """计算完整评估指标"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    auc = None
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}
    }


def print_metrics(metrics, title="Results"):
    """打印格式化结果"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    
    if isinstance(metrics.get('auc'), float):
        print(f"  AUC:          {metrics['auc']:.4f}")
    else:
        print(f"  AUC:          N/A")
    
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  F1-Score:     {metrics['f1']:.4f}")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    n_samples = 100
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=n_samples)
    y_prob = np.random.uniform(0, 1, size=n_samples)
    y_pred = (y_prob >= 0.5).astype(int)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Test Metrics")
