import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

def compute_metrics(probs, targets, threshold=0.5):
    """
    Computes all blueprint-required metrics for medical image segmentation.
    
    Args:
        probs (torch.Tensor or np.ndarray): Probabilities, shape [B, 1, H, W]
        targets (torch.Tensor or np.ndarray): Ground truth binary mask, shape [B, 1, H, W]
        threshold (float): Threshold to binarize probabilities.
        
    Returns:
        dict: A dictionary of computed metrics.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    # Flatten arrays
    probs = probs.flatten()
    targets = targets.flatten()
    
    preds = (probs > threshold).astype(np.uint8)
    targets = targets.astype(np.uint8)
    
    # Fast calculation for TP, TN, FP, FN
    # Using confusion_matrix on millions of pixels is very slow. 
    # We use boolean operations instead.
    TP = np.sum((preds == 1) & (targets == 1))
    TN = np.sum((preds == 0) & (targets == 0))
    FP = np.sum((preds == 1) & (targets == 0))
    FN = np.sum((preds == 0) & (targets == 1))
    
    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)  # Also called Recall
    specificity = TN / (TN + FP + 1e-8)
    
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
    
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    
    # AUC-ROC
    # Subsampling might be needed if memory errors occur, but for flattened arrays it's usually fine
    # or we can compute it on a subset if it's too large, but 512x512 = 262,144 elements is quick
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        # Happens if only one class is present in targets
        auc = 0.5
        
    return {
        "Dice": dice,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "AUC-ROC": auc,
        "IoU": iou
    }
