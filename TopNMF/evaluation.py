import numpy as np
from scipy.ndimage import label

def compute_iou(pred_vector, true_mask, threshold_ratio=0.05):
    """
    Calculate Intersection over Union (IoU) between a predicted basis vector 
    and a binary ground truth mask.
    """
    pred_vector = pred_vector.reshape(true_mask.shape)
    # Convert continuous basis vector to binary mask using threshold
    threshold = threshold_ratio * np.max(pred_vector)
    pred_mask = (pred_vector > threshold).astype(int)
    
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_jamo_iou(V, masks, threshold_ratio=0.05):
    """
    Find the best matching basis vector for each ground truth Jamo mask
    and compute the IoU score.
    """
    results = {}
    used_indices = set()
    
    for jamo, mask in masks.items():
        best_iou = -1.0
        best_index = -1
        
        # Greedy matching: find best basis for this mask
        for i in range(V.shape[0]):
            iou = compute_iou(V[i], mask, threshold_ratio)
            if iou > best_iou:
                best_iou = iou
                best_index = i
        
        results[jamo] = {"iou": best_iou, "basis_index": best_index}
    
    # Calculate Average IoU
    avg_iou = np.mean([info['iou'] for info in results.values()])
    return results, avg_iou

def extract_consonant_vowel_masks(image, threshold=0.5):
    """
    Splits a character image into consonant (left) and vowel (right) parts
    using connected component labeling and centroid coordinates.
    """
    binary = (image > threshold).astype(int)
    labeled, num_features = label(binary)
    
    consonant_mask = np.zeros_like(binary, dtype=int)
    vowel_mask = np.zeros_like(binary, dtype=int)
    mid_col = image.shape[1] / 2
    
    for comp_id in range(1, num_features + 1):
        comp = (labeled == comp_id)
        centroid_col = np.column_stack(np.where(comp))[:, 1].mean()
        
        if centroid_col < mid_col:
            consonant_mask[comp] = 1
        else:
            vowel_mask[comp] = 1
            
    return consonant_mask, vowel_mask
