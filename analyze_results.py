#!/usr/bin/env python
"""Analyze detection results to understand overdetection issues"""

import numpy as np

results = {
    'TP': 2855,
    'FP': 2289,
    'FN': 741,
    'Precision': 0.5550,
    'Recall': 0.7939,
    'F1': 0.6533,
    'Total_GT': 3596,
    'Total_Pred': 5144
}

print("=" * 60)
print("CELL DETECTION RESULTS ANALYSIS")
print("=" * 60)
print(f"\nBasic Metrics:")
print(f"  Precision: {results['Precision']*100:.1f}% ({results['TP']} TP / {results['Total_Pred']} total predictions)")
print(f"  Recall: {results['Recall']*100:.1f}% ({results['TP']} TP / {results['Total_GT']} GT cells)")
print(f"  F1 Score: {results['F1']*100:.1f}%")
print(f"\nDetection Counts:")
print(f"  True Positives: {results['TP']}")
print(f"  False Positives: {results['FP']} (overdetections)")
print(f"  False Negatives: {results['FN']} (missed cells)")
print(f"  Total Predictions: {results['Total_Pred']}")
print(f"  Total Ground Truth: {results['Total_GT']}")
print(f"\nProblem Analysis:")
print(f"  False Positive Rate: {results['FP']/results['Total_Pred']*100:.1f}% of predictions are wrong")
print(f"  False Negative Rate: {results['FN']/results['Total_GT']*100:.1f}% of GT cells missed")
print(f"  Overdetection Factor: {results['Total_Pred']/results['Total_GT']:.2f}x (predicting {results['Total_Pred']-results['Total_GT']} more than GT)")
print(f"\nPer-Image Averages (assuming 20 validation images):")
print(f"  Average FP per image: {results['FP']/20:.1f}")
print(f"  Average TP per image: {results['TP']/20:.1f}")
print(f"  Average FN per image: {results['FN']/20:.1f}")
print(f"\n" + "=" * 60)
print("ROOT CAUSE ANALYSIS:")
print("=" * 60)
print("""
1. THRESHOLD TOO LOW:
   - Current adaptive thresholding uses 0.2-0.3 when model outputs are low
   - This causes many false positives from background noise
   - Need to increase thresholds to reduce FP while maintaining recall

2. MIN COMPONENT SIZE TOO SMALL:
   - Currently min_component_size = 3 pixels
   - This catches very small noise blobs
   - Should increase to filter out noise (e.g., 5-7 pixels)

3. NO MORPHOLOGICAL FILTERING:
   - Removed morphological operations because they were "too aggressive"
   - But light opening/closing could remove small noise without hurting real cells
   - Need to add back light morphological operations

4. NO LOCAL MAXIMA FILTERING:
   - Currently using center_of_mass for all components
   - Should use local maxima in prediction map to find true cell centers
   - This would prevent multiple detections from one blob

5. PSEUDO-LABEL QUALITY:
   - Pseudo-labels use radius=4px which is very small
   - Model might be learning to detect things that aren't cells
   - May need to review pseudo-label generation parameters
""")
print("=" * 60)
print("RECOMMENDED IMPROVEMENTS:")
print("=" * 60)
print("""
1. INCREASE THRESHOLDS:
   - Use higher base threshold (0.4-0.5 instead of 0.2-0.3)
   - Use 80th-85th percentile instead of 70th
   - Cap minimum threshold higher (e.g., 0.3 instead of 0.2)

2. INCREASE MIN COMPONENT SIZE:
   - Change min_component_size from 3 to 5-7 pixels
   - This filters out small noise blobs

3. ADD LIGHT MORPHOLOGICAL OPERATIONS:
   - Use opening (erosion + dilation) with small kernel (3x3)
   - This removes small noise without affecting real cells

4. USE LOCAL MAXIMA INSTEAD OF CENTER OF MASS:
   - Find local maxima in prediction map within each component
   - Use the maximum value point as cell center
   - This prevents multiple detections from one blob

5. ADD NON-MAXIMUM SUPPRESSION:
   - If two detections are too close (< 10-15 pixels), keep only the one with higher confidence
   - This prevents duplicate detections

6. CONSIDER CONFIDENCE-BASED FILTERING:
   - Only keep detections where the prediction value at center is above a threshold
   - E.g., require pred_map[center] > 0.4 even after thresholding
""")

