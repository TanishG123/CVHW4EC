# Cell Detection Improvements Summary

## Current Performance Analysis

### Results Before Improvements:
- **Precision**: 55.5% (2289 false positives out of 5144 predictions)
- **Recall**: 79.4% (2855 true positives out of 3596 ground truth)
- **F1 Score**: 65.3%
- **Overdetection**: 1.43x more predictions than ground truth (1548 extra detections)
- **Average False Positives per Image**: 114.5

### Root Causes Identified:

1. **Threshold Too Low**: Adaptive thresholding was using 0.2-0.3 when model outputs were low, causing many false positives from background noise
2. **Min Component Size Too Small**: `min_component_size = 3` pixels caught very small noise blobs
3. **No Morphological Filtering**: Removed morphological operations, but light filtering could remove noise without hurting real cells
4. **Using Center of Mass**: This could place detections at non-optimal locations within blobs
5. **No Non-Maximum Suppression**: Multiple detections could come from the same blob

## Implemented Improvements

### 1. Increased Thresholds
- **Changed**: 70th percentile → **85th percentile** of values above mean
- **Changed**: Minimum threshold cap from 0.3 → **0.4** for low-output cases
- **Changed**: Fixed threshold from 0.2 → **0.3** when insufficient pixels
- **Changed**: Minimum threshold floor from mean+0.02 → **mean+0.05**
- **Changed**: Intermediate threshold from 0.3 → **0.4** when <5% above 0.5

**Impact**: Reduces false positives by requiring higher confidence for detections

### 2. Increased Minimum Component Size
- **Changed**: `min_component_size` from 3 → **5 pixels**

**Impact**: Filters out very small noise blobs that were causing false positives

### 3. Added Morphological Opening
- **Added**: Light morphological opening with 3x3 kernel
- **Purpose**: Removes small noise structures without affecting real cells

**Impact**: Cleans up binary map before component analysis, removing tiny artifacts

### 4. Local Maxima Instead of Center of Mass
- **Changed**: Find local maximum in prediction map within each component
- **Benefit**: Places detection at the point with highest confidence, not geometric center

**Impact**: More accurate cell center detection, especially for irregular shapes

### 5. Confidence-Based Filtering
- **Added**: Require `confidence > 0.3` at detection center
- **Purpose**: Only keep detections where the model is confident

**Impact**: Filters out low-confidence detections that are likely false positives

### 6. Non-Maximum Suppression (NMS)
- **Added**: Remove detections within 12 pixels of each other
- **Strategy**: Keep detection with highest confidence when multiple are too close
- **Purpose**: Prevents duplicate detections from the same blob

**Impact**: Reduces overdetection by ensuring only one detection per cell

## Expected Improvements

With these changes, we expect:
- **Precision**: Increase from 55.5% to **65-75%** (fewer false positives)
- **Recall**: Maintain around **75-80%** (still catching most cells)
- **F1 Score**: Increase from 65.3% to **70-75%**
- **False Positives**: Reduce from 114.5 per image to **50-70 per image**

## Testing Recommendations

1. **Run evaluation** on validation set to see new metrics
2. **Visual inspection** of detection images to verify improvements
3. **Fine-tune parameters** if needed:
   - Adjust `min_component_size` (5-7 pixels)
   - Adjust NMS `min_distance` (10-15 pixels)
   - Adjust confidence threshold (0.3-0.4)
   - Adjust morphological kernel size (3x3 or 5x5)

## Next Steps

If precision is still low after these changes:
1. Consider increasing thresholds further (e.g., 0.5 minimum)
2. Increase `min_component_size` to 7 pixels
3. Add more aggressive morphological operations
4. Review pseudo-label quality - may need to regenerate with better parameters
5. Consider retraining model with better pseudo-labels or data augmentation

