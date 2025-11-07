import os
import shutil
import numpy as np
from cell_detection import generate_pseudo_labels
from common import read_img
from scipy.ndimage import label

# Clean up old test (ignore errors)
if os.path.exists('test_pseudo'):
    try:
        shutil.rmtree('test_pseudo')
    except:
        pass

# Generate with new parameters
os.makedirs('test_pseudo', exist_ok=True)
generate_pseudo_labels('DATA/val/cells', 'test_pseudo', min_sigma=0.5, max_sigma=8.0, num_scales=15, threshold=0.05)

# Analyze results
pseudo = read_img('test_pseudo/101pseudo.png', greyscale=True)
pseudo_binary = (pseudo > 0.05).astype(float) if pseudo.max() > 1 else (pseudo > 0.05).astype(float)
labeled, num = label(pseudo_binary)

print(f'New pseudo (8px radius): {num} blobs, {np.sum(pseudo_binary > 0)} pixels')

# Compare with GT
dots = read_img('DATA/val/dots/101dots.png', greyscale=False)
dots_max = np.max(dots, axis=2) if len(dots.shape) == 3 else dots
dots_binary = (dots_max > 0.5).astype(float)
labeled_dots, num_dots = label(dots_binary)

print(f'GT: {num_dots} cells, {np.sum(dots_binary > 0)} pixels')
print(f'Blob count ratio: {num}/{num_dots} = {num/num_dots:.2f}x')
print(f'Coverage ratio: {np.sum(pseudo_binary > 0) / np.sum(dots_binary > 0):.1f}x')

