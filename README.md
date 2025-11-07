# CS 376 Homework 4: Cell Detection with CNN

## Overview
In this assignment, you will implement and train a Convolutional Neural Network (CNN) in PyTorch to detect cells in microscopy images at pixel level. The training will use pseudo-labels generated from blob detection, and evaluation will be against ground truth.

## Learning Objectives
- Design and implement a CNN architecture for pixel-level classification
- Understand and use blob detection algorithms for pseudo-label generation
- Train deep learning models with proper loss functions and optimization
- Evaluate model performance using precision, recall, and F1-score metrics

## Dataset Structure
```
DATA/
├── train/
│   ├── cells/          # Training cell images (256x256 PNG)
│   ├── dots/           # Training ground truth dot images (for evaluation only, do not used training!)
│   └── pseudo_labels/  # Generated your pseudo-labels here - use only them for training! 
├── val/
│   ├── cells/          # Validation cell images (256x256 PNG)
│   ├── dots/           # Validation ground truth dot images (for evaluation!)
│   ├── pseudo_labels/  # Generated your pseudo-labels here - use only them for training! 
│   ├── gt/             # Ground truth coordinates (*.txt files) (for evaluation!)
│   ├── pred/           # Predicted coordinates (*.txt files) (Here you save your predictions!)
│   └── results/        # Visualization images Visualization would be saved here automatically
└── checkpoints/        # Saved model checkpoints
```

## Implementation Tasks

### 1. CNN Architecture (`CellDetectionNetwork` class)
**TODO: Implement the `__init__` and `forward` methods**

### 2. Data Loaders (`data_loaders` function)
**TODO: Create data loaders for training and validation**

- **Goal**: Set up PyTorch DataLoaders with proper transforms
- **Steps**:
  1. Get paths to training and validation cells images
  2. Generate pseudo-labels for training and validation data - this plays as GT!
  3. Return trainloader and valloader

**Implementation Hints**:
```python
# Get image paths
train_cell_images = sorted(glob.glob(os.path.join(train_path, 'cells', '*.png')))
val_cell_images = sorted(glob.glob(os.path.join(val_path, 'cells', '*.png')))

# Generate pseudo-labels
train_pseudo_labels = generate_pseudo_labels(train_cells_path, train_pseudo_path)
val_pseudo_labels = generate_pseudo_labels(val_cells_path, val_pseudo_path)

# Create datasets and dataloaders
train_dataset = CellDetectionDataset(train_images, train_labels, train_transform)
val_dataset = CellDetectionDataset(val_images, val_labels, val_transform)
```

### 3. Training Loop (`train` function)
**TODO: Implement complete training pipeline**

- **Goal**: Train the CNN model with proper loss function and optimization
- **Requirements**:
  1. Define loss function
  2. Define optimizer 
  3. Implement training loop with forward pass, loss calculation, backpropagation
  4. Add validation loop to monitor performance
  5. Save trained model checkpoint


### 4. Model Evaluation (`evaluate_only` function)
**TODO: Implement model evaluation and coordinate extraction**

- **Goal**: Evaluate trained model and extract cell coordinates
- **Steps**:
  1. Load trained model from checkpoint
  2. Run inference on validation data
  3. Extract coordinates from prediction maps
  4. Save coordinates to text files

**Implementation Hints**:
```python
# Load model
model = CellDetectionNetwork().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Run inference
with torch.no_grad():
    for data, _ in valloader:
        output = model(data)
        # Extract coordinates from output
        # Save to text files
```

## Provided Helper Functions

The following functions are already implemented and ready to use:

### `load_and_save_ground_truth(val_folder, save_dir="gt")`
- Extracts ground truth coordinates from dot images
- Saves coordinates to text files in "x y" format
- Automatically creates output directories

### `load_coordinates_from_txt(file_path)`
- Loads coordinates from text files
- Returns list of (y, x) coordinate tuples
- Handles file parsing and error checking

### `visualize_results_from_text(val_folder, output_dir)`
- Creates visualizations comparing GT and predicted detections
- Shows original image, GT dots, predicted dots, and comparison
- Saves visualization images as PNG files

### `calculate_metrics_from_files(gt_dir, pred_dir, output_dir)`
- Calculates precision, recall, and F1-score
- Uses distance-based matching (15 pixel threshold)
- Saves detailed evaluation results to file

## Running the Code

### Training Mode
```bash
python cell_detection.py --training --train_folder DATA/train --val_folder DATA/val
```

### Evaluation Only Mode
```bash
python cell_detection.py --model_path DATA/checkpoints/cell_detection_model.pth --val_folder DATA/val
```

### Full Pipeline (Training + Evaluation)
```bash
python cell_detection.py --training --train_folder DATA/train --val_folder DATA/val
```

## Expected Outputs

1. **Model Checkpoint**: `DATA/checkpoints/cell_detection_model.pth`
2. **Ground Truth Coordinates**: `DATA/val/gt/*.txt` (x y format)
3. **Predicted Coordinates**: `DATA/val/pred/*.txt` (x y format)
4. **Visualizations**: `DATA/val/results/*_detection.png`
5. **Evaluation Metrics**: `evaluation_results.txt`

## Evaluation Metrics

The system calculates:
- **Precision**: TP / (TP + FP) - How many predicted cells are correct
- **Recall**: TP / (TP + FN) - How many ground truth cells were found
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Harmonic mean
- **Threshold**: 15 pixels (prediction within 15 pixels of GT is considered correct)

## Submission Requirements

### Code Implementation
- Complete all TODO sections in the provided template
- Ensure your CNN architecture outputs correct dimensions
- Implement proper coordinate extraction and saving

### File Structure
```
cell_detection/
├── DATA/
│   └── checkpoints/
│       └── cell_detection_model.pth 
├── cell_detection.py          # Your completed implementation
├── blob_detection.py          # Blob detection functions implementation
└── common.py                  # Common utility functions
```

Good luck with your implementation! 🚀
