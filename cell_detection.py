#!/usr/bin/env python
# coding: utf-8

"""
CS 376 Homework 4: Cell Detection with CNN

In this assignment, you will implement and train a Convolutional Neural Network (CNN) 
in PyTorch to detect cells in microscopy images at pixel level. The training will use pseudo-labels 
generated from blob detection, and evaluation will be against ground truth.

=== HOW TO RUN ===
Training: python cell_detection.py --training --train_folder DATA/train --val_folder DATA/val
Evaluation: python cell_detection.py --model_path DATA/checkpoints/cell_detection_model.pth --val_folder DATA/val

=== FOLDER STRUCTURE ===
DATA/train/cells, DATA/train/dots
DATA/val/cells, DATA/val/dots, DATA/val/gt, DATA/val/pred, DATA/val/results
DATA/checkpoints

=== OUTPUTS ===
- Model: DATA/checkpoints/cell_detection_model.pth
- Coordinates: DATA/val/gt/*.txt, DATA/val/pred/*.txt (x y format)
- Metrics: evaluation_results.txt
"""

import time
import os
import glob
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

# Third-party imports
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid warnings
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary

# Local imports
from blob_detection import gaussian_filter, find_maxima_modified
from common import read_img, visualize_maxima, save_img

if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = 'cuda'
else:
    print("Using the CPU. Overall speed may be slowed down")
    device = 'cpu'

class CellDetectionNetwork(nn.Module):
    """CNN for cell detection with U-Net architecture"""
    def __init__(self):
        super().__init__()
        
        ##############################################################################
        # TODO: Design your own network, define layers here.                          #
        # Your solution should contain convolutional layers for cell detection.        #
        # The output should be a 2D map (H x W) where each pixel indicates           #
        # whether it's a cell (1) or not (0).                                        #
        # Refer to PyTorch documentations of torch.nn to pick your layers.           #
        # (https://pytorch.org/docs/stable/nn.html)                                  #
        # Some common choices: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout   #
        # If you have many layers, use nn.Sequential() to simplify your code         #
        ##############################################################################
        
        # Encoder-decoder architecture
        # Encoder: downsampling with batch normalization
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: upsampling with dropout
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.output = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        ##############################################################################
        # TODO: Implement the forward pass through your network                       #
        # Input: x is a batch of images of shape (batch_size, 1, H, W)              #
        # Output: should be of shape (batch_size, 1, H, W) with values 0-1          #
        ##############################################################################
        
        # Encoder 
        e1 = self.enc1(x) 
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        if u1.shape[2:] != e2.shape[2:]:
            u1 = F.interpolate(u1, size=e2.shape[2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)
        
        u2 = self.up2(d1)
        if u2.shape[2:] != e1.shape[2:]:
            u2 = F.interpolate(u2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)
        
        # Output
        out = torch.sigmoid(self.output(d2))
        
        return out


def load_and_save_ground_truth(val_folder, save_dir="gt"):
    """Extract ground truth coordinates from dot images and save to text files"""
    dots_path = os.path.join(val_folder, 'dots')
    output_dir = os.path.join(val_folder, save_dir)
    print(f"Loading ground truth from: {dots_path}")
    print(f"Saving ground truth coordinates to: {output_dir}")
    
    # Get all dots images
    dots_images = glob.glob(os.path.join(dots_path, '*.png'))
    dots_images.sort()
    
    if not dots_images:
        print(f"Warning: No dots images found in {dots_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for dots_img_path in tqdm(dots_images, desc="Processing ground truth"):
        # Read dots image
        dots_img = read_img(dots_img_path)
        img_name = os.path.basename(dots_img_path).replace('dots', 'cell')
        
        # Handle RGB ground truth images
        if len(dots_img.shape) == 3:
            dots_img = np.mean(dots_img, axis=2)  # Convert RGB to grayscale
        
        gt_map_np = dots_img / 255.0 if dots_img.max() > 1 else dots_img
        gt_centers_np = np.where(gt_map_np > 0)  # Count all positive dot pixels
        
        # Convert to list of (y, x) coordinates
        if len(gt_centers_np[0]) > 0:
            gt_centers = [(int(gt_centers_np[0][i]), int(gt_centers_np[1][i])) for i in range(len(gt_centers_np[0]))]
        else:
            gt_centers = []
        
        # Save ground truth coordinates to text file
        gt_file = os.path.join(output_dir, f'{img_name.replace(".png", ".txt")}')
        with open(gt_file, 'w') as f:
            for y, x in gt_centers:
                f.write(f"{x} {y}\n")
    
    print(f"Processed and saved ground truth for {len(dots_images)} images")

def generate_pseudo_labels(cell_images_path, output_path, min_sigma=0.5, max_sigma=8.0, num_scales=15, threshold=0.05):
    """
    Generate pseudo-labels using blob detection with optimized parameters.
    """
    os.makedirs(output_path, exist_ok=True)
    
    cell_images = sorted(glob.glob(os.path.join(cell_images_path, '*.png')))
    pseudo_label_paths = []
    
    k = (max_sigma / min_sigma) ** (1.0 / (num_scales - 1))
    
    for cell_img_path in tqdm(cell_images, desc="Generating pseudo-labels"):
        # Check if pseudo-label already exists
        img_name = os.path.basename(cell_img_path)
        pseudo_path = os.path.join(output_path, img_name.replace('cell', 'pseudo'))
        
        if os.path.exists(pseudo_path):
            # Skip generation if pseudo-label already exists
            pseudo_label_paths.append(pseudo_path)
            continue
        
        # Read cell image
        cell_img = read_img(cell_img_path, greyscale=True)
        
        # Normalize to 0-1
        if cell_img.max() > 1:
            cell_img = cell_img / 255.0
        
        # Build scale space
        scale_space = []
        for i in range(num_scales):
            sigma = min_sigma * (k ** i)
            filtered = gaussian_filter(cell_img, sigma)
            scale_space.append(filtered)
        
        scale_space = np.stack(scale_space, axis=2)
        
        # Find maxima
        maxima = find_maxima_modified(scale_space, k_xy=3, k_s=1)
        
        # Create binary mask from maxima
        H, W = cell_img.shape
        pseudo_label = np.zeros((H, W), dtype=np.float32)
        
        # Filter maxima by intensity
        intensity_threshold = np.percentile(cell_img, 50)
        
        filtered_maxima = []
        for y, x, s in maxima:
            if 0 <= y < H and 0 <= x < W:
                if cell_img[y, x] >= intensity_threshold:
                    filtered_maxima.append((y, x, s))
        
        # Apply non-maximum suppression
        min_distance = 10
        suppressed_maxima = []
        for i, (y1, x1, s1) in enumerate(filtered_maxima):
            keep = True
            for j, (y2, x2, s2) in enumerate(suppressed_maxima):
                dist = np.sqrt((y1 - y2)**2 + (x1 - x2)**2)
                if dist < min_distance:
                    # keep  brighter one
                    if cell_img[y1, x1] > cell_img[y2, x2]:
                        # Current  brighter remove previous
                        suppressed_maxima.pop(j)
                        break
                    else:
                        # Previous  brighter skip current
                        keep = False
                        break
            if keep:
                suppressed_maxima.append((y1, x1, s1))
        
        # Small radius
        radius = 4
        
        for y, x, s in suppressed_maxima:
                
                y_min, y_max = max(0, int(y - radius)), min(H, int(y + radius + 1))
                x_min, x_max = max(0, int(x - radius)), min(W, int(x + radius + 1))
                
                # circular blob with smooth Gaussian
                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                dist_sq = (yy - y)**2 + (xx - x)**2
                blob = np.exp(-dist_sq / (2 * (radius/2)**2))
                pseudo_label[y_min:y_max, x_min:x_max] = np.maximum(
                    pseudo_label[y_min:y_max, x_min:x_max], blob
                )
        
        pseudo_label = (pseudo_label > threshold).astype(np.float32)
        
        save_img(pseudo_label, pseudo_path)
        pseudo_label_paths.append(pseudo_path)
    
    return pseudo_label_paths


class CellDetectionDataset(Dataset):
    """Dataset for cell detection"""
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = read_img(self.image_paths[idx], greyscale=True)
        if img.max() > 1:
            img = img / 255.0
        
        # Load label
        label = read_img(self.label_paths[idx], greyscale=True)
        if label.max() > 1:
            label = label / 255.0
        
        # Convert to tensors
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        label_tensor = torch.FloatTensor(label).unsqueeze(0)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            label_tensor = self.transform(label_tensor)
        
        return img_tensor, label_tensor


def data_loaders(train_path, val_path):
    """Create data loaders with automatic pseudo-label generation"""
    ##############################################################################
    # TODO: Create data loaders for training and validation                       #
    # 1. Get paths to training cell images and validation dot images              #
    # 2. Generate pseudo-labels for training and validation data                  #
    # 3. Create and return DataLoader objects                                     #
    ##############################################################################

    # Get image paths
    train_cell_path = os.path.join(train_path, 'cells')
    val_cell_path = os.path.join(val_path, 'cells')
    
    train_cell_images = sorted(glob.glob(os.path.join(train_cell_path, '*.png')))
    val_cell_images = sorted(glob.glob(os.path.join(val_cell_path, '*.png')))
    
    # Get pseudo-label paths
    train_pseudo_path = os.path.join(train_path, 'pseudo_labels')
    val_pseudo_path = os.path.join(val_path, 'pseudo_labels')
    
    # Check if pseudo-labels already exist for all images
    train_pseudo_labels = []
    val_pseudo_labels = []
    
    # training pseudo-labels
    all_train_exist = True
    for img_path in train_cell_images:
        img_name = os.path.basename(img_path)
        pseudo_path = os.path.join(train_pseudo_path, img_name.replace('cell', 'pseudo'))
        if os.path.exists(pseudo_path):
            train_pseudo_labels.append(pseudo_path)
        else:
            all_train_exist = False
            break
    
    if not all_train_exist or len(train_pseudo_labels) != len(train_cell_images):
        print("Generating training pseudo-labels (skipping existing ones)...")
        train_pseudo_labels = generate_pseudo_labels(train_cell_path, train_pseudo_path)
    else:
        print(f"Using existing training pseudo-labels ({len(train_pseudo_labels)} files)")
        train_pseudo_labels = []
        for img_path in train_cell_images:
            img_name = os.path.basename(img_path)
            pseudo_path = os.path.join(train_pseudo_path, img_name.replace('cell', 'pseudo'))
            train_pseudo_labels.append(pseudo_path)
    
    #  validation pseudo-labels
    all_val_exist = True
    for img_path in val_cell_images:
        img_name = os.path.basename(img_path)
        pseudo_path = os.path.join(val_pseudo_path, img_name.replace('cell', 'pseudo'))
        if os.path.exists(pseudo_path):
            val_pseudo_labels.append(pseudo_path)
        else:
            all_val_exist = False
            break
    
    if not all_val_exist or len(val_pseudo_labels) != len(val_cell_images):
        print("Generating validation pseudo-labels (skipping existing ones)...")
        val_pseudo_labels = generate_pseudo_labels(val_cell_path, val_pseudo_path)
    else:
        print(f"Using existing validation pseudo-labels ({len(val_pseudo_labels)} files)")
        val_pseudo_labels = []
        for img_path in val_cell_images:
            img_name = os.path.basename(img_path)
            pseudo_path = os.path.join(val_pseudo_path, img_name.replace('cell', 'pseudo'))
            val_pseudo_labels.append(pseudo_path)
    
    #  datasets
    train_dataset = CellDetectionDataset(train_cell_images, train_pseudo_labels, transform=None)
    val_dataset = CellDetectionDataset(val_cell_images, val_pseudo_labels, transform=None)
    
    #  data loaders
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Created data loaders: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    return trainloader, valloader


def train(model, trainloader, valloader, num_epoch=20):
    """Train CNN with Binary Cross Entropy Loss and save checkpoint"""
    print("Start training...")
    
    ##############################################################################
    # TODO: Implement the training loop                                           #
    # 1. Define loss function                                                     #
    # 2. Define optimizer                                                         #
    # 3. Implement training loop with forward pass, loss calculation,             #
    #    backpropagation, and optimization                                        #
    # 4. You can add add validation loop to monitor performance                   #
    # 5. Save the trained model checkpoint                                        #
    ##############################################################################
    
    #  loss 
    criterion = nn.BCELoss()
    
    #  optimizer, weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    
    #  learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epoch} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            #  forward 
            optimizer.zero_grad()
            outputs = model(images)
            
            #  loss
            loss = criterion(outputs, labels)
            
            #  backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        
        #  validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(valloader, desc=f"Epoch {epoch+1}/{num_epoch} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        #  learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        #  best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join('DATA', 'checkpoints', 'cell_detection_model.pth')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_epochs': num_epoch,
                'batch_size': 8,
            }, model_save_path)
            print(f"Model saved to {model_save_path}")
    
    print("Training completed!")


def evaluate_only(model_path, valloader, val_folder, save_dir='pred'):
    """Evaluate model and save predicted coordinates to text files
    
    Args:
        model_path: Path to saved model checkpoint
        valloader: Validation data loader
        val_folder: Path to validation folder
        save_dir: Directory name for saving predictions (default: 'pred')
    """
    ##############################################################################
    # TODO: Implement model evaluation and prediction saving                     #
    # 1. Create save_dir inside val_folder to store prediction text files        #
    # 2. Load trained model from checkpoint using model_path                     #
    # 3. Set model to evaluation mode                                            #
    # 4. For each image in valloader:                                            #
    #    - Run model inference to get prediction map                             #
    #    - Save coordinates to text file:                                        #
    #      - Format: "x y" (one coordinate pair per line)                        #
    #      - Save as: val_folder/save_dir/image_name.txt                         #
    #      - Coordinates should be integers                                      #
    ##############################################################################
    
    from scipy.ndimage import label
    
    #  output directory
    pred_dir = os.path.join(val_folder, save_dir)
    os.makedirs(pred_dir, exist_ok=True)
    
    #  model
    model = CellDetectionNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    #  model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print("Running inference on validation set...")
    
    #  image filenames
    cells_path = os.path.join(val_folder, 'cells')
    image_files = sorted([f for f in os.listdir(cells_path) if f.endswith('.png')])
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(valloader, desc="Evaluating")):
            images = images.to(device)
            outputs = model(images)
            
            #  each image in batch
            for i in range(images.shape[0]):
                #  prediction map
                pred_map = outputs[i, 0].cpu().numpy()  # (H, W)
                
                # Check how many pixels are above 0.5
                pixels_above_05 = np.sum(pred_map > 0.5)
                total_pixels = pred_map.size
                pct_above_05 = pixels_above_05 / total_pixels
                
                if pred_map.max() < 0.3 or pct_above_05 < 0.005:  # Very low outputs
                    #  adaptive thresholding
                    above_mean = pred_map[pred_map > pred_map.mean()]
                    if len(above_mean) > 100:
                        #  75th percentile of values above mean
                        threshold = np.percentile(above_mean, 75)
                        threshold = min(threshold, 0.35)
                    else:
                        threshold = 0.25
                    threshold = max(threshold, pred_map.mean() + 0.03)
                elif pct_above_05 < 0.10:
                    #  moderate threshold
                    threshold = 0.38
                elif pct_above_05 < 0.20:
                    #  standard threshold
                    threshold = 0.42
                else:
                    #  higher threshold
                    threshold = 0.48
                
                binary_map = (pred_map > threshold).astype(np.uint8)
                
                # Post-processing
                from scipy.ndimage import binary_opening
                #  small 3x3 kernel
                kernel = np.ones((3, 3), dtype=np.uint8)
                binary_map = binary_opening(binary_map, structure=kernel).astype(np.uint8)
                labeled, num_features = label(binary_map)
                
                if num_features > 0:
                    #  very small components
                    min_component_size = 4
                    coordinates = []
                    confidences = []
                    
                    for comp_id in range(1, num_features + 1):
                        component_mask = (labeled == comp_id)
                        component_size = np.sum(component_mask)
                        
                        if component_size >= min_component_size:
                            #  local maximum
                            component_pred = pred_map * component_mask
                            max_idx = np.unravel_index(np.argmax(component_pred), pred_map.shape)
                            y, x = max_idx
                            confidence = component_pred[y, x]
                            
                            #  confidence is above threshold
                            if confidence > 0.35 and not (np.isnan(x) or np.isnan(y)):
                                coordinates.append((int(round(x)), int(round(y))))
                                confidences.append(confidence)
                    
                    # Non-maximum suppression
                    if len(coordinates) > 1:
                        min_distance = 12
                        suppressed_coords = []
                        suppressed_confs = []
                        
                        #  confidence (highest first)
                        sorted_indices = np.argsort(confidences)[::-1]
                        
                        for idx in sorted_indices:
                            x, y = coordinates[idx]
                            conf = confidences[idx]
                            keep = True
                            
                            #  distance to already kept detections
                            for kept_x, kept_y in suppressed_coords:
                                dist = np.sqrt((x - kept_x)**2 + (y - kept_y)**2)
                                if dist < min_distance:
                                    #  close to an existing detection, skip
                                    keep = False
                                    break
                            
                            if keep:
                                suppressed_coords.append((x, y))
                                suppressed_confs.append(conf)
                        
                        coordinates = suppressed_coords
                else:
                    coordinates = []
                
                # Save coordinates
                img_idx = batch_idx * valloader.batch_size + i
                if img_idx < len(image_files):
                    img_name = image_files[img_idx]
                    txt_name = img_name.replace('.png', '.txt')
                    txt_path = os.path.join(pred_dir, txt_name)
                    
                    with open(txt_path, 'w') as f:
                        for x, y in coordinates:
                            f.write(f"{x} {y}\n")
    
    print(f"Saved predictions to {pred_dir}")

def load_coordinates_from_txt(file_path):
    """Load x,y coordinates from text file"""
    coordinates = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        x, y = map(int, line.split())
                        coordinates.append((y, x))  # Return as (y, x) to match our format
                    except ValueError:
                        continue
    return coordinates


def visualize_results_from_text(val_folder, output_dir):
    """Create visualizations from saved coordinate text files

    Args:
        val_folder: Path to validation folder
        output_dir: Directory to save visualizations
    """

    print("Creating visualizations from saved coordinates...")
    
    # Get list of image files to process
    cells_path = os.path.join(val_folder, 'cells')
    image_files = [f for f in os.listdir(cells_path) if f.endswith('.png')]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in tqdm(image_files, desc="Creating visualizations"):
        original_img_path = os.path.join(cells_path, img_name)
        gt_file_path = os.path.join(val_folder, 'gt', f'{img_name.replace(".png", ".txt")}')
        pred_file_path = os.path.join(val_folder, 'pred', f'{img_name.replace(".png", ".txt")}')
        
        if not os.path.exists(gt_file_path) or not os.path.exists(pred_file_path):
            print(f"Skipping {img_name}: missing GT or prediction file")
            continue
            
        # Load coordinates from text files
        gt_centers = load_coordinates_from_txt(gt_file_path)
        pred_centers = load_coordinates_from_txt(pred_file_path)
        
        # Read original image
        original_img = read_img(original_img_path, greyscale=False)
        original_img_rgb = original_img.copy() / 255.0
        
        # Create copies for drawing - ensure they're in 0-255 range for OpenCV
        if original_img_rgb.max() <= 1.0:
            # If normalized (0-1), convert to 0-255 for OpenCV
            vis_img = (original_img_rgb * 255).astype(np.uint8).copy()
            gt_only_img = (original_img_rgb * 255).astype(np.uint8).copy()
            pred_only_img = (original_img_rgb * 255).astype(np.uint8).copy()
        else:
            # If already in 0-255 range, use as is
            vis_img = original_img_rgb.astype(np.uint8).copy()
            gt_only_img = original_img_rgb.astype(np.uint8).copy()
            pred_only_img = original_img_rgb.astype(np.uint8).copy()
        
        # Draw dots (small filled circles) for ground truth and predictions
        for y, x in gt_centers:
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)  # green dot
            cv2.circle(gt_only_img, (int(x), int(y)), 3, (0, 255, 0), -1)  # green dot
            
        for y, x in pred_centers:
            cv2.circle(vis_img, (int(x), int(y)), 3, (255, 0, 0), -1)   # red dot
            cv2.circle(pred_only_img, (int(x), int(y)), 3, (255, 0, 0), -1)   # red dot
        
        # Figure - show original, GT dots, predicted dots, and comparison
        # Normalize to 0-1 range for matplotlib imshow
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0,0].imshow(original_img_rgb, vmin=0, vmax=1); axes[0,0].set_title('Original Image'); axes[0,0].axis('off')
        axes[0,1].imshow(gt_only_img/255.0, vmin=0, vmax=1); axes[0,1].set_title(f'Ground Truth Dots ({len(gt_centers)} cells)'); axes[0,1].axis('off')
        axes[1,0].imshow(pred_only_img/255.0, vmin=0, vmax=1); axes[1,0].set_title(f'Predicted Dots ({len(pred_centers)} cells)'); axes[1,0].axis('off')
        axes[1,1].imshow(vis_img/255.0, vmin=0, vmax=1); axes[1,1].set_title('GT (Green) vs Pred (Red)'); axes[1,1].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{img_name.replace(".png", "_detection.png")}')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Created visualizations for {len(image_files)} images")


def calculate_metrics_from_files(gt_dir, pred_dir, output_path):
    from scipy.optimize import linear_sum_assignment

    """Calculate precision, recall, F1-score from coordinate text files"""
    threshold = 15.0
    BIG_M = 1e6  # large penalty for pairs beyond threshold

    metrics = {}
    total_tp = total_fp = total_fn = 0
    total_gt_cells = total_pred_cells = 0

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.txt')])
    print(f"Found {len(gt_files)} ground truth files to evaluate")

    for file_name in gt_files:
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)

        if not os.path.exists(pred_path):
            print(f"Warning: No prediction file found for {file_name}")
            continue

        gt_centers = load_coordinates_from_txt(gt_path)   # expects list/array of (y, x)
        pred_centers = load_coordinates_from_txt(pred_path)

        gt_arr = np.asarray(gt_centers, dtype=float)
        pr_arr = np.asarray(pred_centers, dtype=float)

        n_gt = len(gt_arr)
        n_pr = len(pr_arr)
        total_gt_cells += n_gt
        total_pred_cells += n_pr

        if n_gt == 0 and n_pr == 0:
            continue
        if n_gt == 0:
            # everything predicted is FP
            total_fp += n_pr
            continue
        if n_pr == 0:
            # everything GT is FN
            total_fn += n_gt
            continue

        # Pairwise Euclidean distances: D[i, j] = dist(GT_i, PR_j)
        # gt: (n_gt, 2), pr: (n_pr, 2) with (y, x)
        D = np.sqrt(((gt_arr[:, None, :] - pr_arr[None, :, :]) ** 2).sum(axis=2))  # (n_gt, n_pr)

        # Build cost matrix; penalize pairs beyond threshold
        C = D.copy()
        C[C >= threshold] = BIG_M

        # Hungarian (works with rectangular matrices; returns min(n_gt, n_pr) assignments)
        row_ind, col_ind = linear_sum_assignment(C)

        # Count true positives only for pairs within threshold
        assigned_dists = D[row_ind, col_ind]
        tp = int(np.sum(assigned_dists < threshold))
        fn = n_gt - tp
        fp = n_pr - tp

        total_tp += tp
        total_fn += fn
        total_fp += fp

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    metrics.update({
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Total_GT_Cells": total_gt_cells,
        "Total_Pred_Cells": total_pred_cells,
    })

    with open(output_path, 'w') as f:
        f.write("=== Cell Detection Evaluation Results ===\n")
        f.write(f"Threshold distance: {threshold} pixels\n")
        f.write("-" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n" if isinstance(v, float) else f"{k}: {v}\n")

    print(f"Evaluation results saved to {output_path}")
    print(f"Final Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return metrics


def main():
    """
    Main function orchestrating the complete cell detection pipeline
    
    Args:
        --training: Enable training mode
        --model_path: Path to pre-trained model
        --train_folder: Training data path (default: DATA/train)
        --val_folder: Validation data path (default: DATA/val)
    
    Pipeline: Data loaders → Training (optional) → GT processing → Evaluation → Visualization → Metrics
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cell Detection CNN Training and Evaluation')
    parser.add_argument('--model_path', type=str, default="DATA/checkpoints/cell_detection_model.pth", 
                       help='Path to pre-trained model for evaluation only')
    parser.add_argument('--training', action='store_true',
                       help='Run training only')
    parser.add_argument('--val_folder', type=str, default='DATA/val',
                       help='Path to validation images folder')
    parser.add_argument('--train_folder', type=str, default='DATA/train',
                       help='Path to training images folder')
    
    args = parser.parse_args()

    trainloader, valloader = data_loaders(args.train_folder, args.val_folder)
    
    if args.training:
        model = CellDetectionNetwork().to(device)
        train(model, trainloader, valloader, num_epoch=30)
        
    # Run model evaluation and save predictions
    print("\nRunning model evaluation...")
    evaluate_only(args.model_path, valloader, args.val_folder, save_dir="pred")

    ## example of how to save ground truth coordinates from dots folder
    load_and_save_ground_truth(args.val_folder, save_dir="gt")
    
    # Create visualizations from saved text files
    output_dir = os.path.join(args.val_folder, 'results')
    visualize_results_from_text(args.val_folder, output_dir)
    
    # Calculate metrics from saved text files
    print("\nCalculating final metrics from saved results...")
    calculate_metrics_from_files(os.path.join(args.val_folder, 'gt'), os.path.join(args.val_folder, 'pred'), os.path.join(args.val_folder, 'results', 'evaluation_results.txt'))

if __name__ == "__main__":
    main()
