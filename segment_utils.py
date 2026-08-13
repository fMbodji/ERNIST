import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import cv2
import math  # Add math import
from dotenv import load_dotenv

# Load environment variables for model paths
load_dotenv()

# Import SAM
from segment_anything import SamPredictor, sam_model_registry


class SegmentAnythingHelper:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.original_image = None  # Store the original image here
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize SAM model and predictor"""
        # Get SAM model path from environment variables or use default
        sam_model_path = os.getenv("SAM_CHECKPOINT_PATH", "sam_vit_h_4b8939.pth")
        sam_model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
        
        # Check if model file exists
        if not os.path.exists(sam_model_path):
            print(f"SAM model checkpoint not found at {sam_model_path}")
            print("You need to download the model checkpoint from the SAM repository:")
            print("https://github.com/facebookresearch/segment-anything#model-checkpoints")
            raise FileNotFoundError(f"SAM model checkpoint not found at {sam_model_path}")
        
        # Initialize SAM model
        self.model = sam_model_registry[sam_model_type](checkpoint=sam_model_path)
        self.model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.predictor = SamPredictor(self.model)
        
        print(f"SAM model initialized on {self.model.device}")

    def set_image(self, image_path):
        """Set the image for SAM predictor"""
        # Open image with PIL first to handle different formats
        pil_image = Image.open(image_path)
        # Convert to numpy array with RGB order
        image = np.array(pil_image)
        
        # Store the original image
        self.original_image = image.copy()
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        return image

    def predict_masks_from_points(self, points, labels=None):
        """
        Generate masks based on input points
        
        Args:
            points: List of points [[x, y], [x, y], ...] in pixel coordinates
            labels: Optional list of labels (1 for foreground, 0 for background)
                   If None, all points are treated as foreground
        
        Returns:
            masks: Generated binary masks
            scores: Confidence scores for each mask
            image_with_masks: PIL Image with all masks overlaid
        """
        if not points:
            return None, None, None
        
        # Convert points to numpy array and normalize format for SAM
        input_points = np.array(points)
        
        # If no labels provided, assume all points are foreground
        if labels is None:
            labels = np.ones(len(points), dtype=np.int64)
        else:
            labels = np.array(labels, dtype=np.int64)
        
        # Get masks from predictor
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=labels,
            multimask_output=True,  # Return multiple masks per point
        )
        
        # Get the best mask for each point (highest score)
        best_mask_idx = np.argmax(scores, axis=0)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        # Convert mask to image with overlay
        # Use the stored original image instead of trying to get it from predictor
        image_with_masks = self._create_mask_overlay(self.original_image, best_mask)
        
        return best_mask, best_score, image_with_masks
    
    def _create_mask_overlay(self, image, mask, alpha=0.5, color=[30, 144, 255]):
        """Create an overlay of the mask on the image"""
        # Create a colored overlay for the mask
        mask_image = np.zeros_like(image, dtype=np.uint8)
        mask_image[mask] = color
        
        # Blend the original image with the mask overlay
        overlay = cv2.addWeighted(image, 1, mask_image, alpha, 0)
        
        # Convert to PIL Image for easier display in Gradio
        return Image.fromarray(overlay)
    
    def create_binary_mask(self, mask):
        """
        Create a binary mask image (pure black and white)
        
        Args:
            mask: Binary mask array (True/False values)
            
        Returns:
            PIL Image with white (255) for mask area and black (0) for background
        """
        # Create a single-channel image with 0s for background and 255s for mask
        binary_mask = np.zeros(mask.shape, dtype=np.uint8)
        binary_mask[mask] = 255
        
        # Convert to PIL Image
        return Image.fromarray(binary_mask)
    
    def convert_grid_cells_to_points(self, grid_cells, image_size, grid_size=50):
        """
        Convert grid cell coordinates to pixel coordinates for SAM
        
        Args:
            grid_cells: List of grid cell coordinates [(row, col), ...]
            image_size: (width, height) of the image
            grid_size: Number of grid cells in the smaller dimension
            
        Returns:
            points: List of points [[x, y], ...] in pixel coordinates
        """
        img_width, img_height = image_size
        
        # Calculate cell size based on the smaller dimension for square cells
        cell_size = min(img_width, img_height) / grid_size
        
        # Calculate actual grid dimensions (rows and columns)
        num_cols = math.ceil(img_width / cell_size)
        num_rows = math.ceil(img_height / cell_size)
        
        points = []
        for row, col in grid_cells:
            # Skip invalid grid cells, including negative indices
            if row >= num_rows or col >= num_cols or row < 0 or col < 0:
                continue
            
            # Calculate center of the grid cell in pixel coordinates
            x = int((col + 0.5) * cell_size)
            y = int((row + 0.5) * cell_size)
            
            # Ensure point is within image bounds
            x = min(max(x, 0), img_width - 1)
            y = min(max(y, 0), img_height - 1)
            
            points.append([x, y])
        
        return points
    
    def mask_to_grid_cells(self, mask, image_size, grid_size=50):
        """
        Convert a binary mask to grid cell coordinates
        
        Args:
            mask: Binary mask array
            image_size: (width, height) of the image
            grid_size: Number of grid cells in the smaller dimension
            
        Returns:
            grid_cells: List of grid cell coordinates [(row, col), ...]
        """
        img_width, img_height = image_size
        
        # Calculate cell size based on the smaller dimension for square cells
        cell_size = min(img_width, img_height) / grid_size
        
        # Number of cells in each dimension (use math.ceil to get proper boundaries)
        num_cols = math.ceil(img_width / cell_size)
        num_rows = math.ceil(img_height / cell_size)
        
        grid_cells = set()
        
        # For each cell, check if any pixel in the cell is True in the mask
        for row in range(num_rows):
            for col in range(num_cols):
                # Skip negative indices (shouldn't happen in this loop but added for consistency)
                if row < 0 or col < 0:
                    continue
                    
                # Calculate cell bounds
                x_min = int(col * cell_size)
                y_min = int(row * cell_size)
                x_max = min(int((col + 1) * cell_size), img_width)  # Respect image boundaries
                y_max = min(int((row + 1) * cell_size), img_height) # Respect image boundaries
                
                # Ensure we don't go out of mask bounds
                if y_min >= mask.shape[0] or x_min >= mask.shape[1]:
                    continue
                
                y_max = min(y_max, mask.shape[0])
                x_max = min(x_max, mask.shape[1])
                
                # Make sure we have a valid region to check
                if y_max <= y_min or x_max <= x_min:
                    continue
                
                cell_region = mask[y_min:y_max, x_min:x_max]
                if np.any(cell_region):
                    grid_cells.add((row, col))
        
        return list(grid_cells) 