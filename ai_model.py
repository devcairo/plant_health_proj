"""
AI Model Handler for PlantVision
================================

This module handles all AI-related operations for plant disease detection.
It loads a pre-trained MobileNetV2 model trained on the PlantVillage dataset
and provides methods for preprocessing images and analyzing plant health.

The model can classify 120 different plant species and disease combinations,
including healthy plants and various disease states.

Author: PlantVision Team
Date: 2024
"""

import torch
import torchvision.models as models
import numpy as np
import cv2
import os
from plant_classes import get_plant_info, is_healthy, get_plant_species

class PlantAIModel:
    """
    Handles all AI model operations for plant disease detection.
    
    This class manages:
    - Loading the pre-trained MobileNetV2 model
    - Preprocessing camera frames for AI input
    - Running inference on plant images
    - Decoding predictions to human-readable results
    
    The model is trained on the PlantVillage dataset with 120 classes
    covering 38 plant species and their various disease states.
    """
    
    def __init__(self, model_path="MobileNetV2-model-91.pth"):
        """
        Initialize the AI model handler.
        
        Args:
            model_path (str): Path to the pre-trained model file
        """
        self.model_path = model_path
        self.model = None  # PyTorch model instance
        self.model_loaded = False  # Flag to track if model is loaded
        
    def load_model(self):
        """
        Loads the pre-trained plant disease detection model.
        
        This method:
        1. Checks if the model file exists
        2. Creates a MobileNetV2 architecture with 120 output classes
        3. Loads the pre-trained weights from the .pth file
        4. Sets the model to evaluation mode for inference
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False
            
            # Create MobileNetV2 architecture with 120 output classes
            # (matches the PlantVillage dataset structure)
            self.model = models.mobilenet_v2(pretrained=False, num_classes=120)
            
            # Load the pre-trained weights from file
            # Use CPU for compatibility (can be changed to GPU if available)
            state_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
            
            # Set model to evaluation mode (no training, only inference)
            self.model.eval()
            
            # Update status and return success
            self.model_loaded = True
            print("PyTorch model loaded from file.")
            return True
            
        except Exception as e:
            print(f"Error loading AI model: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """
        Preprocess a camera frame for AI model input.
        
        This method performs the following transformations:
        1. Resize image to 224x224 (MobileNetV2 input size)
        2. Convert BGR to RGB color space
        3. Normalize pixel values to [0, 1] range
        4. Apply ImageNet normalization (mean and std)
        5. Convert to PyTorch tensor format
        6. Add batch dimension
        
        Args:
            frame (numpy.ndarray): Raw camera frame in BGR format
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model inference
        """
        # Resize to model input size (224x224 for MobileNetV2)
        resized = cv2.resize(frame, (224, 224))
        
        # Convert BGR to RGB (OpenCV uses BGR, but model expects RGB)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize to [0, 1] range
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (standard preprocessing for ImageNet models)
        # These values are the mean and standard deviation of ImageNet dataset
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to PyTorch tensor format (Channels, Height, Width)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        
        # Add batch dimension (model expects batch of images)
        batched = tensor.unsqueeze(0)
        return batched
    
    def analyze_frame(self, frame):
        """
        Analyze a camera frame using the AI model.
        
        This method:
        1. Preprocesses the frame for model input
        2. Runs inference using the loaded model
        3. Extracts prediction results
        4. Decodes the prediction to human-readable format
        5. Provides debug output to console
        
        Args:
            frame (numpy.ndarray): Raw camera frame
            
        Returns:
            str: Human-readable analysis result or error message
        """
        # Check if model is loaded before attempting analysis
        if not self.model_loaded or self.model is None:
            return "AI model not loaded"
        
        try:
            # Preprocess the frame for model input
            processed_frame = self.preprocess_frame(frame)
            
            # Run inference with PyTorch (no gradient computation needed)
            with torch.no_grad():
                # Get model prediction
                prediction = self.model(processed_frame)
                
                # Extract prediction details for debug output
                predicted_class = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1).max().item()
                print(f"DEBUG: Class {predicted_class}, Confidence: {confidence:.3f}")
                
                # Decode prediction to human-readable result
                result = self.decode_prediction(prediction)
            
            return result
        except Exception as e:
            return f"AI Analysis Error: {str(e)}"
    
    def decode_prediction(self, prediction):
        """
        Decode model prediction to human-readable result.
        
        This method:
        1. Extracts the predicted class index (0-119)
        2. Calculates confidence score (0-1)
        3. Maps class index to plant species and condition
        4. Determines if plant is healthy or diseased
        5. Formats result for display
        
        Args:
            prediction (torch.Tensor): Raw model prediction tensor
            
        Returns:
            str: Formatted result string for GUI display
        """
        try:
            # Get the predicted class index (which of the 120 classes)
            predicted_class = torch.argmax(prediction, dim=1).item()
            
            # Get confidence score (how sure the model is)
            confidence = torch.softmax(prediction, dim=1).max().item()
            
            # Get plant information using the class mapping from plant_classes.py
            plant_info = get_plant_info(predicted_class)
            plant_species = get_plant_species(predicted_class)
            
            # Determine health status based on class name
            health_status = "Healthy" if is_healthy(predicted_class) else "Diseased"
            
            # Format result for GUI display
            return f"{plant_species} - {plant_info} ({health_status}, {confidence:.2f})"
            
        except Exception as e:
            return f"Prediction Error: {str(e)}"
    
    def is_loaded(self):
        """
        Check if the AI model is loaded and ready for inference.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model_loaded
