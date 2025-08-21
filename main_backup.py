# Necessary libraries
import cv2  # For camera access and image processing
import tkinter as tk 
from tkinter import ttk  # For themed widgets in Tkinter
from PIL import Image, ImageTk
import threading  # (Not used yet, but may be useful for future threading)
import requests  # For API calls (to be used later)
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np  # For numerical operations
import os  # For file operations
from plant_classes import get_plant_info, is_healthy, get_plant_species

MODEL_PATH = "MobileNetV2-model-91.pth"

class PlantVisionApp:
    def __init__(self, root):
        # Main window for the application
        self.root = root
        self.root.title('PlantVision - Live Plant Health Analyzer')
        self.root.geometry('900x700')
        
        # Label to display status messages (e.g., health, watering info)
        self.status_label = ttk.Label(self.root, text='Status: Loading AI model...', font=('Arial', 14))
        self.status_label.pack(pady=10)
        
        # AI Model setup
        self.model = None
        self.model_loaded = False
        self.load_ai_model()  # Load the pre-trained model
        
        # Label to display the video feed
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)
        
        # Label to display AI analysis results
        self.ai_result_label = ttk.Label(self.root, text='AI Analysis: Not started', font=('Arial', 12))
        self.ai_result_label.pack(pady=5)
        
        # Button to start AI analysis
        self.analyze_button = ttk.Button(self.root, text='Start Plant Analysis', command=self.toggle_analysis)
        self.analyze_button.pack(pady=5)
        
        # Analysis control flag
        self.analysis_active = False
        self.frame_counter = 0  # Add frame counter to slow down analysis
        
        # Opens the default camera (usually the first webcam)
        self.cap = cv2.VideoCapture(0)
        self.running = True  # Control flag for the video loop
        self.update_video()  # Start updating the video feed
        
        # Handle window close event to release camera properly
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def load_ai_model(self):
        """Loads the pre-trained plant disease detection model"""
        try:
            if not os.path.exists(MODEL_PATH):
                self.status_label.config(text='Status: Model file not found!')
                print(f"Model file not found: {MODEL_PATH}")
                return
            
            # Create the model architecture with 120 output classes (matching the pre-trained model)
            self.model = models.mobilenet_v2(pretrained=False, num_classes=120)
            
            # Load the state dictionary (weights) into the model
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.status_label.config(text='Status: AI model loaded successfully!')
            self.model_loaded = True
            print("PyTorch model loaded from file.")
        except Exception as e:
            self.status_label.config(text=f'Status: Error loading AI model: {str(e)}')
            print(f"Error loading AI model: {e}")

    def preprocess_frame_for_ai(self, frame):
        """Preprocess the frame for AI model input"""
        # Resize to model input size (224x224 for MobileNetV2)
        resized = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Convert to float and normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        # Apply ImageNet normalization (mean and std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        # Convert to PyTorch tensor format (C, H, W) with float32
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        # Add batch dimension
        batched = tensor.unsqueeze(0)
        return batched

    def analyze_frame_with_ai(self, frame):
        """Analyze the frame using the AI model"""
        if not self.model_loaded or self.model is None:
            return "AI model not loaded"
        
        try:
            # Preprocess the frame
            processed_frame = self.preprocess_frame_for_ai(frame)
            
            # Run inference with PyTorch
            with torch.no_grad():
                prediction = self.model(processed_frame)
                
                # Add debug output to console
                predicted_class = torch.argmax(prediction, dim=1).item()
                confidence = torch.softmax(prediction, dim=1).max().item()
                print(f"DEBUG: Class {predicted_class}, Confidence: {confidence:.3f}")
                
                result = self.decode_prediction(prediction)
            
            return result
        except Exception as e:
            return f"AI Analysis Error: {str(e)}"

    def decode_prediction(self, prediction):
        """Decode model prediction to human-readable result"""
        try:
            # Get the predicted class index
            predicted_class = torch.argmax(prediction, dim=1).item()
            
            # Get confidence score
            confidence = torch.softmax(prediction, dim=1).max().item()
            
            # Get plant information using the class mapping
            plant_info = get_plant_info(predicted_class)
            plant_species = get_plant_species(predicted_class)
            health_status = "Healthy" if is_healthy(predicted_class) else "Diseased"
            
            return f"{plant_species} - {plant_info} ({health_status}, {confidence:.2f})"
            
        except Exception as e:
            return f"Prediction Error: {str(e)}"

    def toggle_analysis(self):
        """Toggles the AI analysis on/off"""
        if self.model_loaded:
            if self.analysis_active:
                self.stop_analysis()
            else:
                self.start_analysis()
        else:
            self.ai_result_label.config(text='AI Analysis: Model not loaded!')

    def start_analysis(self):
        """Start the AI analysis when button is pressed"""
        if self.model_loaded:
            self.analysis_active = True
            self.analyze_button.config(text='Analysis Active - Click to Stop')
            self.ai_result_label.config(text='AI Analysis: Ready - Point camera at plant')
        else:
            self.ai_result_label.config(text='AI Analysis: Model not loaded!')

    def stop_analysis(self):
        """Stop the AI analysis"""
        self.analysis_active = False
        self.analyze_button.config(text='Start Plant Analysis')
        self.ai_result_label.config(text='AI Analysis: Stopped')

    def update_video(self):
        # This function grabs a frame from the camera and updates the GUI
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            # Only run AI analysis if user has activated it (every 10 frames = ~3 FPS for analysis)
            if self.analysis_active and self.model_loaded:
                self.frame_counter += 1
                if self.frame_counter % 10 == 0:  # Only analyze every 10th frame
                    ai_result = self.analyze_frame_with_ai(frame)
                    self.ai_result_label.config(text=f'AI Analysis: {ai_result}')
            
            # Convert frame for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Tkinter
            img = Image.fromarray(rgb)  # Convert to PIL Image
            imgtk = ImageTk.PhotoImage(image=img)  # Convert to Tkinter-compatible image
            self.video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
            self.video_label.configure(image=imgtk)  # Update the label with the new image
        self.root.after(30, self.update_video)  # Schedule the next frame update (about 30 FPS)

    def on_close(self):
        # Clean up: release the camera and close the window
        self.running = False
        self.cap.release()
        self.root.destroy()

    # Placeholder for future plant health analysis
    def analyze_frame(self, frame):
        # TODO: Add rule-based or AI-based analysis here
        return frame

if __name__ == '__main__':
    # Start the Tkinter GUI application
    root = tk.Tk()
    app = PlantVisionApp(root)
    root.mainloop() 
    