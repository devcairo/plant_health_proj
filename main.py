# Necessary libraries
import cv2  # For camera access and image processing
import tkinter as tk 
from tkinter import ttk  # For themed widgets in Tkinter
from PIL import Image, ImageTk
import threading  # (Not used yet, but may be useful for future threading)
import requests  # For API calls (to be used later)
import tensorflow as tf  # For AI model inference
import numpy as np  # For numerical operations
import os  # For file operations

MODEL_URL = "https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/plant_disease_model.h5"
MODEL_PATH = "plant_disease_model.h5"

class PlantVisionApp:
    def __init__(self, root):
        # Main window
        self.root = root
        self.root.title('PlantVision - Live Plant Health Analyzer')
        self.root.geometry('900x700')
        
        # AI Model setup
        self.model = None
        self.model_loaded = False
        self.load_ai_model()  # Load the pre-trained model
        
        # Label to display the video feed
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)
        
        # Label to display status messages (e.g., health, watering info)
        self.status_label = ttk.Label(self.root, text='Status: Loading AI model...', font=('Arial', 14))
        self.status_label.pack(pady=10)
        
        # Label to display AI analysis results
        self.ai_result_label = ttk.Label(self.root, text='AI Analysis: Not started', font=('Arial', 12))
        self.ai_result_label.pack(pady=5)
        
        # Opens the default camera (usually the first webcam)
        self.cap = cv2.VideoCapture(0)
        self.running = True  # Control flag for the video loop
        self.update_video()  # Start updating the video feed
        
        # Handle window close event to release camera properly
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def ensure_model_downloaded(self):
        """Check if the model file exists, and download it if not."""
        if not os.path.exists(MODEL_PATH):
            self.status_label.config(text='Status: Downloading AI model...')
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                self.status_label.config(text='Status: Model downloaded successfully!')
            except Exception as e:
                self.status_label.config(text=f'Status: Error downloading model: {str(e)}')
                raise

    def load_ai_model(self):
        """Loads the pre-trained plant disease detection model"""
        try:
            self.ensure_model_downloaded()
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.status_label.config(text='Status: AI model loaded successfully!')
            self.model_loaded = True
            print("AI model loaded from file.")
        except Exception as e:
            self.status_label.config(text=f'Status: Error loading AI model: {str(e)}')
            print(f"Error loading AI model: {e}")

    def preprocess_frame_for_ai(self, frame):
        """Preprocess the frame for AI model input"""
        # Resize to model input size (typically 224x224 or 299x299)
        resized = cv2.resize(frame, (224, 224))
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        return batched

    def analyze_frame_with_ai(self, frame):
        """Analyze the frame using the AI model"""
        if not self.model_loaded or self.model is None:
            return "AI model not loaded"
        
        try:
            # Preprocess the frame
            processed_frame = self.preprocess_frame_for_ai(frame)
            
            # Run inference (placeholder for now)
            # prediction = self.model.predict(processed_frame)
            # result = self.decode_prediction(prediction)
            
            # Placeholder result
            result = "Healthy Plant Detected"
            return result
        except Exception as e:
            return f"AI Analysis Error: {str(e)}"

    def decode_prediction(self, prediction):
        """Decode model prediction to human-readable result"""
        # This will be implemented when we have the actual model
        # For now, return a placeholder
        return "Analysis Complete"

    def update_video(self):
        # This function grabs a frame from the camera and updates the GUI
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            # Run AI analysis on the frame (every 30 frames to avoid lag)
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 0
            
            # Run AI analysis every 30 frames (about once per second at 30 FPS)
            if self.frame_count % 30 == 0 and self.model_loaded:
                ai_result = self.analyze_frame_with_ai(frame)
                self.ai_result_label.config(text=f'AI Analysis: {ai_result}')
            
            # Placeholder for health analysis (to be added later)
            # frame = self.analyze_frame(frame)
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