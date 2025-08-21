"""
Camera Handler for PlantVision
==============================

This module handles all camera-related operations for the PlantVision application.
It manages camera initialization, frame capture, and video format conversion
for display in the GUI.

The camera handler provides a clean interface for:
- Camera initialization and management
- Frame capture and reading
- Video format conversion (OpenCV to Tkinter)
- Resource cleanup

Author: Sa'Cairo Bonner
Date: 2025
"""

import cv2
from PIL import Image, ImageTk

class CameraHandler:
    """
    Handles camera operations and video processing.
    
    This class manages:
    - Camera initialization and connection
    - Frame capture from the camera
    - Video format conversion for GUI display
    - Proper resource cleanup when closing
    
    The camera handler abstracts away the complexity of OpenCV camera
    operations and provides a simple interface for the main application.
    """
    
    def __init__(self, camera_index=0):
        """
        Initialize the camera handler.
        
        Args:
            camera_index (int): Index of the camera to use (0 for default camera)
        """
        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(camera_index)
        
        # Flag to control camera operation
        self.running = True
        
    def read_frame(self):
        """
        Read a frame from the camera.
        
        This method captures a single frame from the camera and returns
        both the frame data and a success flag.
        
        Returns:
            tuple: (frame, success) where frame is numpy array or None,
                   and success is boolean indicating if frame was captured
        """
        # Check if camera is still running
        if not self.running:
            return None, False
        
        # Read frame from camera (returns frame and success flag)
        return self.cap.read()
    
    def convert_frame_for_display(self, frame):
        """
        Convert OpenCV frame to Tkinter-compatible format.
        
        This method performs the necessary conversions to display
        OpenCV frames in the Tkinter GUI:
        1. Converts BGR color space to RGB
        2. Converts numpy array to PIL Image
        3. Converts PIL Image to Tkinter PhotoImage
        
        Args:
            frame (numpy.ndarray): Raw camera frame in BGR format
            
        Returns:
            ImageTk.PhotoImage: Tkinter-compatible image for display
        """
        # Convert BGR to RGB (OpenCV uses BGR, but Tkinter expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(rgb)
        
        # Convert PIL Image to Tkinter-compatible image
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk
    
    def release(self):
        """
        Release camera resources.
        
        This method properly cleans up camera resources to prevent
        memory leaks and ensure the camera is available for other applications.
        """
        # Stop camera operation
        self.running = False
        
        # Release OpenCV camera capture
        self.cap.release()
    
    def is_running(self):
        """
        Check if camera is running.
        
        Returns:
            bool: True if camera is active, False otherwise
        """
        return self.running
