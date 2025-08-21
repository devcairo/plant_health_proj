import tkinter as tk
from tkinter import ttk

class GUIHandler:
    """Handles all GUI-related operations"""
    
    def __init__(self, root):
        self.root = root
        self.status_label = None
        self.video_label = None
        self.ai_result_label = None
        self.analyze_button = None
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI components"""
        self.root.title("PlantVision - Live Plant Health Analyzer")
        self.root.geometry("900x700")
        
        # Status label
        self.status_label = ttk.Label(self.root, text="Status: Loading AI model...", font=("Arial", 14))
        self.status_label.pack(pady=10)
        
        # Video label
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)
        
        # AI result label
        self.ai_result_label = ttk.Label(self.root, text="AI Analysis: Not started", font=("Arial", 12))
        self.ai_result_label.pack(pady=5)
        
        # Analysis button
        self.analyze_button = ttk.Button(self.root, text="Start Plant Analysis")
        self.analyze_button.pack(pady=5)
    
    def update_status(self, text):
        """Update status label"""
        if self.status_label:
            self.status_label.config(text=text)
    
    def update_video(self, image):
        """Update video display"""
        if self.video_label:
            self.video_label.imgtk = image  # Keep reference
            self.video_label.configure(image=image)
    
    def update_ai_result(self, text):
        """Update AI analysis result"""
        if self.ai_result_label:
            self.ai_result_label.config(text=f"AI Analysis: {text}")
    
    def set_button_command(self, command):
        """Set button command"""
        if self.analyze_button:
            self.analyze_button.config(command=command)
    
    def update_button_text(self, text):
        """Update button text"""
        if self.analyze_button:
            self.analyze_button.config(text=text)
    
    def set_close_handler(self, handler):
        """Set window close handler"""
        self.root.protocol("WM_DELETE_WINDOW", handler)
