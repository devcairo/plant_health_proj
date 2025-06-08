# Necessary libraries
import cv2  # For camera access and image processing
import tkinter as tk 
from tkinter import ttk  # For themed widgets in Tkinter
from PIL import Image, ImageTk
import threading  # (Not used yet, but may be useful for future threading)
import requests  # For API calls (to be used later)

class PlantVisionApp:
    def __init__(self, root):
        # Main window
        self.root = root
        self.root.title('PlantVision - Live Plant Health Analyzer')
        self.root.geometry('900x700')
        
        # Label to display the video feed
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)
        
        # Label to display status messages (e.g., health, watering info)
        self.status_label = ttk.Label(self.root, text='Status: Waiting for camera...', font=('Arial', 14))
        self.status_label.pack(pady=10)
        
        # Opens the default camera (usually the first webcam)
        self.cap = cv2.VideoCapture(0)
        self.running = True  # Control flag for the video loop
        self.update_video()  # Start updating the video feed
        
        # Handle window close event to release camera properly
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def update_video(self):
        # This function grabs a frame from the camera and updates the GUI
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
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