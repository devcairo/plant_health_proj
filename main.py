# Necessary libraries
import tkinter as tk
from ai_model import PlantAIModel
from camera_handler import CameraHandler
from gui_handler import GUIHandler

class PlantVisionApp:
    """Main application class that coordinates all components"""
    
    def __init__(self, root):
        # Store root reference
        self.root = root
        
        # Initialize components
        self.gui = GUIHandler(root)
        self.ai_model = PlantAIModel()
        self.camera = CameraHandler()
        
        # Analysis control
        self.analysis_active = False
        self.frame_counter = 0
        
        # Setup GUI callbacks
        self.gui.set_button_command(self.toggle_analysis)
        self.gui.set_close_handler(self.on_close)
        
        # Load AI model
        self.load_ai_model()
        
        # Start video loop
        self.update_video()
    
    def load_ai_model(self):
        """Load the AI model and update status"""
        if self.ai_model.load_model():
            self.gui.update_status("Status: AI model loaded successfully!")
        else:
            self.gui.update_status("Status: Error loading AI model!")
    
    def toggle_analysis(self):
        """Toggle AI analysis on/off"""
        if self.ai_model.is_loaded():
            if self.analysis_active:
                self.stop_analysis()
            else:
                self.start_analysis()
        else:
            self.gui.update_ai_result("Model not loaded!")
    
    def start_analysis(self):
        """Start AI analysis"""
        if self.ai_model.is_loaded():
            self.analysis_active = True
            self.gui.update_button_text("Analysis Active - Click to Stop")
            self.gui.update_ai_result("Ready - Point camera at plant")
        else:
            self.gui.update_ai_result("Model not loaded!")
    
    def stop_analysis(self):
        """Stop AI analysis"""
        self.analysis_active = False
        self.gui.update_button_text("Start Plant Analysis")
        self.gui.update_ai_result("Stopped")
    
    def update_video(self):
        """Main video update loop"""
        if not self.camera.is_running():
            return
        
        ret, frame = self.camera.read_frame()
        if ret:
            # Run AI analysis if active
            if self.analysis_active and self.ai_model.is_loaded():
                self.frame_counter += 1
                if self.frame_counter % 10 == 0:  # Every 10th frame
                    ai_result = self.ai_model.analyze_frame(frame)
                    self.gui.update_ai_result(ai_result)
            
            # Update video display
            display_image = self.camera.convert_frame_for_display(frame)
            self.gui.update_video(display_image)
        
        # Schedule next update
        self.root.after(30, self.update_video)
    
    def on_close(self):
        """Handle application close"""
        self.camera.release()
        self.root.destroy()

if __name__ == "__main__":
    # Start the application
    root = tk.Tk()
    app = PlantVisionApp(root)
    root.mainloop()