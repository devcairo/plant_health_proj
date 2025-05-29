# PlantVision

AI-Powered Plant Health and Watering Predictor

## Overview
PlantVision is a Python application that uses your computer's camera to analyze plant health in real time and predict watering needs. It fetches plant care information from the free Open Plantbook API and displays results in a user-friendly GUI.

## Features
- Live camera feed analysis
- Rule-based plant health detection (coming soon)
- Fetches plant care info from Open Plantbook (https://open.plantbook.io/)
- GUI built with Tkinter

## Requirements
- Python 3.8+
- pip

## Installation
1. Clone this repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pillow
   ```

## Usage
Run the main application:
```bash
python main.py
```

The GUI will open, showing your camera feed. Future updates will add live plant health and watering analysis.

## Project Structure
- `main.py` - Main application with GUI and camera feed
- `requirements.txt` - Python dependencies
- `README.md` - This file

## License
This project is for educational and personal use. 