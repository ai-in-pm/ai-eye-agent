import os
import time
import json
import numpy as np
import pyautogui
import logging
import cv2
import dlib
from dotenv import load_dotenv
import win32gui
import win32con
import win32api
from openai import OpenAI
import torch
from sklearn.model_selection import train_test_split
import threading
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

class CalibrationWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.configure(background='black')
        
        # Get screen dimensions
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, 
                              bg='black', highlightthickness=0)
        self.canvas.pack()
        
        # Create text
        self.text = self.canvas.create_text(self.width//2, 50, 
                                          text="Look at the green dot", 
                                          fill="white", font=('Arial', 24))
        
        # Create dot
        self.dot = self.canvas.create_oval(0, 0, 0, 0, fill='green')
    
    def show_point(self, x, y):
        """Show calibration point at given coordinates"""
        size = 20
        screen_x = int(x * self.width)
        screen_y = int(y * self.height)
        self.canvas.coords(self.dot, 
                         screen_x - size, screen_y - size,
                         screen_x + size, screen_y + size)
        self.root.update()
    
    def close(self):
        """Close the calibration window"""
        self.root.destroy()

class WebcamEyeTracker:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        # Initialize face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        # Download the shape predictor file if it doesn't exist
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Downloading facial landmarks predictor...")
            import urllib.request
            url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, predictor_path + ".bz2")
            import bz2
            with bz2.open(predictor_path + ".bz2") as f_in, open(predictor_path, 'wb') as f_out:
                f_out.write(f_in.read())
            os.remove(predictor_path + ".bz2")
            
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Initialize variables for eye tracking
        self.eye_position = (0, 0)
        self.running = True
        self.tracking_thread = threading.Thread(target=self._track_eyes)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
    
    def _get_eye_position(self, frame):
        """Extract eye position from webcam frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Get the eye regions
            left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                               (landmarks.part(37).x, landmarks.part(37).y),
                               (landmarks.part(38).x, landmarks.part(38).y),
                               (landmarks.part(39).x, landmarks.part(39).y),
                               (landmarks.part(40).x, landmarks.part(40).y),
                               (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
            
            right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                (landmarks.part(43).x, landmarks.part(43).y),
                                (landmarks.part(44).x, landmarks.part(44).y),
                                (landmarks.part(45).x, landmarks.part(45).y),
                                (landmarks.part(46).x, landmarks.part(46).y),
                                (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
            
            # Calculate the center of both eyes
            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)
            gaze_center = np.mean([left_center, right_center], axis=0)
            
            # Normalize coordinates
            frame_height, frame_width = frame.shape[:2]
            x = gaze_center[0] / frame_width
            y = gaze_center[1] / frame_height
            
            return (x, y)
            
        return None
    
    def _track_eyes(self):
        """Continuous eye tracking thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                position = self._get_eye_position(frame)
                if position is not None:
                    self.eye_position = position
            time.sleep(0.03)  # ~30 FPS
    
    def get_pos(self):
        """Get the current eye position"""
        return self.eye_position
    
    def close(self):
        """Clean up resources"""
        self.running = False
        self.tracking_thread.join()
        self.cap.release()

class AIEyeAgent:
    def __init__(self):
        logging.info("Initializing AI Eye Agent...")
        try:
            # Initialize OpenAI client
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            logging.info("OpenAI client initialized successfully")
            
            # Initialize webcam eye tracker
            logging.info("Initializing webcam eye tracker...")
            self.tracker = WebcamEyeTracker()
            logging.info("Webcam eye tracker initialized successfully")
            
            # Initialize data collection
            self.training_data = []
            self.model = self._create_model()
            logging.info("Neural network model created successfully")
            
            # Screen properties
            self.screen_width, self.screen_height = pyautogui.size()
            logging.info(f"Screen size detected: {self.screen_width}x{self.screen_height}")
            
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise
    
    def _create_model(self):
        """Create a simple neural network model for eye tracking prediction"""
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )
            return model
        except Exception as e:
            logging.error(f"Error creating model: {str(e)}")
            raise
    
    def collect_training_data(self, duration=60):
        """Collect training data for the specified duration"""
        logging.info("Starting training data collection...")
        print("\nCalibration starting...")
        print("Please follow the green dot with your eyes as it moves around the screen.")
        print(f"Calibration will take {duration} seconds.")
        
        start_time = time.time()
        
        try:
            # Create calibration window
            calibration = CalibrationWindow()
            
            # Create calibration points
            points = [
                (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
                (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
            ]
            
            point_duration = duration / len(points)
            
            for px, py in points:
                # Show calibration point
                calibration.show_point(px, py)
                
                point_start = time.time()
                while time.time() - point_start < point_duration:
                    # Get eye position
                    eye_pos = self.tracker.get_pos()
                    
                    # Store the data point
                    self.training_data.append({
                        'gaze_x': eye_pos[0],
                        'gaze_y': eye_pos[1],
                        'cursor_x': px,
                        'cursor_y': py
                    })
                    
                    time.sleep(0.1)
                
                print(f"\rCalibrating... {int((time.time() - start_time) / duration * 100)}%", end="")
            
            print("\nCalibration completed!")
            calibration.close()
            
        except Exception as e:
            logging.error(f"Error collecting training data: {str(e)}")
            raise
            
    def train_model(self):
        """Train the model on collected data"""
        if not self.training_data:
            logging.warning("No training data available!")
            return
            
        logging.info("Starting model training...")
        try:
            # Prepare training data
            X = torch.tensor([[d['gaze_x'], d['gaze_y']] for d in self.training_data], dtype=torch.float32)
            y = torch.tensor([[d['cursor_x'], d['cursor_y']] for d in self.training_data], dtype=torch.float32)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Training loop
            optimizer = torch.optim.Adam(self.model.parameters())
            criterion = torch.nn.MSELoss()
            
            print("\nTraining the model...")
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                    
            print("Model training completed!")
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
    
    def execute_command(self, command):
        """Execute system commands based on AI interpretation"""
        logging.info(f"Executing command: {command}")
        try:
            if 'open' in command.lower():
                if 'browser' in command.lower():
                    os.system('start chrome')
                elif 'document' in command.lower() or 'file' in command.lower():
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": f"What document should I open based on this command: {command}"}
                        ]
                    )
                    document_path = response.choices[0].message.content
                    os.startfile(document_path)
        except Exception as e:
            logging.error(f"Error executing command: {str(e)}")
            raise
    
    def run(self):
        """Main loop for the AI Eye Agent"""
        logging.info("Starting AI Eye Agent main loop...")
        print("\nAI Eye Agent is now running!")
        print("Look around your screen - the cursor will follow your gaze.")
        print("Press Ctrl+C to exit.")
        
        try:
            while True:
                # Get eye position
                eye_pos = self.tracker.get_pos()
                logging.debug(f"Current eye position: {eye_pos}")
                
                # Use the trained model to predict cursor position
                with torch.no_grad():
                    eye_tensor = torch.tensor([eye_pos], dtype=torch.float32)
                    predicted_pos = self.model(eye_tensor).numpy()[0]
                    logging.debug(f"Predicted cursor position: {predicted_pos}")
                
                # Move cursor to predicted position
                screen_x = int(predicted_pos[0] * self.screen_width)
                screen_y = int(predicted_pos[1] * self.screen_height)
                pyautogui.moveTo(screen_x, screen_y)
                
                time.sleep(0.1)  # Prevent too frequent updates
                
        except KeyboardInterrupt:
            logging.info("Stopping AI Eye Agent...")
            print("\nStopping AI Eye Agent...")
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            raise
        finally:
            self.tracker.close()

if __name__ == "__main__":
    try:
        print("Welcome to AI Eye Agent!")
        print("This version uses your webcam for eye tracking.")
        print("Please make sure you have good lighting and your face is visible to the camera.")
        
        agent = AIEyeAgent()
        
        # Collect training data with calibration
        agent.collect_training_data(duration=45)  # 45 seconds for calibration
        
        # Train the model
        agent.train_model()
        
        # Run the agent
        agent.run()
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")
        print("Check the logs for more details.")
