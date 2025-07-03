# vep/frame_producer.py
import cv2
import numpy as np
import mediapipe as mp
import time
import sys
import os
import queue

# Since all files are in the same 'vep' folder, direct imports are possible
from modules.face import FaceDetector
from modules.hands import HandDetector
from modules.pose import PoseDetector
from shared_data import frame_queue, stop_processing_flag # Import from shared_data

# Initialize detectors (passing only the filename)
try:
    face_detector = FaceDetector('face_landmarker.task')
    hand_detector = HandDetector('hand_landmarker.task')
    pose_detector = PoseDetector('pose_landmarker_heavy.task')
except FileNotFoundError as e:
    print(f"Error loading MediaPipe model in frame_producer: {e}")
    sys.exit(1) # Exit if models can't be loaded

def run_frame_processing():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam in frame_producer. Please ensure it's connected and not in use.")
        return

    print("Frame producer started and is pushing frames to queue.")
    try:
        # Loop while the stop_processing_flag is not set to True
        while not stop_processing_flag:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam in producer.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            output = frame.copy()
            current_blendshapes = []

            # Perform detections
            face_result = face_detector.detect(mp_image)
            hand_result = hand_detector.detect(mp_image)
            pose_result = pose_detector.detect(mp_image)

            # Extract blendshapes if available
            if face_result and face_result.face_blendshapes:
                if len(face_result.face_blendshapes) > 0:
                    # --- SIMPLIFIED FIX FOR AttributeError: 'list' object has no attribute 'categories' ---
                    # The error indicates that face_result.face_blendshapes[0] is already the list of categories.
                    # We will directly iterate over it.
                    
                    # This line is the core change:
                    blendshape_categories = face_result.face_blendshapes[0] 
                    
                    # Optional: Add a debug print to confirm the type
                    # print(f"DEBUG: Type of blendshape_categories: {type(blendshape_categories)}")
                    # if isinstance(blendshape_categories, list) and len(blendshape_categories) > 0:
                    #    print(f"DEBUG: Type of first item in blendshape_categories: {type(blendshape_categories[0])}")


                    for category in blendshape_categories: # Direct iteration
                        # Ensure 'category' actually has these attributes, for safety
                        if hasattr(category, 'display_name') or hasattr(category, 'category_name'):
                            name = category.display_name or category.category_name
                            score = category.score
                            current_blendshapes.append({
                                'name': name,
                                'score': score
                            })
                        else:
                            print(f"Warning: Unexpected category object structure in blendshapes: {type(category)}, dir: {dir(category)}")
                            # You might want to break or handle this case more specifically
                            

            output = face_detector.draw(output, face_result)
            output = hand_detector.draw(output, hand_result)
            output = pose_detector.draw(output, pose_result)

            # Convert processed frame to bytes (e.g., JPEG compression for efficiency)
            ret, buffer = cv2.imencode('.jpg', output)
            if ret:
                try:
                    # Put frame into the queue. Non-blocking with timeout
                    # This prevents producer from blocking indefinitely if consumer is slow
                    frame_queue.put(buffer.tobytes(), timeout=0.1)
                except queue.Full:
                    # If queue is full, skip this frame to keep up
                    pass
            
            time.sleep(0.01) # Small delay to prevent busy-waiting

    finally:
        cap.release()
        face_detector.close()
        hand_detector.close()
        pose_detector.close()
        print("Frame producer stopped and resources released.")

if __name__ == "__main__":
    # This block ensures run_frame_processing is called only when frame_producer.py is run directly
    # and not when it's imported by streamlit.py.
    run_frame_processing()
