# vep/hands.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os

class HandDetector:
    def __init__(self, model_filename='hand_landmarker.task'):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(current_script_dir, model_filename)

        print(f"HandDetector: Attempting to load model from: {full_model_path}")
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"HandDetector: Model file not found at: {full_model_path}")

        base_options = python.BaseOptions(model_asset_path=full_model_path)
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Using IMAGE mode
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(self.options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def draw(self, image, result):
        if not result.hand_landmarks:
            return image
        for hand in result.hand_landmarks:
            landmark_proto = landmark_pb2.NormalizedLandmarkList()
            for lm in hand:
                landmark_proto.landmark.append(
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                )
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_proto,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style())
        return image

    def close(self):
        self.detector.close()