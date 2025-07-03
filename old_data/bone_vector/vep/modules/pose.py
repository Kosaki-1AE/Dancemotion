# vep/pose.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os

class PoseDetector:
    def __init__(self, model_filename='pose_landmarker_heavy.task'):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(current_script_dir, model_filename)

        print(f"PoseDetector: Attempting to load model from: {full_model_path}")
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"PoseDetector: Model file not found at: {full_model_path}")

        base_options = python.BaseOptions(model_asset_path=full_model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE # Using IMAGE mode
        )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def draw(self, image, result):
        if not result.pose_landmarks:
            return image
        for pose in result.pose_landmarks:
            landmark_proto = landmark_pb2.NormalizedLandmarkList()
            for lm in pose:
                landmark_proto.landmark.append(
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                )
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_proto,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())
        return image

    def close(self):
        self.detector.close()