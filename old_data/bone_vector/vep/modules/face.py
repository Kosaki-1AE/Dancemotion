# vep/face.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import os

class FaceDetector:
    def __init__(self, model_filename='face_landmarker.task'):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        full_model_path = os.path.join(current_script_dir, model_filename)

        print(f"FaceDetector: Attempting to load model from: {full_model_path}")
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"FaceDetector: Model file not found at: {full_model_path}")

        base_options = python.BaseOptions(model_asset_path=full_model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            # REMOVED: output_facial_transformation_matrices=True,
            running_mode=vision.RunningMode.IMAGE, # Using IMAGE mode
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(self.options)

    def detect(self, mp_image):
        return self.detector.detect(mp_image)

    def draw(self, image, result):
        if not result.face_landmarks:
            return image
        for landmarks in result.face_landmarks:
            landmark_proto = landmark_pb2.NormalizedLandmarkList()
            for lm in landmarks:
                landmark_proto.landmark.append(
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                )
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=landmark_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
        return image

    def close(self):
        self.detector.close()