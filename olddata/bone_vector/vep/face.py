# face.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class FaceDetector:
    def __init__(self, model_path='models/face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE,
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

