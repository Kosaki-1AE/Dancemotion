# main.py
import cv2
import numpy as np
import mediapipe as mp

from face import FaceDetector
from hands import HandDetector
from pose import PoseDetector


face_detector = FaceDetector()
hand_detector = HandDetector()
pose_detector = PoseDetector()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Full Body Holistic Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Full Body Holistic Capture", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    face_result = face_detector.detect(mp_image)
    hand_result = hand_detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)

    output = frame.copy()
    output = face_detector.draw(output, face_result)
    output = hand_detector.draw(output, hand_result)
    output = pose_detector.draw(output, pose_result)

    cv2.imshow("Full Body Holistic Capture", output)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

face_detector.close()
hand_detector.close()
pose_detector.close()
