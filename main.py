import cv2
import mediapipe as mp
import numpy as np

mpose=mp.solutions.pose
pose=mpose.Pose()
draw=mp.solutions.drawing_utils
d1=draw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,255))
d2=draw.DrawingSpec(thickness=3,circle_radius=8,color=(0,255,0))
cap = cv2.VideoCapture('1.mp4')

while True:
    _, img = cap.read()
    img=cv2.resize(img,(600,600))
    results=pose.process(img)
    draw.draw_landmarks(img,results.pose_landmarks,mpose.POSE_CONNECTIONS,d1,d2)
    cv2.imshow("Pose Detection", img)

    img_blank=np.zeros(img.shape)
    img_blank.fill(255)
    draw.draw_landmarks(img_blank, results.pose_landmarks, mpose.POSE_CONNECTIONS, d1, d2)
    cv2.imshow("Extracted Pose",img_blank)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
