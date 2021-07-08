import time
import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

start_time = current_time = 0


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time
    cv2.putText(image, 'fps: ' + str(int(fps)), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

    dots = []
    for data_point in results.pose_landmarks.landmark:
      dots.append(data_point)
      # x, y, z  = data_point.x, data_point.y, data_point.z
      # print(x, y, z)
    # print(dots[0])
    # # break

    x1 = dots[12].x
    y1 = dots[12].y
    z1 = dots[12].z
    x2 = dots[14].x
    y2 = dots[14].y
    z2 = dots[14].z
    x3 = dots[16].x
    y3 = dots[16].y
    z3 = dots[16].z
    cos_alpha = (
      (x1*x3 + y1*y3) / (math.sqrt(x1**2 + y1**2) * math.sqrt(x3**2 + y3**2))
    )
    arccos_alpha = math.acos(
            (x1 * x3 + y1 * y3) / (math.sqrt(x1 ** 2 + y1 ** 2) * math.sqrt(x3 ** 2 + y3 ** 2))
    )

    cv2.putText(image, str(cos_alpha), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv2.putText(image, str(arccos_alpha), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # cv2.putText(image, "x:{}, y:{}, z:{}".format(dots[15].x, dots[15].y, dots[15].z), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    # # print(landmarks[14])

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
