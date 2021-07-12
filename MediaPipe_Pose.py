import time
import cv2
import mediapipe as mp
import numpy as np


def calculate_angle(a, b, c):
    """Calculation of the angle between the vectors ab and bc"""
    a = np.array(a)
    b = np.array(b)  # joint point
    c = np.array(c)
    # Get angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    # Set angle sign to be positive
    if angle > 180.0:
        angle = 360 - angle
    return angle


def visualize_angle(image, joint_point, cam_resolution):
    """Visializing angle value near the joint landmark (b)"""
    to_show = cv2.putText(image,
                          str(int(angle)),
                          tuple(np.multiply(joint_point, cam_resolution).astype(int)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1
                          )
    return to_show


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

start_time = current_time = 0

# Webcam input:
cap = cv2.VideoCapture(0)
# Local disk movie input
# cap = cv2.VideoCapture('c:/TASK 1 - POSE ESTIMATION/GolfDB_SwingNet/videos_160/1063.mp4')
# Local disk image input

# Getting cam resolution
W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'Input resolution: {H}x{W}')


# Default resolutions of the frame are obtained.
# The default resolutions are system dependent.
# Convert the resolutions from float to int.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('test_golf_1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))


pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(
    #     image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display fps
    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time
    cv2.putText(image, 'fps: ' + str(int(fps)), (5, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    # Extracting and displaying angles of different parts of the body
    try:
        landmarks = results.pose_landmarks.landmark

        # The angle between left shoulder and left wrist in left elbow joint
        a = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        b = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        c = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # Calculate angle
        angle = calculate_angle(a, b, c)
        # Visualize angle
        visualize_angle(image, b, cam_resolution=[W, H])

        # The angle between right shoulder and right wrist in right elbow joint
        a = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        b = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        c = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        # Calculate angle
        angle = calculate_angle(a, b, c)
        # Visualize angle
        visualize_angle(image, b, cam_resolution=[W, H])

        # The angle between right hip and right elbow in right shoulder joint
        a = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        b = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        c = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        # Calculate angle
        angle = calculate_angle(a, b, c)
        # Visualize angle
        visualize_angle(image, b, cam_resolution=[W, H])

        # The angle between left hip and left elbow in left shoulder joint
        a = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        b = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        c = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        # Calculate angle
        angle = calculate_angle(a, b, c)
        # Visualize angle
        visualize_angle(image, b, cam_resolution=[W, H])

        # The angle between left wrist and left hip in left shoulder joint
        # # ???????
        # # ???????  doesn't work!!! (((
        # # ???????
        # a = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
        #      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # b = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        #      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        # c = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        #      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        # # Calculate angle
        # angle = calculate_angle(a, b, c)
        # # Visualize angle
        # visualize_angle(image, b, cam_resolution=[W, H])


    except:
        pass

    # Display landmarks with connecting lines
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                              )

    out.write(image)


    # Display cam capturing itself
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()