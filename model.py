import numpy as np
import tensorflow as tf
import cv2
import time


def model():
    inputs = tf.keras.layers.Input(shape=IMSIZE)
    x = basemodel(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation=None)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# BATCH_SIZE = 64
IMSIZE = (224, 224, 3)
LEARNING_RATE = 1e-5

basemodel = tf.keras.applications.ResNet50(include_top=False,
                                           weights=None,
                                           input_shape=IMSIZE
                                           )
model = model()
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=LEARNING_RATE),
              loss=tf.losses.MeanSquaredError(),
              metrics=tf.metrics.RootMeanSquaredError(name='rmse')
              )
model.load_weights('C:/TASK 1 - POSE ESTIMATION/weights_golf_club_localization.h5')

########################  ####################  ########################
########################  DELETE ME AFTER TEST  ########################
########################  ####################  ########################
cap = cv2.VideoCapture("C:/TASK 1 - POSE ESTIMATION/swing_slowmotion_video.mp4")
# cap = cv2.VideoCapture(0)
start_time = current_time = 0



# # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
# # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
# # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
# def calculate_angle(a, b, c):
#     """Calculation of the angle between the vectors ab and bc"""
#     a = np.array(a)
#     b = np.array(b)  # joint point
#     c = np.array(c)
#     # Get angle
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     # Set angle sign to be positive
#     if angle > 180.0:
#         angle = 360 - angle
#     return angle
#
#
# def visualize_angle(image, joint_point, cam_resolution):
#     """Visializing angle value near the joint landmark (b)"""
#     to_show = cv2.putText(image,
#                           str(int(angle)),
#                           tuple(np.multiply(joint_point, cam_resolution).astype(int)),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1
#                           )
#     return to_show
#
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
# # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
# # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE



# Recording output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('test_golf_1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

########################  ####################  ########################
########################  DELETE ME AFTER TEST  ########################
########################  ####################  ########################

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Movie ended")
        # If loading a video, use 'break' instead of 'continue'.
        break

    # Display fps
    current_time = time.time()
    fps = 1 / (current_time - start_time + 0.001)
    start_time = current_time
    cv2.putText(image, 'fps: ' + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)



    # # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
    # # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
    # # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
    # image.flags.writeable = False
    # results = pose.process(image)
    # # Draw the pose annotation on the image.
    # image.flags.writeable = False
    #
    # # Extracting and displaying angles of different parts of the body
    # try:
    #     landmarks = results.pose_landmarks.landmark
    #
    #     # The angle between left shoulder and left wrist in left elbow joint
    #     a = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
    #          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    #     b = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
    #          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    #     c = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
    #          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    #     # Calculate angle
    #     angle = calculate_angle(a, b, c)
    #     # Visualize angle
    #     visualize_angle(image, b, cam_resolution=[W, H])
    #
    #     # The angle between right shoulder and right wrist in right elbow joint
    #     a = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
    #          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    #     b = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
    #          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    #     c = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
    #          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    #     # Calculate angle
    #     angle = calculate_angle(a, b, c)
    #     # Visualize angle
    #     visualize_angle(image, b, cam_resolution=[W, H])
    #
    #     # The angle between right hip and right elbow in right shoulder joint
    #     a = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
    #          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    #     b = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
    #          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    #     c = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
    #          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    #     # Calculate angle
    #     angle = calculate_angle(a, b, c)
    #     # Visualize angle
    #     visualize_angle(image, b, cam_resolution=[W, H])
    #
    #     # The angle between left hip and left elbow in left shoulder joint
    #     a = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
    #          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    #     b = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
    #          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    #     c = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
    #          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    #     # Calculate angle
    #     angle = calculate_angle(a, b, c)
    #     # Visualize angle
    #     visualize_angle(image, b, cam_resolution=[W, H])
    # except:
    #     pass
    #
    # # Display landmarks with connecting lines
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2, circle_radius=2),
    #                           mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    #                           )
    # # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
    # # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE
    # # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE # MEDIAPIPE



    # Predict coordinates
    resized_img = cv2.resize(image, IMSIZE[:2])     # resize for for input to the model
    resized_img = np.expand_dims(resized_img, 0)    # add the first axis
    resized_normalized_img = resized_img / 255.0      # normalize input image
    pred_coordinates = model.predict(resized_normalized_img)
    pred_coordinates = pred_coordinates[0]
    # print(pred_coordinates)
    # print(image.shape, resized_img.shape)

    # Draw dots on the image
    x1 = int(pred_coordinates[0] * (image.shape[1] / resized_img.shape[2]))
    y1 = int(pred_coordinates[1] * (image.shape[0] / resized_img.shape[1]))
    x2 = int(pred_coordinates[2] * (image.shape[1] / resized_img.shape[2]))
    y2 = int(pred_coordinates[3] * (image.shape[0] / resized_img.shape[1]))
    # print(pred_coordinates)
    image = cv2.circle(image, (x1, y1), radius=5, color=(255, 0, 0), thickness=10)
    image = cv2.circle(image, (x2, y2), radius=5, color=(0, 255, 0), thickness=10)

    cv2.imshow('Test frame', image)
    out.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()