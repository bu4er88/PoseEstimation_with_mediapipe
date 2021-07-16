import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pandas as pd
import seaborn as sns
import time


def model():
    inputs = tf.keras.layers.Input(shape=IMSIZE)
    x = basemodel(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation=None)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


# BATCH_SIZE = 64
IMSIZE = (224, 224, 3)
LEARNING_RATE = 1e-3

basemodel = tf.keras.applications.ResNet50(include_top=False,
                                           weights='imagenet',
                                           input_shape=IMSIZE
                                           )
basemodel.trainable = True
model = model()
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=LEARNING_RATE),
              loss=tf.losses.MeanSquaredError(),
              metrics=tf.metrics.RootMeanSquaredError(name='rmse')
              )
model.load_weights('C:/TASK 1 - POSE ESTIMATION/weights_golf_club_localization.h5')

########################  ####################  ########################
########################  DELETE ME AFTER TEST  ########################
########################  ####################  ########################
cap = cv2.VideoCapture("C:/TASK 1 - POSE ESTIMATION/swing_slowmotion_video_1.mp4")
# cap = cv2.VideoCapture(0)
start_time = current_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Movie ended")
        # If loading a video, use 'break' instead of 'continue'.
        break

    # Display fps
    # current_time = time.time()
    # fps = 1 / (current_time - start_time + 0.001)
    # start_time = current_time
    # cv2.putText(image, 'fps: ' + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    # Predict coordinates
    resized_img = cv2.resize(image, IMSIZE[:2])     # resize for for input to the model
    resized_img = np.expand_dims(resized_img, 0)    # add the first axis
    resized_normalized_img = resized_img /255.0      # normalize input image
    pred_coordinates = model.predict(resized_normalized_img)
    pred_coordinates = pred_coordinates[0]
    # print(pred_coordinates)
    # print(image.shape, resized_img.shape)

    # Draw dots on the image
    x1 = int(pred_coordinates[0] * (image.shape[1] / resized_img.shape[2]))
    y1 = int(pred_coordinates[1] * (image.shape[0] / resized_img.shape[1]))
    x2 = int(pred_coordinates[2] * (image.shape[1] / resized_img.shape[2]))
    y2 = int(pred_coordinates[3] * (image.shape[0] / resized_img.shape[1]))
    print(pred_coordinates)
    image = cv2.circle(image, (x1, y1), radius=10, color=(255, 0, 0), thickness=10)
    image = cv2.circle(image, (x2, y2), radius=10, color=(0, 255, 0), thickness=10)


    cv2.imshow('Test frame', image)
    # out.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# out.release()
cap.release()
# cv2.destroyAllWindows()