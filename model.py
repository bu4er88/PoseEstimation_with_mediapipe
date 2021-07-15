import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pandas as pd
import seaborn as sns
import time



BATCH_SIZE = 32
IMSIZE = (224, 224, 3)
LEARNING_RATE = 5e-4


basemodel = tf.keras.applications.ResNet50(include_top=False,
                                           weights='imagenet',
                                           input_shape=IMSIZE
                                           )


def model():
    inputs = tf.keras.layers.Input(shape=IMSIZE)
    x = basemodel(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation=None)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


model = model()
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=LEARNING_RATE),
              loss=tf.losses.MeanSquaredError(),
              metrics=tf.metrics.RootMeanSquaredError(name='rmse')
              )

model.load_weights('C:/TASK 1 - POSE ESTIMATION/weights_golf_club.h5')

########################  ####################  ########################
########################  DELETE ME AFTER TEST  ########################
########################  ####################  ########################
cap = cv2.VideoCapture("C:/TASK 1 - POSE ESTIMATION/swing_slowmotion_video.mp4")

start_time = current_time = 0

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


    # FINISHED HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # SHAPE = image.shape
    #
    # resized_img = cv2.resize(image, IMSIZE[:2])
    # resized_img = np.expand_dims(resized_img, 0)
    # print(resized_img.shape)
    #
    # # Draw dots on the image
    # pred_coordinates = model.predict(resized_img)
    # pred_coordinates = pred_coordinates[0]
    # print(pred_coordinates[0])
    #
    # x1 = int(pred_coordinates[0] * (SHAPE[1] / image.shape[1]))
    # y1 = int(pred_coordinates[1])
    # x2 = int(pred_coordinates[2])
    # y2 = int(pred_coordinates[3])
    #
    # image = cv2.circle(image, (x1, y1), radius=1, color=(255, 0, 0), thickness=5)
    # image = cv2.circle(image, (x2, y2), radius=1, color=(0, 255, 0), thickness=5)


    cv2.imshow('Test frame', image)
    # out.write(image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# out.release()
cap.release()
# cv2.destroyAllWindows()