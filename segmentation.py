import numpy as np
import tensorflow as tf
import cv2
import time
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU
from keras.layers import Conv2DTranspose, BatchNormalization

import matplotlib.pyplot as plt
import keras



IMSIZE = (384, 384, 3)
LEARNING_RATE = 1e-3

color_codes = {'golf_ball': [0, 127, 0], 'golf_club': [127,0,0], 'none': [0, 0, 0]}
color_codes_list = [color_codes[i] for i in color_codes.keys()]



KERNEL = tf.keras.initializers.HeNormal()

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ReLU, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

KERNEL = tf.keras.initializers.HeNormal()


def segnet(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    # Encoder
    # x, p_1 = encoder(inputs, 2, 64, 3, (2, 2), 2)(inputs)
    x = Conv2D(64, 3, padding='same', kernel_initializer=KERNEL)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    p_1 = x
    # x, p_2 = encoder(inputs, 2, 128, 3, (2, 2), 2)(inputs)
    x = Conv2D(128, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    p_2 = x
    # x, p_3 = encoder(inputs, 2, 256, 3, (2, 2), 2)(inputs)
    x = Conv2D(256, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    p_3 = x
    # x, p_4 = encoder(inputs, 2, 512, 3, (2, 2), 2)(inputs)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    p_4 = x

    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    p_5 = x

    # Decoder
    o = tf.concat([x, p_5], axis=3)
    x = UpSampling2D(2, interpolation='nearest')(o)
    # x = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, use_bias=False, padding='same')(o)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    o = tf.concat([x, p_4], axis=3)
    x = UpSampling2D(2, interpolation='nearest')(o)
    # x = tf.keras.layers.Conv2DTranspose(512, 2, strides=2, use_bias=False, padding='same')(o)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(512, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    o = tf.concat([x, p_3], axis=3)
    x = UpSampling2D(2, interpolation='nearest')(o)
    # x = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, use_bias=False, padding='same')(o)
    x = Conv2D(256, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(256, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    o = tf.concat([x, p_2], axis=3)
    x = UpSampling2D(2, interpolation='nearest')(o)
    # x = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, use_bias=False, padding='same')(o)
    x = Conv2D(128, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(128, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    o = tf.concat([x, p_1], axis=3)
    x = UpSampling2D(2, interpolation='nearest')(o)
    # x = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, use_bias=False, padding='same')(o)
    x = Conv2D(64, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, padding='same', kernel_initializer=KERNEL)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    outputs = Conv2D(n_classes, 1, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model


model = segnet(IMSIZE, n_classes=len(color_codes.keys()))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=['accuracy', tf.keras.metrics.MeanIoU(3)]
)

model.load_weights('C:/TASK 1 - POSE ESTIMATION/unet_weights.h5')
cap = cv2.VideoCapture('C:/TASK 1 - POSE ESTIMATION/makro_golf_impact.mp4')


# Recording output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('test_golf_1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Movie ended")
        # If loading a video, use 'break' instead of 'continue'.
        break

    # Predict mask
    x = np.expand_dims(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), IMSIZE[:2]), 0)
    prediction = model.predict(x / 255.0)

    club = prediction[0][:, :, 0]
    # ball = prediction[0][:, :, 1]

    club = np.where(club > 0.5, 255, 0)
    # ball = np.where(ball > 0.5, 255, 0)


    image = cv2.resize(image, IMSIZE[:2])
    club = club.astype(np.uint8)
    club = np.expand_dims(club, 2)
    club = cv2.cvtColor(club, cv2.COLOR_GRAY2BGR)

    # Shows 2 separated frames
    # cv2.imshow('image', image)
    # cv2.imshow('golf club segmentation', club)

    # Shows 2 connected frames
    # numpy_vertical = np.vstack((image, club, ball))
    # numpy_horizontal = np.hstack((image, club))
    # # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
    numpy_horizontal_concat = np.concatenate((image, club), axis=1)
    # cv2.imshow('Numpy Vertical', numpy_vertical)
    # cv2.imshow('Numpy Horizontal', numpy_horizontal)
    # cv2.imshow('Numpy Vertical Concat', numpy_vertical_concat)
    cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)

    # cv2.imshow('Masks', club)
    # out.write(image)
    out.write(numpy_horizontal_concat)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()