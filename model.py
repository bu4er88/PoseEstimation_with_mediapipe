import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
import pandas as pd
import seaborn as sns



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
