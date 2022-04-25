# Training script, attempt 1.

import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/cuDNN/bin")

import pandas as pd
from skimage.io import imread # Used for image processing
from skimage.transform import resize # Used for image processing
import json
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


class FacialDataGenerator(tf.keras.utils.Sequence):
    # Create a generator that returns batches of mappings of data.
    # It maps X to Y, where X is an image and Y is 80 values representing
    # the coordinates of 40 points around a subject's eyes.

    def __init__(self, image_dir, annotation_dir, batch_size=32, target_size=(150, 150), shuffle=True):
        # Initialize the generator.
        # Directories containing face images and point annotations, respectively.
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        self.filenames = [name[:-4] for name in os.listdir(image_dir)]
        print(self.filenames)

        self.on_epoch_end()


    def __len__(self):
        # Returns the necessary number of batches to complete an epoch.
        return math.ceil(len(self.filenames) / self.batch_size)


    def __getitem__(self, index):
        # Return one batch of data.
        img_batch = [n + '.jpg' for n in self.filenames[index * self.batch_size : (index+1) * self.batch_size]]
        ann_batch = [n + '.txt' for n in self.filenames[index * self.batch_size : (index+1) * self.batch_size]]

        pts_batch = []
        for ann in ann_batch:
            with open(os.path.join(self.annotation_dir, ann), 'r') as ann_file:
                lines = ann_file.readlines()
                points = []
                for i in range(115, 155, 5):
                    line = lines[i]
                    parts = line[:-1].split(' , ')
                    points.append(float(parts[0]))
                    points.append(float(parts[1]))
                pts_batch.append(points)

        X = np.array([resize(imread(os.path.join(self.image_dir, img)), self.target_size) for img in img_batch])
        Y = np.array(pts_batch)

        return X, Y


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.filenames)


    #def __data_generation(self, list_paths, list_paths_wo_ext):
        # Not strictly necessary.
        #pass




model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(16)
])

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

train_dir = "HELEN/"
validation_dir = "HELEN_test/test/"
annotation_dir = "HELEN_annotation/"

train_gen = FacialDataGenerator(train_dir, annotation_dir)
validation_gen = FacialDataGenerator(validation_dir, annotation_dir)

history = model.fit(train_gen,
                    validation_data=validation_gen,
                    epochs=1)

# Predict on a picture of me
from keras.preprocessing import image
img = image.load_img("me.jpg", target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
results = model.predict(images, batch_size=10)



print(results)
