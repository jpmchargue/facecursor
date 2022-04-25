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
import cv2
import sys

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


class FacialDataGenerator(tf.keras.utils.Sequence):
    # Create a generator that returns batches of mappings of data.
    # It maps X to Y, where X is an image and Y is 80 values representing
    # the coordinates of 40 points around a subject's eyes.

    def __init__(self, image_dir, annotation_dir, target_size=(224, 224, 3), shuffle=True):
        # Initialize the generator.
        # Directories containing face images and point annotations, respectively.
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.target_size = target_size

        self.shuffle = shuffle

        self.filenames = [name[:-4] for name in os.listdir(image_dir)]

        self.on_epoch_end()


    def __len__(self):
        # Returns the necessary number of batches to complete an epoch.
        return len(self.filenames)


    def __getitem__(self, index):
        # Return one batch of data.
        img = imread(os.path.join(self.image_dir, self.filenames[index] + '.jpg'))
        dimensions = img.shape
        X = np.array([resize(img, self.target_size)])

        ann = os.path.join(self.annotation_dir, self.filenames[index] + '.txt')
        points = []
        with open(ann, 'r') as ann_file:
            lines = ann_file.readlines()
            for i in range(115, 155, 5):
                line = lines[i]
                parts = line[:-1].split(' , ')
                x, y = float(parts[0]), float(parts[1])
                points.append(self.target_size[1] * (x / dimensions[1]))
                points.append(self.target_size[0] * (y / dimensions[0]))
        print(points)

        X = np.array([resize(img, self.target_size)])
        Y = np.array([points])

        return X, Y


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.filenames)


    #def __data_generation(self, list_paths, list_paths_wo_ext):
        # Not strictly necessary.
        #pass


# BUILDING THE MODEL
# MobileNet requires 224x224 images
base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                               weights='imagenet')
base_model.trainable = False
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dense = tf.keras.layers.Dense(16)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = dense(x)
model = tf.keras.Model(inputs, outputs)

base_lr = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
              loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
              metrics=['accuracy'])

#model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error', metrics=['acc'])



train_dir = "HELEN/"
validation_dir = "HELEN_test/test/"
annotation_dir = "HELEN_annotation/"

train_gen = FacialDataGenerator(train_dir, annotation_dir, target_size=(224, 224))
validation_gen = FacialDataGenerator(validation_dir, annotation_dir, target_size=(224, 224))

images, landmarks = next(iter(train_gen))
print(images.shape)
print(landmarks.shape)
features = model(images)
print(features.shape)


if len(sys.argv) > 1 and sys.argv[1] == "train":
    model.load_weights('./checkpoints/my_checkpoint')
    history = model.fit(train_gen,
                        validation_data=validation_gen,
                        epochs=1)
    model.save_weights('./checkpoints/my_checkpoint')
else:
    model.load_weights('./checkpoints/my_checkpoint')


print("*** TRAINED ***")




# Predict on a picture of me
from keras.preprocessing import image
img = image.load_img("me.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
results = model.predict(images, batch_size=10)[0]
print("*** PREDICTED ***")
print(results)


img = cv2.imread('wink.jpg')
dimensions = img.shape
xs = [dimensions[1] * (x/224.0) for x in results[::2]]
ys = [dimensions[0] * (y/224.0) for y in results[1::2]]
for i in range(len(xs)):
    cv2.circle(img, (round(xs[i]), round(ys[i])), radius=2, color=(255, 0, 0), thickness=-1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



print(results)
