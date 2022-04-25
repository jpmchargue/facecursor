# Training script, attempt 1.

import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/cuDNN/bin")

import pandas as pd
from skimage.io import imread # Used for image processing
#from skimage.transform import resize # Used for image processing
import json
import numpy as np
import math
import cv2
import sys
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

class FacialDataGenerator(tf.keras.utils.Sequence):
    # Create a generator that returns batches of mappings of data.
    # It maps X to Y, where X is an image and Y is 80 values representing
    # the coordinates of 40 points around a subject's eyes.

    def __init__(self, xml, batch_size=10, target_size=(224, 224, 3), shuffle=True):
        # Initialize the generator.
        # Directories containing face images and point annotations, respectively.
        self.xml = xml
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle

        ibug_directory = "ibug/ibug"
        self.filenames = []
        self.boxes = []
        self.landmarks = []
        xmlroot = ET.parse(os.path.join(ibug_directory, self.xml)).getroot()
        for image in xmlroot[2]:
            self.filenames.append(os.path.join(ibug_directory, image.attrib["file"]))
            box = image[0]
            self.boxes.append(box.attrib)
            l = []
            for p in range(36, 48):
                l.append(float(box[p].attrib['x']))
                l.append(float(box[p].attrib['y']))
            self.landmarks.append(l)

        self.indices = np.arange(len(self.filenames))
        self.on_epoch_end()


    def __len__(self):
        # Returns the necessary number of batches to complete an epoch.
        return math.floor(len(self.filenames) / self.batch_size)


    def __getitem__(self, index):
        # Return one batch of data.
        batch_indices = self.indices[(index) * self.batch_size:(index+1) * self.batch_size]
        img_batch = []
        lmk_batch = []

        for i in range(self.batch_size):
            newindex = batch_indices[i]

            img = cv2.imread(self.filenames[newindex])
            shape = img.shape
            lmk = self.landmarks[newindex]
            box = self.boxes[newindex]

            x = max(int(box['left']), 0)
            y = max(int(box['top']), 0)
            w = int(box['width'])
            h = int(box['height'])
            #print(f"{x}, {y}, {w}, {h}")
            img = img[y:y+h, x:x+w]
            resized = cv2.resize(img, self.target_size)

            # Randomly translate and rotate image to avoid memorization
            angle = (random.random() * 24.0) - 12.0 # degrees
            rotated = rotate_image(resized, angle)

            lmk_rotated = []
            for i in range(12): # rotate the landmarks
                coord = (lmk[2*i] - (x + (w/2)), lmk[(2*i)+1] - (y + (h/2)))
                rads = math.radians(-angle)
                newcoord = ((coord[0]*math.cos(rads)) + (coord[1]*-math.sin(rads)), (coord[0]*math.sin(rads)) + (coord[1]*math.cos(rads)))
                lmk_rotated.append(newcoord[0] + (x + (w/2)))
                lmk_rotated.append(newcoord[1] + (y + (h/2)))

            lmk_adjusted = []
            for p in range(len(lmk)):
                if p % 2 == 0: # x coordinate
                    lmk_adjusted.append(((lmk_rotated[p] - x) / w) - 0.5)
                else: # y coordinate
                    lmk_adjusted.append(((lmk_rotated[p] - y) / h) - 0.5)
            #print(lmk_adjusted)

            img_batch.append(rotated)
            lmk_batch.append(lmk_adjusted)

        #print(self.filenames[newindex])
        #print(img.shape)
        X = np.array(img_batch)
        Y = np.array(lmk_batch)

        return X, Y


    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)


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
dense = tf.keras.layers.Dense(24)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = dense(x)
model = tf.keras.Model(inputs, outputs)

base_lr = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
              loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM), # custom loss function?
              metrics=['accuracy'])

#model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error', metrics=['acc'])



#train_dir = "HELEN/"
#validation_dir = "HELEN_test/test/"
#annotation_dir = "HELEN_annotation/"

train_gen = FacialDataGenerator("labels_ibug_300W_train.xml", target_size=(224, 224, 3))
validation_gen = FacialDataGenerator("labels_ibug_300W_test.xml", target_size=(224, 224, 3))

# Testing the generators
images, landmarks = next(iter(train_gen))
print(images.shape)
print(landmarks.shape)
#features = model(images)
#print(features.shape)
if False:
    image = images[0]
    landmarks = [224 * (l+0.5) for l in landmarks[0]]
    for i in range(12):
        cv2.circle(image, (round(landmarks[2*i]), round(landmarks[(2*i)+1])), radius=2, color=(255, 0, 0), thickness=-1)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if len(sys.argv) > 1 and sys.argv[1] == "train":
    model.load_weights('./checkpoints/my_checkpoint')
    history = model.fit(train_gen,
                        validation_data=validation_gen,
                        epochs=25)
    model.save_weights('./checkpoints/my_checkpoint')
else:
    model.load_weights('./checkpoints/my_checkpoint')


print("*** TRAINED ***")




# Predict on a picture of me
#from keras.preprocessing import image
#img = image.load_img("me.jpg", target_size=(224, 224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)

#images = np.vstack([x])
#results = model.predict(images, batch_size=10)[0]



#img = cv2.imread('HELEN/115774957_1.jpg')


#print(wrapped.shape)
if True:
    images, landmarks = next(iter(validation_gen))
    img = images[0]
    landmarks = landmarks[0]
    wrapped = np.array([img])
else:
    test_image = "wink_crop.jpg"
    img = imread(test_image)
    landmarks = [0] * 24
    wrapped = np.array([cv2.resize(img, (224, 224, 3))])
results = model(wrapped).numpy()[0]
dimensions = img.shape

print("*** PREDICTED ***")
print(results)

true_xs = [dimensions[1] * (x + 0.5) for x in landmarks[::2]]
true_ys = [dimensions[0] * (y + 0.5) for y in landmarks[1::2]]
xs = [dimensions[1] * (x + 0.5) for x in results[::2]]
ys = [dimensions[0] * (y + 0.5) for y in results[1::2]]
#img = cv2.imread(test_image)
for i in range(len(xs)):
    cv2.circle(img, (round(true_xs[i]), round(true_ys[i])), radius=2, color=(0, 255, 0), thickness=-1)
    cv2.circle(img, (round(xs[i]), round(ys[i])), radius=2, color=(255, 0, 0), thickness=-1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



print(results)
