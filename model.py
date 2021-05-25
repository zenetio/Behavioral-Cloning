import os
import cv2
import csv
import numpy as np
from PIL import Image
#from numpy.testing import verbose
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math

# path to generated images
data_path = '../../../term1_data/'
# csv file generated
csv_file = 'driving_log.csv'
img_file = 'IMG'
csv_path = os.path.join(data_path, csv_file)
img_path = os.path.join(data_path, img_file)

# Read the image from directory images
def process_image(file_path):
    filename = os.path.basename(file_path)
    cur_path = os.path.join(img_path, filename)
    return np.asarray(Image.open(cur_path))

# Create a generator that will iterate over the image set
# returning a suffled batch of images, plus its augmented 
# left and right camera images
def generator(samples, batch_size=32):
    n_samples = len(samples)
    while 1:    # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2    # parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                img_center = process_image(batch_sample[0])
                img_left   = process_image(batch_sample[1])
                img_right  = process_image(batch_sample[2])

                # add images and angles to data set
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
            # create X,y dataset
            X , y = np.array(images), np.array(angles)
            X, y = shuffle(X, y)
            yield X, y

# Read the csv file
lines = []
with open(csv_path) as fp:
    reader = csv.reader(fp)
    for line in reader:
        lines.append(line)

# Split the image set in training and validation set where
# training has 80% and validation, 20%
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# hyperparameters
batch_size = 64

# Create the generators
train_gen = generator(train_samples, batch_size=batch_size)
val_gen   = generator(validation_samples, batch_size=batch_size)

images = []
measurements = []

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Cropping2D, Lambda

# Model architecture
model = tf.keras.Sequential()
# Image processing
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Cropping to remove unnecessary image located above the lane like trees, sky
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
# Checkpoint
checkpoint_path = "./model.{epoch:02d}-{val_loss:.4f}.hdf5"

# Optimizer
l_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)

# callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    verbose=2)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.1,
    patience=3, min_lr=1E-7, verbose=2)
    
es_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=5, verbose=2,
    mode='auto', baseline=None, restore_best_weights=True)

model.compile(loss='mse', optimizer=optimizer, metrics='accuracy')

# Train the model
history = model.fit(train_gen,
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                    validation_data = val_gen,
                    validation_steps=math.ceil(len(validation_samples)/batch_size),
                    epochs=20, 
                    callbacks=[cp_callback, reduce_lr_cb, es_cb],
                    verbose=1)

#model.save('model.h5')

from matplotlib import pyplot as plt

### print the keys contained in the history object
print(history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
