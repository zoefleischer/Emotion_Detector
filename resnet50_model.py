#------------------RESNET50 MODEL-----------------

import cv2
import os
import numpy as np
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense


# SPLIT
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
    return images

smile = load_images_from_folder(r"C:\Users\InsertPath")
rage= load_images_from_folder(r"C:\Users\InsertPath")
cry= load_images_from_folder(r"C:\Users\InsertPath")
orgasm= load_images_from_folder(r"C:\Users\InsertPath")



X_smile= smile
X_rage= rage
X_cry=cry
X_excitement=excitement

y_cry = np.full(len(cry),0)
y_smile = np.full(len(smile),1)
y_rage= np.full(len(rage),2)
y_excitement= np.full(len(excitement),3)

y= list(y_cry)+list(y_smile)+list(y_rage)+list(y_excitement)
X= X_cry+X_smile+X_rage+X_excitement

X=[i/255 for i in X]


#TRAIN

model= Sequential()
model.add(tf.keras.applications.ResNet50V2(
    include_top=False, classes=4,
    input_shape=(450, 450, 3)))
model.layers[0].trainable = False
model.add(Dropout(0.25))
model.add(Dense(127))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(207))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(4,activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(
        np.array(X), np.array(y),
        epochs=5000,
        batch_size=10,
        validation_split=0.15,
        shuffle=True,
        workers=10,
        use_multiprocessing=True)


model.save('model.h5')