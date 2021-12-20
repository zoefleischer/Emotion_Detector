# -----------SEQUENTIAL NEURAL NETWORK------------------

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization

model = Sequential()
model.add(Conv2D(32,kernel_size=2, input_shape=(450, 450, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(127,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(270,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(127,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(270,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,kernel_size=2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
# how many output categories there are

model.compile(loss='sparse_categorical_crossentropy',
              # expects labelEncoded input
              # binary_crossentropy expects binary output
              # categorical_crossentropy expects one-hot-encoded input
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(
        np.array(X), np.array(y),
        epochs=5000,
        batch_size=10,
        validation_split=0.12,
        shuffle=True,
        workers=10,
        use_multiprocessing=True)


model.save('model_nn.h5')