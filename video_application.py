# -----------------VIDEO APPLICATION OF MODEL----------------


import numpy as np
import cv2
import sys
import tensorflow as tf
import time

model = tf.keras.models.load_model(r"C:\Users\InsertPath")
face_cascade = cv2.CascadeClassifier(r"C:\Users\InsertPath")
cap = cv2.VideoCapture(0)

emotions = ('sad', 'happy', 'angry', 'excited')

while True:
    ret, img = cap.read()
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = frame[y:y + w, x:x + h]
        roi_gray = cv2.resize(frame, (200, 200))
        img_pixels = np.array(roi_gray) / 255
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = model.predict(img_pixels)
        print(predictions)
        max_index = np.argmax(predictions[0])
        print(max_index)

        predicted_emotion = emotions[max_index] + ' ' + str(int(predictions[0][max_index] * 100)) + '%'

        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('img', img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()