# ----------Testing the Model on Pictures-----------


model = tf.keras.models.load_model(r"C:\Users\InsertPath")

import os

for x in os.listdir(r"C:\Users\InsertPath"):
    img_pred = tf.keras.preprocessing.image.load_img("C:\\Users\\InsertPath" + str(x), target_size=(200, 200))
    img_pred = tf.keras.preprocessing.image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)

    result = model.predict(img_pred)
    print(result)
    print('Pic', x, 'is:')

    if result[0][0] == 1:
        prediction = 'Happy!'

    elif result[0][0] == 0:
        prediction = 'Sad!'

    elif result[0][0] == 2:
        prediction = 'Angry!'

    elif result[0][0] == 3:
        prediction = 'Excited!'

    else:
        prediction = 'Error'

    print(prediction)