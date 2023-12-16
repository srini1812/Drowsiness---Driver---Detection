import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the trained model
model = load_model('Validation_Loss&Acc_5_28112023.h5')

# Load an image you want to classify (change 'path_to_image.jpg' to the path of your image)
img_path = r'Driver Drowsiness Dataset (DDD)\Testing\Non Drowsy\b0344.png'
#img_path = r'Driver Drowsiness Dataset (DDD)\Testing\Drowsy\A0001.png'

img = image.load_img(img_path, target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.0  # Normalize the image

# Prediction
result = model.predict(img)

# The result will be an array of probabilities for each class (open and closed eyes)
# The class with the highest probability is the predicted class
if result[0][0] > result[0][1]:
    prediction = 'Drowsy'
else:
    prediction = 'Non Drowsy'

print(f'Prediction: {prediction}')