import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Adding fully-connected layer
x = Dense(1024, activation='relu')(x)

# Add a logistic layer with 2 classes (Non Drowsy and drowsy)
predictions = Dense(2, activation='softmax')(x)

# Define the model that we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers - which were randomly initialized
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with an Adam optimizer and a very slow learning rate.
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define our example directories and files
train_dir = 'Driver Drowsiness Dataset (DDD)\Training'
validation_dir = 'Driver Drowsiness Dataset (DDD)\Validation'

# Add data augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=50, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=50, class_mode='categorical')

# Training the model
history = model.fit(train_generator, steps_per_epoch=100, epochs=5, validation_data = validation_generator, validation_steps=50)

# Save the model
model.save('Validation_Loss&Acc_5_28112023.h5')

# Calculate the number of validation steps based on the number of validation samples and batch size
total_validation_samples = len(validation_generator.filenames)
batch_size = 50
validation_steps = total_validation_samples // batch_size

# Calculate and print the validation accuracy and loss
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_steps)
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_accuracy)

# Evaluate the model on the validation set and create a confusion matrix
y_true = []
y_pred = []

for i in range(validation_steps):
    x_val, y_val = next(validation_generator)
    y_true.extend(np.argmax(y_val, axis=1))  # True labels
    y_pred.extend(np.argmax(model.predict(x_val), axis=1))  # Predicted labels

# confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non Drowsy', 'Drowsy'], yticklabels=['Non Drowsy', 'Drowsy'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Plot training and validation accuracy and loss
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.title('Training and Validation Metrics')
plt.legend()

plt.show()
