# Drowsiness-Driver-detection
This script is designed to train and evaluate a deep learning model for detecting driver drowsiness using the VGG16 pre-trained network as a base model. It uses the `Keras` and `TensorFlow` frameworks, and includes steps to:

1. **Load and Preprocess Data:**
   - The dataset is divided into training, validation, and testing sets, containing images of "Drowsy" and "Non-Drowsy" individuals.
   - Data augmentation is applied to the training set to introduce variations in the data and avoid overfitting.
   - The `ImageDataGenerator` is used to rescale the pixel values and apply real-time augmentation like rotation, zoom, and flipping.

2. **Model Construction:**
   - **VGG16 Base:** A pre-trained VGG16 model (without the top fully connected layers) is used as a feature extractor. The model is initialized with ImageNet weights.
   - **Custom Layers:** Global average pooling, fully connected (Dense) layers, and a logistic layer (with 2 classes) are added to fine-tune the model for binary classification (Drowsy vs. Non-Drowsy).
   - Only the top layers are trainable, while the VGG16 base remains frozen, preventing modification of its pre-trained weights.

3. **Model Compilation & Training:**
   - The model is compiled with the Adam optimizer and a learning rate of `0.0001`. The loss function used is `categorical_crossentropy`, which is suitable for multi-class classification.
   - The model is trained for 5 epochs using augmented training data and evaluated on a separate validation dataset.

4. **Saving and Evaluating the Model:**
   - After training, the model is saved in an HDF5 file (`Validation_Loss&Acc_5_28112023.h5`).
   - Validation accuracy and loss are calculated.
   - Predictions are made on the validation dataset, and a **confusion matrix** is computed to visualize classification performance.

5. **Plotting and Visualization:**
   - A confusion matrix heatmap shows how well the model distinguishes between Drowsy and Non-Drowsy classes.
   - The training and validation accuracy and loss curves are plotted to observe model performance over time.

6. **Real-Time Image Classification:**
   - The script allows testing the model on a specific image (`Non Drowsy` or `Drowsy`), where the image is preprocessed and passed through the trained model for prediction.

### Detailed Explanation for GitHub Repository:
When uploading this project to GitHub, ensure the repository includes the following:

#### 1. **Project Structure:**
   - `Driver Drowsiness Dataset (DDD)/`: Include the dataset folder, if public.
     - `Training/`: Training set images divided into subfolders for 'Drowsy' and 'Non Drowsy'.
     - `Validation/`: Validation set images for evaluation.
     - `Testing/`: Testing set images for final evaluation.
   - `model/`: Folder containing the saved model file, e.g., `Validation_Loss&Acc_5_28112023.h5`.
   - `scripts/`: Folder with the script file(s) for model training, evaluation, and testing.

#### 2. **README.md:**
   - **Project Overview:** Explain the goal of the project, i.e., detecting driver drowsiness using VGG16.
   - **Dataset Description:** Provide a description of the dataset, including how the images are structured, and mention if the dataset is public or needs to be uploaded separately.
   - **Installation Instructions:**
     - List the dependencies (Keras, TensorFlow, NumPy, etc.).
     - Provide installation instructions using `pip` or `requirements.txt`.
   - **Model Architecture:**
     - Describe how the VGG16 model was used and the custom layers added.
     - Provide details on training (e.g., number of epochs, batch size, data augmentation).
   - **Training Instructions:** Explain how to run the script to train the model.
   - **Usage Instructions:**
     - Show how to load the model and make predictions on custom images.
   - **Results:**
     - Include validation accuracy, confusion matrix, and any insights gained during model evaluation.

#### 3. **Confusion Matrix & Training Curves:**
   - Include plots of the confusion matrix and training/validation accuracy and loss in the repository for easy reference.
   - Save these as image files, or plot them programmatically in the script.

#### 4. **License & Contribution Guidelines:**
   - Include an appropriate open-source license (like MIT).
   - Provide contribution guidelines if open to community participation.

#### Sample README.md Outline:
```markdown
# Driver Drowsiness Detection with VGG16

## Overview
This project aims to detect driver drowsiness using a deep learning model based on VGG16. The model is trained on a dataset of drowsy and non-drowsy images, and it predicts whether a driver is drowsy or alert from a given image.

## Dataset
- The dataset consists of images categorized into `Drowsy` and `Non Drowsy`.
- Data augmentation is applied during training to improve model robustness.

## Model Architecture
- The pre-trained VGG16 model is used as a feature extractor.
- Custom fully connected layers are added for binary classification.
- The model is trained using the Adam optimizer with a learning rate of `0.0001`.

## Results
- The model achieved an accuracy of `xx%` on the validation set.
- Confusion Matrix:
![Confusion Matrix](images/confusion_matrix.png)

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/your-repository-url
cd driver-drowsiness-detection
pip install -r requirements.txt
```

## Training
To train the model, run the following command:
```bash
python train_model.py
```

## Testing
To make predictions on a test image:
```bash
python predict.py --image path_to_image.jpg
```

## License
[MIT License](LICENSE)
```

This structure should help you organize the repository clearly for others to understand and use the project.
