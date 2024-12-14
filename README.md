Image Enhancement Using LLNet

Overview

This project implements an image enhancement pipeline using a neural network architecture inspired by the Low-Light Network (LLNet). The system processes low-light images, enhances their quality, and visualizes the improvements.

Features

Automatic Extraction: Handles zip files containing images and extracts them for further processing.

Preprocessing: Reads, resizes, and normalizes images for input into the neural network.

Neural Network: Implements an LLNet-inspired architecture for image enhancement using TensorFlow/Keras.

Visualization: Displays original and enhanced images for comparison.

File Structure

Img (Input Zip File): Contains folders with images to be processed.

Images/New folder: Directory where images are extracted and stored for preprocessing.

lo.zip (Dataset): Contains low-light images for enhancement.

lo (Input Folder): Folder with input low-light images.

br (Output Folder): Folder with enhanced output images.

Dependencies

Python 3.8+

Libraries:

opencv-python

numpy

tensorflow

Pillow

scikit-learn

matplotlib

Install dependencies using:

pip install opencv-python numpy tensorflow Pillow scikit-learn matplotlib

Steps

1. Extract Images

Extract images from a zip file for preprocessing:

import zipfile
import os

zip_file_path = "/content/Img"
output_directory = "/content/Images"

os.makedirs(output_directory, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_directory)

print("Files extracted to:", output_directory)

2. Preprocess Images

Resize and normalize images:

from glob import glob
import cv2
import numpy as np

def preprocess_images(data_dir, image_size=(128, 128)):
    images = []
    folders = os.listdir(data_dir)

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        image_paths = glob(f'{folder_path}/*')

        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, image_size) / 255.0
                images.append(img)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return np.array(images)

3. Train-Test Split

Split the dataset into training and testing sets:

from sklearn.model_selection import train_test_split

all_images = preprocess_images('/content/Images/New folder')
train_images, test_images = train_test_split(all_images, test_size=0.2, random_state=42)

4. Build LLNet Model

Define and compile the LLNet-inspired architecture:

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model

def build_llnet(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs)
    return model

llnet_model = build_llnet()
llnet_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

5. Visualize Results

Compare original and enhanced images:

from PIL import Image
import matplotlib.pyplot as plt

input_folder = '/content/lo'
output_folder = '/content/br'

input_images = sorted(os.listdir(input_folder))
output_images = sorted(os.listdir(output_folder))

num_images = len(input_images)
fig, axes = plt.subplots(2, num_images, figsize=(15, 4))

for i in range(num_images):
    original = Image.open(os.path.join(input_folder, input_images[i]))
    enhanced = Image.open(os.path.join(output_folder, output_images[i]))

    axes[0, i].imshow(original)
    axes[0, i].set_title(f'Original')
    axes[0, i].axis('off')

    axes[1, i].imshow(enhanced)
    axes[1, i].set_title(f'Enhanced')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()

Results

Training Dataset: Preprocessed low-light images.

Testing Dataset: Separate test set for evaluation.

Visualization: Side-by-side comparison of original and enhanced images.

Future Work

Improve model performance by fine-tuning hyperparameters.

Extend the pipeline to handle video frames.

Experiment with additional architectures and datasets.

License

This project is licensed under the MIT License.
