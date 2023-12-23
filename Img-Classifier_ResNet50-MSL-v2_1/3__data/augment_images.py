##############################################################################################################################################################################
"""
Image Augmentation Script for Deep Learning Dataset Preparation

Description:
This script is tailored for augmenting a specific class of images within a deep learning dataset. It selectively processes images based on their 
class label, which is indicated at the end of each image's filename. The script applies a series of transformations to create augmented variations, 
thereby enhancing the diversity of the dataset and potentially improving the generalization capability of a deep learning model trained on this data.

The transformations applied include:
- Random horizontal and vertical flips to mimic alternative orientations.
- Random rotations to represent tilted angles.
- Color jittering to simulate variations in lighting and camera quality.
- Random affine transformations to introduce changes in perspective.
- Random cropping to the target size to represent different scales and focal points.
- Occasional conversion to grayscale to account for the absence of color.

The script processes images with a fixed size of 227x227 pixels, a standard dimension for many convolutional neural networks.

Configuration:
Before running the script, configure the following variables:
- `IMAGE_DIR`: Path to the directory containing the original images.
- `TRAIN_SET_FILE`: Filename of the text file listing all images with their labels.
- `SPECIFIED_LABEL`: The label identifier for the class of images to augment.
- `NUM_AUGMENTED_IMAGES`: The number of augmented images to generate per original image.

Behavior:
The script reads through the list of image filenames from `TRAIN_SET_FILE`, filtering for those ending with the `SPECIFIED_LABEL`. For each filtered image, 
it applies the augmentation transformations and saves the new images in the same directory as the originals. Augmented images are named using a prefix 'aug_' 
followed by an index and the original filename.

Naming Convention:
The script assumes that image filenames conclude with an underscore followed by their class label, e.g., '1996MH0006030020800245C00_DRCL.jpg 4', 
where '4' is the class label. The script extracts this label from each line in the training set file to match against `SPECIFIED_LABEL`.

Usage:
Directly run the script after setting the necessary configuration variables. No command-line arguments are required.

Requirements:
- Python 3.x
- Pillow library (PIL fork)
- torchvision library

Ensure that all dependencies are installed and that the Python environment is set up correctly before executing this script.

Oganization: Intelligent Systems Lab, Fayetteville State University (https://www.uncfsu.edu/intelligent-systems-lab)
PI: Dr. Sambit Bhattacharya
Author: Matthew Wilkerson [adapted from a script by Taylor Brown]
Date: 2023-12-22
"""
##############################################################################################################################################################################


import os
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIR = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/images'
TRAIN_SET_FILE = 'train-set-v2.1.txt'
SPECIFIED_LABEL = '12'  # Update this label as needed
NUM_AUGMENTED_IMAGES = 4  # Update this number as needed

# Define the augmentation pipeline with additional transformations
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop(227),
    transforms.RandomGrayscale(p=0.1)
])

def load_training_images(train_set_file, specified_label):
    labeled_images = []
    with open(train_set_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 2 and parts[1] == specified_label:
                labeled_images.append(parts[0])
    return labeled_images

def augment_image(image_path, save_dir, num_augmented_images, label):
    image = Image.open(image_path)
    augmented_image_paths = []
    for i in range(num_augmented_images):
        augmented_image = augmentations(image)
        augmented_filename = f'AUG_{i}_{os.path.basename(image_path)}'
        augmented_image_path = os.path.join(save_dir, augmented_filename)
        augmented_image.save(augmented_image_path)
        # Append only the filename and label to the list, not the full path
        augmented_image_paths.append(augmented_filename + ' ' + label)
    return augmented_image_paths

def append_to_train_set_file(train_set_file, augmented_image_paths):
    with open(train_set_file, 'a') as file:
        for path in augmented_image_paths:
            file.write(path + '\n')

def main():

    print(f"The current working directory is: {SCRIPT_DIR}")

    # Full path to the file containing the dataset information
    train_set_path = os.path.join(SCRIPT_DIR, TRAIN_SET_FILE)

    # Check if the file exists at the full path
    if not os.path.isfile(train_set_path):
        raise FileNotFoundError(f"The file {TRAIN_SET_FILE} was not found in the directory {SCRIPT_DIR}. Please check the file path.")

    # Load training images with the specified label from file using the full path
    labeled_images = load_training_images(train_set_path, SPECIFIED_LABEL)
    
    # Perform augmentation on the filtered images and update the training set file
    for image_name in labeled_images:
        image_path = os.path.join(IMAGE_DIR, image_name)
        augmented_image_paths = augment_image(image_path, IMAGE_DIR, NUM_AUGMENTED_IMAGES, SPECIFIED_LABEL)
        append_to_train_set_file(train_set_path, augmented_image_paths)  # Here too, use train_set_path

if __name__ == '__main__':
    main()



