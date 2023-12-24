import os
import shutil

def copy_images(source_dir, source_file, destination_dir):
    with open(source_file, 'r') as file:
        for line in file:
            line = line.strip()
            filename, number = line.split(' ')
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, filename)
            shutil.copy(source_path, destination_path)

# Specify the source directory where the images are currently located
source_dir = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/images'

# Specify the file with the image filenames
source_file = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/test-set-v2.1.txt'

# Specify the destination directory where you want to copy the images
destination_dir = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/images_benchmarking'

copy_images(source_dir, source_file, destination_dir)