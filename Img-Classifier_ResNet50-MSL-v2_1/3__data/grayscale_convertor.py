import argparse
import os
from PIL import Image

def convert_images(input_dir, output_dir):
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Converting images to grayscale...")
    count = 0  # Initialize count variable
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            image = Image.open(image_path)
            image = image.convert('L')
            image.save(output_path)
            count += 1  # Increment count for each converted image
            
    print(f"Total images converted: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image conversion script')
    parser.add_argument('input_dir', type=str, help='input directory containing images')
    parser.add_argument('output_dir', type=str, help='output directory for converted images')
    args = parser.parse_args()

    convert_images(args.input_dir, args.output_dir)
