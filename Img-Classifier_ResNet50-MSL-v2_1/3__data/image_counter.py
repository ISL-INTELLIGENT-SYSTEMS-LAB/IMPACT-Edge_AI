from PIL import Image
import os
import csv


def is_grayscale(image):
    if image.mode == 'L':
        return True
    elif image.mode == 'RGB':
        pixels = image.getdata()
        for pixel in pixels:
            r, g, b = pixel
            if r != g != b:
                return False
        return True
    else:
        return False


def write_results_to_csv(results):
    csv_file = "image_results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def scan_images(directory):
    results = []
    for root, dirs, files in os.walk(directory):
        if root == directory:
            continue
        grayscale_count = 0
        color_count = 0

        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                if is_grayscale(image):
                    grayscale_count += 1
                else:
                    color_count += 1

        total_count = grayscale_count + color_count
        if total_count == 0:
            grayscale_percentage = 0
            color_percentage = 0
        else:
            grayscale_percentage = (grayscale_count / total_count) * 100
            color_percentage = (color_count / total_count) * 100

        result = {
            "Folder": os.path.basename(root),
            "Total Images": total_count,
            "Grayscale Images": grayscale_count,
            "Grayscale Percentage": "{:.4f}".format(grayscale_percentage),
            "Color Images": color_count,
            "Color Percentage": "{:.4f}".format(color_percentage)
        }
        results.append(result)

    write_results_to_csv(results)


def main():
    # Specify the current directory
    current_directory = os.getcwd()

    # Call the function to scan images in each folder
    scan_images(current_directory)
    
    
if __name__ == "__main__":
    main()