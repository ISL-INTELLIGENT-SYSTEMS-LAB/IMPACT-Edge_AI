import torch
from torchvision import transforms
from PIL import Image
import os
import csv
MDL_PATH = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/2__TorchScript_Saved_Model/99percent-ResNet50-v2.1/mars_classifier_scripted.pt'

# Load the TorchScript model
model = torch.jit.load(MDL_PATH)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing images
image_directory = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/test_dataset/'
image_files = os.listdir(image_directory)

# Check if a GPU is available and if so, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create a CSV file to save the predictions
csv_file = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/test_dataset/predictions.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Prediction'])

    # Loop through images in the directory and make predictions
    for image_name in image_files:
        image_path = os.path.join(image_directory, image_name)
        try:
            # Load an image
            image = Image.open(image_path)
            
            # Preprocess the image and add batch dimension
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)  # create a mini-batch as expected by the model

            # Predict the image class
            with torch.no_grad():  # No need to track gradients for inference
                output = model(input_batch)
         
            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            # Write the prediction to the CSV file
            predicted_class = torch.argmax(probabilities).item()
            writer.writerow([image_name, predicted_class])

        except Exception as e:
            print(f"An error occurred while processing the image '{image_name}': {e}")

