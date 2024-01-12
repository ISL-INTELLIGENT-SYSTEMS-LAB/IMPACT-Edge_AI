import torch
from torchvision import transforms
from PIL import Image
import os
import psutil
import matplotlib.pyplot as plt
import time
from jtop import jtop

MDL_PATH = '/home/jetson/impactAI/ResNet50MSL/benchmark/mars_classifier_scripted.pt'

# Load the TorchScript model
model = torch.jit.load(MDL_PATH)
model.eval()  # Set the model to evaluation mode
print("ResNet50-MSL-v2.1 loaded successfully")

# Define the image transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing images
image_directory = '/home/jetson/impactAI/ResNet50MSL/benchmark/images_benchmarking/'
image_files = os.listdir(image_directory)

# Metrics storage
cpu_usage = []
gpu_utilization = []
gpu_memory = []
io_time = []
prepoc_time = []

# Check if a GPU is available and if so, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(torch.cuda.is_available())
total_time = 0

# Using jtop context manager to automatically open and close the connection to jtop
with jtop() as jetson:
    # Loop through images in the directory and make predictions
    for image_name in image_files:
        image_path = os.path.join(image_directory, image_name)
        try:
            # Measure IO Time
            io_start_time = time.time()
            
            # Load an image
            image = Image.open(image_path)
            
            io_end_time = time.time()
            io_duration = io_end_time - io_start_time
            io_time.append(io_duration)

            # Preprocess the image and add batch dimension
            preproc_start_time = time.time()
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)  # create a mini-batch as expected by the model
            preproc_end_time = time.time()
            prepoc_time.append(preproc_end_time - preproc_start_time)

            # Record CPU and GPU metrics before prediction
            cpu_percent = psutil.cpu_percent()
            cpu_usage.append(cpu_percent)

            # Get GPU utilization and memory usage from jetson-stats (jtop)
           # gpu_utilization.append(jetson.gpu['GPU'])  # GPU utilization in percentage
           # gpu_memory.append(jetson.gpu['RAM']['used'])       # Used GPU memory in MB

            # Predict the image class
            with torch.no_grad():  # No need to track gradients for inference
                start_time = time.time()
                output = model(input_batch)
                end_time = time.time()
                total_time += end_time - start_time            
                
            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            print(f"{image_name} done.")

        except Exception as e:
            print(f"An error occurred while processing the image '{image_name}': {e}")

# Print the average inference time, CPU usage, GPU utilization, and IO Time
print(f"Average inference time: {total_time / len(image_files) * 1000} milliseconds")
print(f"Average CPU usage: {sum(cpu_usage) / len(cpu_usage)}%")
#print(f"Average GPU utilization: {sum(gpu_utilization) / len(gpu_utilization)}%")
#print(f"Average GPU memory usage: {sum(gpu_memory) / len(gpu_memory) / (1024**2)} MB")
print(f"Average IO Time: {sum(io_time) / len(io_time) * 1000} milliseconds")
print(f"Average Preprocessing Time: {sum(prepoc_time) / len(prepoc_time) * 1000} milliseconds")

# Plotting CPU and GPU metrics
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(cpu_usage, label='CPU Usage (%)')
plt.xlabel('Image Index')
plt.ylabel('Percentage')
plt.title('CPU Usage')
plt.ylim(0, 100)  # Set the y-axis limit to 100
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(gpu_utilization, label='GPU Utilization (%)')
plt.xlabel('Image Index')
plt.ylabel('Percentage')
plt.title('GPU Utilization')
plt.ylim(0, 100)  # Set the y-axis limit to 100
plt.legend()

plt.subplot(1, 3, 3)
plt.plot([mem / (1024**2) for mem in gpu_memory], label='GPU Memory (MB)')
plt.xlabel('Image Index')
plt.ylabel('Memory Usage (MB)')
plt.title('GPU Memory Usage')
plt.ylim(0, 4000)  # Adjust y-axis limit to total GPU memory available
plt.legend()

plt.tight_layout()
plt.show()