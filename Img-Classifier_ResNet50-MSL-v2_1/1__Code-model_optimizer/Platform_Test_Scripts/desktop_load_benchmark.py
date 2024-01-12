import torch
from torchvision import transforms
from PIL import Image
import os
import psutil
import matplotlib.pyplot as plt
import pynvml
import time
from fvcore.nn import FlopCountAnalysis

MDL_PATH = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/2__TorchScript_Saved_Model/99percent-ResNet50-v2.1/mars_classifier_scripted.pt'

# Initialize NVML for GPU metrics
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  

# Load the TorchScript model
model = torch.jit.load(MDL_PATH)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing images
image_directory = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/images_benchmarking/'
image_files = os.listdir(image_directory)

# Metrics storage
cpu_usage = []
gpu_utilization = []
gpu_memory = []
io_time = []
prepoc_time = []
ram_usage = []

# Check if a GPU is available and if so, use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model.to(device)
total_time = 0

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
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization.append(gpu_util.gpu)
        gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory.append(gpu_mem_info.used)
        ram_usage.append(psutil.virtual_memory().percent)

        # Predict the image class
        with torch.no_grad():  # No need to track gradients for inference
            start_time = time.time()
            output = model(input_batch)
            end_time = time.time()
            total_time += end_time - start_time            
            
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    except Exception as e:
        print(f"An error occurred while processing the image '{image_name}': {e}")

# Print the average inference time, CPU usage, GPU utilization, GPU memory usage, IO Time, Preprocessing Time, and RAM usage
print(f"Average inference time: {total_time / len(image_files) * 1000} milliseconds")
print(f"Average CPU usage: {sum(cpu_usage) / len(cpu_usage)}%")
print(f"Average GPU utilization: {sum(gpu_utilization) / len(gpu_utilization)}%")
print(f"Average GPU memory usage: {sum(gpu_memory) / len(gpu_memory) / (1024**2)} MB")
print(f"Average IO Time: {sum(io_time) / len(io_time) * 1000} milliseconds")
print(f"Average Preprocessing Time: {sum(prepoc_time) / len(prepoc_time) * 1000} milliseconds")
print(f"Average RAM usage: {sum(ram_usage) / len(ram_usage)}%")
print(f"ResNet50 Model Flops: 4.11864 billion Flops")
print(f"ResNet50 Model Params: 25.557 million Parameters")

# Plotting CPU, GPU, and RAM metrics
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.plot(cpu_usage, label='CPU Usage (%)')
plt.xlabel('Image Index')
plt.ylabel('Percentage')
plt.title('CPU Usage')
plt.ylim(0, 100)  # Set the y-axis limit to 100
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(gpu_utilization, label='GPU Utilization (%)')
plt.xlabel('Image Index')
plt.ylabel('Percentage')
plt.title('GPU Utilization')
plt.ylim(0, 100)  # Set the y-axis limit to 100
plt.legend()

plt.subplot(1, 4, 3)
plt.plot([mem / (1024**2) for mem in gpu_memory], label='GPU Memory (MB)')
plt.xlabel('Image Index')
plt.ylabel('Memory Usage (MB)')
plt.title('GPU Memory Usage')
plt.ylim(0, 12288)  # Set the y-axis limit to 12288
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(ram_usage, label='RAM Usage (%)')
plt.xlabel('Image Index')
plt.ylabel('Percentage')
plt.title('RAM Usage')
plt.ylim(0, 100)  # Set the y-axis limit to 100
plt.legend()

plt.tight_layout()
plt.show()

# Clean up NVML
pynvml.nvmlShutdown()
