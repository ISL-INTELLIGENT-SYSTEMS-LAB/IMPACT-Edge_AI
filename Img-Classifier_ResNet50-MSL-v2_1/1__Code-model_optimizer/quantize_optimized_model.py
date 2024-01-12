import torch.quantization
from torch.quantization import get_default_qconfig

MDL_PATH = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/2__TorchScript_Saved_Model/99percent-ResNet50-v2.1/mars_classifier_scripted.pt'

# Assuming MDL_PATH is already set to the path of your model
model_fp32 = torch.jit.load(MDL_PATH)
model_fp32.eval()

# Move the model to CPU
model_fp32.cpu()

# Set the qconfig for the model to use 'qnnpack'
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

# Apply the qconfig to the model using `prepare`
model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace=False)

# Convert the prepared model to a quantized state
model_fp32_quantized = torch.quantization.convert(model_fp32_prepared, inplace=False)