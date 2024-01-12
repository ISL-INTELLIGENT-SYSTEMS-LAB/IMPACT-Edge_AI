import os
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.quantization.resnet import ResNet50_QuantizedWeights

torch.backends.quantized.engine = 'fbgemm'

model_quantized = models.quantization.resnet50(weights=ResNet50_QuantizedWeights.DEFAULT, quantize=True)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

def print_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_model_size(model)
print_model_size(model_quantized)

"""
Output
    Size (MB): 102.523238
    Size (MB): 26.151336
"""