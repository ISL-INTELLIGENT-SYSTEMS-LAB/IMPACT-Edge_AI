import onnx

# Load an ONNX model
onnx_model = onnx.load("/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/1__Code-model_optimizer/ResNet50-MSL-v2_1.onnx")

# Check the model
onnx.checker.check_model(onnx_model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))
