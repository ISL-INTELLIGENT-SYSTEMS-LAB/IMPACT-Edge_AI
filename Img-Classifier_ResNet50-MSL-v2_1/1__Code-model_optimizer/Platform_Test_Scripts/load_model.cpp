#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <torchvision/transforms/transforms.h>

namespace fs = std::filesystem;
namespace transforms = torchvision::transforms;

int main() {
    // Set the path to the directory containing the images
    std::string imageDir = "/path/to/images";

    // Set the path to the TorchScript model
    std::string modelPath = "/path/to/model.pt";

    // Load the TorchScript model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(modelPath);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return 1;
    }

    // Define the image preprocessing transform
    transforms::Compose transform = transforms::Compose({
        transforms::ToTensor(),
        transforms::Normalize({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
    });

    // Iterate over the images in the directory
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        // Load the image using OpenCV
        cv::Mat image = cv::imread(entry.path().string());

        // Preprocess the image
        torch::Tensor inputTensor = transform(image);

        // Run the model and make predictions
        torch::Tensor outputTensor = model.forward({inputTensor}).toTensor();

        // Process the output (e.g., get predicted labels, probabilities, etc.)

        // Print the predictions
        std::cout << "Predictions for " << entry.path().filename() << ":" << std::endl;
        std::cout << outputTensor << std::endl;
    }

    return 0;
}
