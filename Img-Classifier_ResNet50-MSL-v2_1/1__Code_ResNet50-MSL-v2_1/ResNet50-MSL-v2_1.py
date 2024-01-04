##############################################################################################################################################################################
"""
██████╗ ███████╗███████╗███╗   ██╗███████╗████████╗███████╗ ██████╗       ███╗   ███╗███████╗██╗      ██╗   ██╗██████╗    ██╗
██╔══██╗██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝██╔════╝██╔═████╗      ████╗ ████║██╔════╝██║      ██║   ██║╚════██╗  ███║
██████╔╝█████╗  ███████╗██╔██╗ ██║█████╗     ██║   ███████╗██║██╔██║█████╗██╔████╔██║███████╗██║█████╗██║   ██║ █████╔╝  ╚██║
██╔══██╗██╔══╝  ╚════██║██║╚██╗██║██╔══╝     ██║   ╚════██║████╔╝██║╚════╝██║╚██╔╝██║╚════██║██║╚════╝╚██╗ ██╔╝██╔═══╝    ██║
██║  ██║███████╗███████║██║ ╚████║███████╗   ██║   ███████║╚██████╔╝      ██║ ╚═╝ ██║███████║███████╗  ╚████╔╝ ███████╗██╗██║
╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝       ╚═╝     ╚═╝╚══════╝╚══════╝   ╚═══╝  ╚══════╝╚═╝╚═╝
This script defines a custom dataset class for loading and processing Mars images, as well as functions for creating data loaders, setting up the model 
and optimizer, and training the model. The MarsDataset class loads the image labels from a file and checks if the corresponding image files exist. 
It provides methods for retrieving the length of the dataset and getting an image and its label at a given index. The get_train_transforms function 
returns a composition of image transformations for training data, including color jitter, random flipping, rotation, resizing, cropping, grayscale 
conversion, tensor conversion, normalization, and random erasing. The get_val_test_transforms function returns a transformation pipeline for validation 
and testing data, including resizing, center cropping, tensor conversion, and normalization. The create_datasets function creates datasets for training, 
validation, and testing, using the MarsDataset class and the specified transformations. The create_data_loaders function creates data loaders for the 
training, validation, and testing datasets, with specified batch size, shuffle, number of workers, and pin memory. The setup_model_optimizer function 
sets up the ResNet50 model with a modified fully connected layer, the optimizer, learning rate scheduler, and loss function for training. The train_model 
function trains the model using the provided data loaders, criterion, optimizer, scheduler, device, number of epochs, patience for early stopping, 
and save frequency for model checkpoints.

Note: The code assumes that the necessary libraries (torch, torchvision, numpy, PIL, seaborn, matplotlib) are installed.
"""
##############################################################################################################################################################################
# Standard library imports
import csv
import os
# Third-party imports for array handling and machine learning
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tqdm
# PyTorch and torchvision for deep learning
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import GaussianBlur

# Set global variables
ROOT_DIR = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/IMG-Classifier_ResNet50-MSL-v2_1/'
DATA_DIR = 'data'
LABELS_FILE = 'train-set-v2.1.txt'
IMG_DIR = 'images'
CLASS_NAMES = ['arm cover', 'other rover part', 'artifact', 'nearby surface', 'close-up rock', 
               'DRT','DRT spot', 'distant landscape', 'drill hole', 'night sky', 'light-toned veins', 
               'mastcam cal target','sand', 'sun', 'wheel', 'wheel joint', 'wheel tracks']

class MarsDataset(Dataset):
    """
    A custom dataset class for loading and processing Mars images.

    Args:
        root_dir (str): Root directory of the dataset.
        data_dir (str): Directory containing the data files.
        labels_file (str): File name of the labels file.
        img_dir (str): Directory containing the image files.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        root_dir (str): Root directory of the dataset.
        data_dir (str): Directory containing the data files.
        img_dir (str): Directory containing the image files.
        transform (callable, optional): Optional transform to be applied on a sample.
        labels (list): List of tuples containing image names and labels.

    Methods:
        _load_labels(labels_file): Loads the labels from the labels file.
        _check_images_exist(): Checks if all image files exist.
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the image and label at the given index.
    """

    def __init__(self, root_dir, data_dir, labels_file, img_dir, transform=None):
        """
        Initialize the MarsDataset class.

        Args:
            root_dir (str): Root directory of the dataset.
            data_dir (str): Directory containing the dataset.
            labels_file (str): Path to the file containing the labels.
            img_dir (str): Directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self._load_labels(labels_file)
        self._check_images_exist()

    def _load_labels(self, labels_file):
        """
        Load labels from a file.

        Args:
            labels_file (str): The name of the file containing the labels.

        Returns:
            list: A list of tuples, where each tuple represents a label.
            Each tuple contains two elements: the label name and its corresponding value.
        """
        labels_path = os.path.join(self.root_dir, self.data_dir, labels_file)
        with open(labels_path, 'r') as file:
            labels = [tuple(line.strip().split()) for line in file]
        return labels

    def _check_images_exist(self):
        """
        Check if all the image files exist in the specified directory.

        Raises:
            AssertionError: If any image file is not found.
        """
        for img_name, _ in self.labels:
            img_path = os.path.join(self.root_dir, self.data_dir, self.img_dir, img_name)
            assert os.path.isfile(img_path), f"File not found: {img_path}"

    def __len__(self):
        """
        Returns the length of the dataset.
            
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding label at the given index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        img_name, image_label = self.labels[idx]
        image_path = os.path.join(self.root_dir, self.data_dir, self.img_dir, img_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, int(image_label)

def get_train_transforms():
    """
    Returns a composition of image transformations for training data.
    
    The transformations include:
    - ColorJitter: Adjusts brightness, contrast, saturation, and hue of the image.
    - RandomHorizontalFlip: Randomly flips the image horizontally.
    - RandomRotation: Randomly rotates the image by a certain degree.
    - RandomResizedCrop: Randomly crops and resizes the image.
    - CenterCrop: Crops the image from the center.
    - RandomGrayscale: Randomly converts the image to grayscale.
    - ToTensor: Converts the image to a tensor.
    - Normalize: Normalizes the image by subtracting mean and dividing by standard deviation.
    - RandomErasing: Randomly erases parts of the image.
    
    Returns:
    - A composition of image transformations.
    """
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.CenterCrop(size=224),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

def get_val_test_transforms():
    """
    Returns the transformation pipeline for validation and testing data.
    
    This function applies a series of transformations to the input data, including resizing,
    center cropping, converting to tensor, and normalizing the pixel values.
    
    Returns:
        transforms.Compose: A composition of transformations to be applied to the data.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def create_datasets(root_dir, data_dir, labels_file, img_dir, train_transform, val_test_transform):
    """
    Create datasets for training, validation, and testing.

    Args:
        root_dir (str): Root directory of the dataset.
        data_dir (str): Directory containing the dataset.
        labels_file (str): File containing the labels for the dataset.
        img_dir (str): Directory containing the images.
        train_transform (torchvision.transforms.Compose): Transformations to apply to the training dataset.
        val_test_transform (torchvision.transforms.Compose): Transformations to apply to the validation and testing datasets.

    Returns:
        tuple: A tuple containing the training, validation, and testing datasets.
    """
    train_dataset = MarsDataset(root_dir, data_dir, labels_file, img_dir, transform=train_transform)
    val_dataset = MarsDataset(root_dir, data_dir, labels_file, img_dir, transform=val_test_transform)
    test_dataset = MarsDataset(root_dir, data_dir, labels_file, img_dir, transform=val_test_transform)
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset):
    """
    Create data loaders for training, validation, and testing datasets.
    
    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The testing dataset.
    
    Returns:
        train_loader (DataLoader): The data loader for the training dataset.
        val_loader (DataLoader): The data loader for the validation dataset.
        test_loader (DataLoader): The data loader for the testing dataset.
    """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    return train_loader, val_loader, test_loader

def setup_model_optimizer():
    """
    Set up the model, optimizer, scheduler, and criterion for training.

    Returns:
        model (torch.nn.Module): The ResNet50 model with modified fully connected layer.
        optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        criterion (torch.nn.Module): The loss function for training.
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    num_classes = 19
    model.fc = torch.nn.Linear(num_features, num_classes)
    fc_params = list(model.fc.parameters())
    base_params = [param for param in model.parameters() if param.requires_grad and not any(id(param) == id(p) for p in fc_params)]
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},  # Learning rate for pre-trained layers
        {'params': fc_params, 'lr': 1e-3}     # Learning rate for the fully connected layer
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, scheduler, criterion

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience, save_frequency):
    """
    Trains a model using the provided data loaders, criterion, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        device (torch.device): The device to be used for training.
        epochs (int): The number of epochs to train the model.
        patience (int): The number of epochs to wait for improvement in validation loss before early stopping.
        save_frequency (int): The frequency (in epochs) at which to save model checkpoints.

    Returns:
        tuple: A tuple containing the training losses, validation losses, training accuracies, and validation accuracies.
    """    
    def forward_pass(model, inputs, labels, criterion, device):
        """
        Performs a forward pass through the model and calculates the loss and predicted labels.

        Args:
            model (torch.nn.Module): The neural network model.
            inputs (torch.Tensor): The input data.
            labels (torch.Tensor): The target labels.
            criterion (torch.nn.Module): The loss function.
            device (torch.device): The device to perform the computation on.

        Returns:
            tuple: A tuple containing the outputs, loss, and predicted labels.
        """
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        return outputs, loss, predicted
    
    def backward_pass(optimizer, loss):
        """
        Performs the backward pass of the optimization algorithm.

        Args:
            optimizer: The optimizer used for updating the model parameters.
            loss: The loss value computed during the forward pass.

        Returns:
            None
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def compute_accuracy(predicted, labels, device):
        """
        Computes the accuracy of the predicted labels compared to the ground truth labels.

        Args:
            predicted (Tensor): The predicted labels.
            labels (Tensor): The ground truth labels.
            device (str): The device on which the computation is performed.

        Returns:
            tuple: A tuple containing the number of correct predictions and the total number of predictions.
        """
        labels = labels.to(device)
        predicted = predicted.to(device)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct, total

    def train_one_epoch(model, train_loader, criterion, optimizer, device):
        """
        Trains the model for one epoch using the provided data loader.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            criterion: The loss function.
            optimizer: The optimizer for updating model parameters.
            device: The device to be used for training.

        Returns:
            tuple: A tuple containing the average training loss and training accuracy for the epoch.
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm.tqdm(train_loader, leave=False):
            outputs, loss, predicted = forward_pass(model, inputs, labels, criterion, device)
            if torch.isnan(loss):
                print("Loss is NaN. Stopping training.")
                return None, None
            backward_pass(optimizer, loss)
            running_loss += loss.item()
            c, t = compute_accuracy(predicted, labels, device)
            correct += c
            total += t
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        return train_loss, train_accuracy

    def validate_one_epoch(model, val_loader, criterion, device):
        """
        Validate the model on the validation dataset for one epoch.

        Args:
            model (torch.nn.Module): The model to be validated.
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            criterion: The loss function.
            device (torch.device): The device to perform the validation on.

        Returns:
            tuple: A tuple containing the validation loss and accuracy.
        """
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs, loss, predicted = forward_pass(model, inputs, labels, criterion, device)
                val_loss += loss.item()
                c, t = compute_accuracy(predicted, labels)
                correct += c
                total += t
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy

    def print_epoch_stats(epoch, epochs, train_loss, val_loss, train_accuracy, val_accuracy):
        """
        Prints the statistics for each epoch during training.

        Parameters:
            epoch (int): The current epoch number.
            epochs (int): The total number of epochs.
            train_loss (float): The training loss for the current epoch.
            val_loss (float): The validation loss for the current epoch.
            train_accuracy (float): The training accuracy for the current epoch.
            val_accuracy (float): The validation accuracy for the current epoch.
        """
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    def save_checkpoint(epoch, model, optimizer, filepath):
        """"
        Save the checkpoint of the model and optimizer.

        Args:
            epoch (int): The current epoch number.
            model (torch.nn.Module): The model to be saved.
            optimizer (torch.optim.Optimizer): The optimizer to be saved.
            filepath (str): The path where the checkpoint will be saved.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
         }, filepath)

    best_val_loss = float('inf')
    trigger_times = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f'Training Epoch {epoch+1}/{epochs}')
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if train_loss is None:  # NaN Loss Check
            break
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            save_checkpoint(epoch, model, optimizer, 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        scheduler.step()
        print_epoch_stats(epoch, epochs, train_loss, val_loss, train_accuracy, val_accuracy)
        if epoch % save_frequency == 0:
            save_checkpoint(epoch, model, optimizer, f'checkpoint_{epoch}.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluates the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device on which the evaluation will be performed.

    Returns:
        tuple: A tuple containing the confusion matrix and the normalized confusion matrix.
    """
    
    def forward_pass(inputs, labels, model, criterion, device):
        """
        Performs a forward pass through the model and calculates the loss.

        Args:
            inputs (torch.Tensor): The input data.
            labels (torch.Tensor): The ground truth labels.
            model (torch.nn.Module): The model to be evaluated.
            criterion (torch.nn.Module): The loss function used for evaluation.
            device (torch.device): The device on which the evaluation will be performed.

        Returns:
            tuple: A tuple containing the model outputs and the loss value.
        """
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return outputs, loss.item()

    def update_metrics(outputs, labels, loss, correct, total, all_predicted, all_labels):
        """
        Updates the evaluation metrics.

        Args:
            outputs (torch.Tensor): The model outputs.
            labels (torch.Tensor): The ground truth labels.
            loss (float): The current loss value.
            correct (int): The number of correct predictions.
            total (int): The total number of predictions.
            all_predicted (list): A list to store all predicted labels.
            all_labels (list): A list to store all ground truth labels.

        Returns:
            tuple: A tuple containing the updated loss, correct predictions, and total predictions.
        """
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        return loss, correct, total

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, batch_loss = forward_pass(inputs, labels, model, criterion, device)
            test_loss += batch_loss
            test_loss, correct, total = update_metrics(outputs, labels, test_loss, correct, total, all_predicted, all_labels)

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    cm = confusion_matrix(all_labels, all_predicted)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    return cm, cm_normalized

def plot_loss(train_losses, val_losses):
    """
    Plots the training and validation loss over epochs.
    
    Parameters:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies):
    """
    Plots the training and validation accuracies over epochs.
    
    Parameters:
        train_accuracies (list): List of training accuracies for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
    """
    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix (Unnormalized)', cmap=plt.cm.Blues):
    """
    This function prints and plots the unnormalized confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.yticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.show()

def plot_normalized_confusion_matrix(cm, classes, title='Confusion Matrix (Normalized)', cmap=plt.cm.Blues):
    """
    This function prints and plots the normalized confusion matrix.
    """
    # Normalize by row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.yticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.show()

def main():
    # Set up the data loaders   
    train_transform = get_train_transforms()
    val_test_transform = get_val_test_transforms()
    train_dataset, val_dataset, test_dataset = create_datasets(ROOT_DIR, DATA_DIR, LABELS_FILE, IMG_DIR, train_transform, val_test_transform)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)
    # Set up the model, optimizer, scheduler, and criterion
    model, optimizer, scheduler, criterion = setup_model_optimizer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
    model.to(device) # Move model to GPU if available
    epochs = 25  # Number of epochs to train the model
    patience = 5  # Number of epochs to wait for improvement in validation loss before early stopping
    save_frequency = 5# Frequency (in epochs) at which to save model checkpoints
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        patience=patience,
        save_frequency=save_frequency
    )

    plot_loss(train_losses, val_losses) # Plot the training and validation losses
    plot_accuracy(train_accuracies, val_accuracies) # Plot the training and validation accuracies
    # Evaluate the model on the test set
    conf_matrix, conf_matrix_normalized = evaluate_model(model, test_loader, criterion, device)
    # Plot the confusion matrices
    plot_confusion_matrix(conf_matrix, CLASS_NAMES)
    plot_normalized_confusion_matrix(conf_matrix_normalized, CLASS_NAMES)
    # Save the model
    scripted_model = torch.jit.script(model)
    scripted_model.save('mars_classifier_scripted.pt')

if __name__ == '__main__':
    main()