'''
This script implements a k-fold cross-validation for training and evaluating a image classifier model loaded from Torchvision. The script defines a custom 
dataset class, transforms for data augmentation, and functions for setting up the model, optimizer, and scheduler. It also includes functions for training 
the model, performing k-fold cross-validation, and evaluating the model on a test set. The script uses PyTorch and torchvision for deep learning, and sklearn 
for k-fold cross-validation and evaluation metrics. The main function of the script is the `Main()` function, which orchestrates the training and evaluation 
process.

Organization: Intelligent Systems Laboratory at Fayetteville State University
PI: Dr. Sambit Bhattacharya
Author: Matthew Wilkerson

Update the following:
    - ROOT_DIR
    - DATA_DIR
    - LABELS_FILE
    - IMG_DIR
    - CLASS_NAMES
    - get_train_transforms()
    - get_val_test_transforms()
    - setup_model_optimizer()
        model
        classes
        optimizer
        weight_decay
        scheduler
        criterion (loss function)
    - kfold_cross_validation()
        kfold
        epochs
        patience
        save_frequency
    - Main()
        train_model() (epochs=, patience=, save_frequency=)
        TorchScript name
'''

# Standard library imports
import csv
import os
# Third-party imports for array handling and machine learning
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import tqdm
# PyTorch and torchvision for deep learning
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import GaussianBlur
import numpy as np

# Set global variables
ROOT_DIR = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/IMG-Classifier_ResNet50-MSL-v2_1/'
DATA_DIR = '/home/mwilkers1/Documents/Projects/IMPACT/Edge-AI/IMPACT-Edge_AI/Img-Classifier_ResNet50-MSL-v2_1/3__data/'
LABELS_FILE = 'train-set-v2.1.txt'
IMG_DIR = 'images'
CLASS_NAMES = ['arm cover', 'other rover part', 'artifact', 'nearby surface', 'close-up rock', 
               'DRT','DRT spot', 'distant landscape', 'drill hole', 'night sky', 'light-toned veins', 
               'mastcam cal target','sand', 'sun', 'wheel', 'wheel joint', 'wheel tracks']

class MarsDataset(Dataset): # Inherits from the PyTorch Dataset class
    def __init__(self, root_dir, data_dir, labels_file, img_dir, transform=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self._load_labels(labels_file) # Load the labels from the labels file
        self._check_images_exist() # Check that all images actually exist

    def _load_labels(self, labels_file): # Load the labels from the labels file
        labels_path = os.path.join(self.root_dir, self.data_dir, labels_file)
        with open(labels_path, 'r') as file:
            labels = [tuple(line.strip().split()) for line in file]
        return labels

    def _check_images_exist(self): # Check that all images actually exist
        for img_name, _ in self.labels:
            img_path = os.path.join(self.root_dir, self.data_dir, self.img_dir, img_name)
            assert os.path.isfile(img_path), f"File not found: {img_path}"

    def __len__(self):  # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx): # Return the image and its label
        img_name, image_label = self.labels[idx]
        image_path = os.path.join(self.root_dir, self.data_dir, self.img_dir, img_name)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, int(image_label)
    
def get_train_transforms(): # Define the transformations to be applied to the training images
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

def get_val_test_transforms(): # Define the transformations to be applied to the validation and test images
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # For RGB images
    ])

def create_datasets(root_dir, data_dir, labels_file, img_dir, train_transform, val_test_transform): # Create the datasets
    train_dataset = MarsDataset(root_dir, data_dir, labels_file, img_dir, transform=train_transform)
    val_dataset = MarsDataset(root_dir, data_dir, labels_file, img_dir, transform=val_test_transform)
    test_dataset = MarsDataset(root_dir, data_dir, labels_file, img_dir, transform=val_test_transform)
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset): # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    return train_loader, val_loader, test_loader

def setup_model_optimizer(): # Set up the model, optimizer, scheduler, and criterion
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=1, patience=5, save_frequency=5):
    
    def forward_pass(model, inputs, labels, criterion, device): 
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        return outputs, loss, predicted
    
    def backward_pass(optimizer, loss): 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def compute_accuracy(predicted, labels, device):
        labels = labels.to(device)
        predicted = predicted.to(device)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct, total

    def train_one_epoch(model, train_loader, criterion, optimizer, device): 
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
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs, loss, predicted = forward_pass(model, inputs, labels, criterion, device)
                val_loss += loss.item()
                c, t = compute_accuracy(predicted, labels, device)  # Pass the 'device' argument to compute_accuracy
                correct += c
                total += t
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        return val_loss, val_accuracy
        
    def print_epoch_stats(epoch, epochs, train_loss, val_loss, train_accuracy, val_accuracy):
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    def save_checkpoint(epoch, model, optimizer, filepath):
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

    def forward_pass(inputs, labels, model, criterion, device):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return outputs, loss.item()

    def update_metrics(outputs, labels, loss, correct, total, all_predicted, all_labels):
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.to(device)).sum().item()
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

def kfold_cross_validation(train_dataset, val_dataset, k=5):
    # Concatenate train_dataset and val_dataset
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    train_dataset = combined_dataset

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True)
    fold = 1

    # Initialize lists to store metrics for each fold
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    # Initialize variables to store the best model  
    best_val_accuracy = 0.0
    best_model = None
        
    for train_index, val_index in kf.split(train_dataset):
        BATCH_SIZE = 32  # Define the batch size
        print(f"Training on Fold {fold}")
        
        # Create train and validation data loaders for the current fold
        train_fold_dataset = torch.utils.data.Subset(train_dataset, train_index)
        val_fold_dataset = torch.utils.data.Subset(train_dataset, val_index)
        train_loader = torch.utils.data.DataLoader(train_fold_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_fold_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Set up the model, optimizer, scheduler, and criterion
        model, optimizer, scheduler, criterion = setup_model_optimizer()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        model.to(device) # Move model to GPU if available

        epochs = 1  # Number of epochs to train the model
        patience = 20  # Number of epochs to wait for improvement in validation loss before early stopping
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
        # Append the metrics for the current fold to the lists    
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accuracies.append(train_accuracies)
        all_val_accuracies.append(val_accuracies)

        # Increment the fold number    
        fold += 1  

        # Save the best model based on validation accuracy
        current_val_accuracy = max(val_accuracies)
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            #best_model_state = model.state_dict()
            #print("Best model updated.")
                        
        # Reinitialize the model for the next fold
        model = None
        
    # Calculate average training and validation loss
    avg_train_loss = np.mean([min(fold_losses) for fold_losses in all_train_losses])
    avg_val_loss = np.mean([min(fold_losses) for fold_losses in all_val_losses])
      
    # Calculate average training and validation accuracy
    avg_train_accuracy = np.mean([max(fold_accuracies) for fold_accuracies in all_train_accuracies])
    avg_val_accuracy = np.mean([max(fold_accuracies) for fold_accuracies in all_val_accuracies])
        
    # Calculate variability between folds
    val_accuracy_std = np.std([max(fold_accuracies) for fold_accuracies in all_val_accuracies])
        
    print("Average Training Loss:", avg_train_loss)
    print("Average Validation Loss:", avg_val_loss)
    print("Average Training Accuracy:", avg_train_accuracy)
    print("Average Validation Accuracy:", avg_val_accuracy)
    print("Variability between Folds (Validation Accuracy):", val_accuracy_std)

    # Plotting all training losses
    plt.figure(figsize=(10, 5))
    for i, fold_losses in enumerate(all_train_losses):
        plt.plot(range(1, len(fold_losses) + 1), fold_losses, label=f'Fold {i+1}')
    plt.title('Training Loss per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting all validation losses
    plt.figure(figsize=(10, 5))
    for i, fold_losses in enumerate(all_val_losses):
        plt.plot(range(1, len(fold_losses) + 1), fold_losses, label=f'Fold {i+1}')
    plt.title('Validation Loss per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting all training accuracies
    plt.figure(figsize=(10, 5))
    for i, fold_accuracies in enumerate(all_train_accuracies):
        plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, label=f'Fold {i+1}')
    plt.title('Training Accuracy per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plotting all validation accuracies
    plt.figure(figsize=(10, 5))
    for i, fold_accuracies in enumerate(all_val_accuracies):
        plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies, label=f'Fold {i+1}')
    plt.title('Validation Accuracy per Fold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies):
    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix (Unnormalized)', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.yticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.show()

def plot_normalized_confusion_matrix(cm, classes, title='Confusion Matrix (Normalized)', cmap=plt.cm.Blues):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.yticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.show()

def visualize_model(model, dataloader, class_names, num_images=12):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')

                # Display the image
                plt.imshow(inputs.cpu().data[j].permute(1, 2, 0).numpy())

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def main():
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        ROOT_DIR, 
        DATA_DIR, 
        LABELS_FILE, 
        IMG_DIR, 
        train_transform = get_train_transforms(), 
        val_test_transform = get_val_test_transforms())

    # Concatenate train_dataset and val_dataset
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    train_dataset = combined_dataset

    # Perform k-fold cross-validation
    kfold_cross_validation(train_dataset, val_dataset, k=2)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Set up the model, optimizer, scheduler, and criterion
    model, optimizer, scheduler, criterion = setup_model_optimizer()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device) # Move model to GPU if available

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=1,
        patience=20,
        save_frequency=5
        )
    
    #visualize_model(model, test_loader, CLASS_NAMES, num_images=6)

    # Plot the training and validation loss
    plot_loss(train_losses, val_losses)

    # Plot the training and validation accuracy
    plot_accuracy(train_accuracies, val_accuracies)

    # Evaluate the model on the test set
    conf_matrix, conf_matrix_normalized = evaluate_model(model, test_loader, criterion, device)

    # Plot the confusion matrices
    plot_confusion_matrix(conf_matrix, CLASS_NAMES)
    plot_normalized_confusion_matrix(conf_matrix_normalized, CLASS_NAMES)

    '''# Save the model as a TorchScript
    torch.jit.save(torch.jit.script(model), 'ResNet50-MSL-v2_1.pt')    
    print("Model saved as TorchScript.")'''

    # Save the model as to ONNX
    model.eval()
    model.cuda()
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    torch.onnx.export(model, dummy_input, 'ResNet50-MSL-v2_1.onnx', verbose=True)
    print("Model saved as ONNX.")

if __name__ == '__main__':
        main()