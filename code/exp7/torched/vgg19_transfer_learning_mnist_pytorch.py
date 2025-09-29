"""
Experiment 7: Digit Classification using VGG-19 with Transfer Learning (PyTorch Version)
This script explores both fixed feature extraction and fine-tuning approaches
Optimized for CPU execution
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import random

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform MNIST to 3 channels and resize to 32x32 for VGG-19
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Datasets and loaders (subset for faster CPU training)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_subset, _ = torch.utils.data.random_split(train_dataset, [10000, len(train_dataset)-10000])
test_subset, _ = torch.utils.data.random_split(test_dataset, [2000, len(test_dataset)-2000])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

# Utility functions
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)

def plot_history(history, label):
    plt.plot(history['val_acc'], label=f'{label} (val acc)')
    plt.plot(history['train_acc'], label=f'{label} (train acc)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation/Training Accuracy')
    plt.legend()

def plot_loss(history, label):
    plt.plot(history['val_loss'], label=f'{label} (val loss)')
    plt.plot(history['train_loss'], label=f'{label} (train loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation/Training Loss')
    plt.legend()

def show_sample_predictions(model, dataset, num_samples=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        input_img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_img)
            pred = output.argmax(dim=1).item()
            conf = torch.softmax(output, dim=1)[0, pred].item()
        img_np = img.permute(1,2,0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img_np)
        plt.title(f'True: {label}\nPred: {pred}\nConf: {conf*100:.1f}%', color='green' if label==pred else 'red', fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true, pred, model_name):
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def print_classification_report(true, pred, model_name):
    print(f"\n{model_name} - Classification Report:")
    print(classification_report(true, pred, digits=4))


def get_vgg19_model(finetune=False, unfreeze_last_n=4):
    vgg19 = models.vgg19(weights='IMAGENET1K_V1')
    for param in vgg19.parameters():
        param.requires_grad = False
    if finetune:
        # Unfreeze last n layers
        ct = 0
        for child in reversed(list(vgg19.features.children())):
            if ct < unfreeze_last_n:
                for param in child.parameters():
                    param.requires_grad = True
                ct += 1
    vgg19.classifier[6] = nn.Linear(4096, 10)
    vgg19 = vgg19.to(DEVICE)
    return vgg19

# 1. VGG-19 as fixed feature extractor
vgg19_fixed = get_vgg19_model(finetune=False)
optimizer_fixed = optim.Adam(vgg19_fixed.classifier[6].parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
history_fixed = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

print("Training with VGG-19 as fixed feature extractor...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(vgg19_fixed, train_loader, criterion, optimizer_fixed)
    val_loss, val_acc, _, _ = evaluate(vgg19_fixed, test_loader, criterion)
    history_fixed['train_acc'].append(train_acc)
    history_fixed['val_acc'].append(val_acc)
    history_fixed['train_loss'].append(train_loss)
    history_fixed['val_loss'].append(val_loss)
    print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

# 2. Fine-tuning the last 4 VGG-19 layers
vgg19_finetune = get_vgg19_model(finetune=True, unfreeze_last_n=4)
optimizer_finetune = optim.Adam(filter(lambda p: p.requires_grad, vgg19_finetune.parameters()), lr=LR/10)
history_finetune = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

print("\nFine-tuning last 4 VGG-19 layers...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(vgg19_finetune, train_loader, criterion, optimizer_finetune)
    val_loss, val_acc, _, _ = evaluate(vgg19_finetune, test_loader, criterion)
    history_finetune['train_acc'].append(train_acc)
    history_finetune['val_acc'].append(val_acc)
    history_finetune['train_loss'].append(train_loss)
    history_finetune['val_loss'].append(val_loss)
    print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

# Plot results
plt.figure(figsize=(8,5))
plot_history(history_fixed, "Feature Extractor")
plot_history(history_finetune, "Fine-tuned")
plt.show()

plt.figure(figsize=(8,5))
plot_loss(history_fixed, "Feature Extractor")
plot_loss(history_finetune, "Fine-tuned")
plt.show()

# Evaluate and visualize
val_loss, val_acc, pred_fixed, true_fixed = evaluate(vgg19_fixed, test_loader, criterion)
val_loss, val_acc, pred_finetune, true_finetune = evaluate(vgg19_finetune, test_loader, criterion)

show_sample_predictions(vgg19_fixed, test_subset, num_samples=5)
show_sample_predictions(vgg19_finetune, test_subset, num_samples=5)

plot_confusion_matrix(true_fixed, pred_fixed, "Feature Extractor")
plot_confusion_matrix(true_finetune, pred_finetune, "Fine-tuned")

print_classification_report(true_fixed, pred_fixed, "Feature Extractor")
print_classification_report(true_finetune, pred_finetune, "Fine-tuned")

print(f"\nBest Test Accuracy (Feature Extractor): {max(history_fixed['val_acc']):.2f}%")
print(f"Best Test Accuracy (Fine-tuned): {max(history_finetune['val_acc']):.2f}%")
