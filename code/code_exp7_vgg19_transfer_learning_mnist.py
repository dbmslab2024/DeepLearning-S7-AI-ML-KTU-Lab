import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

# Transform MNIST to 3 channels and resize to 224x224 for VGG-19
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Datasets and loaders
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def plot_history(history, label):
    plt.plot(history['val_acc'], label=f'{label} (val)')
    plt.plot(history['train_acc'], label=f'{label} (train)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation/Training Accuracy')
    plt.legend()

def get_vgg19_model(finetune=False):
    vgg19 = models.vgg19(pretrained=True)
    for param in vgg19.parameters():
        param.requires_grad = finetune
    vgg19.classifier[6] = nn.Linear(4096, 10)  # MNIST has 10 classes
    vgg19 = vgg19.to(DEVICE)
    return vgg19

# 1. VGG-19 as fixed feature extractor (only classifier is trained)
vgg19_fixed = get_vgg19_model(finetune=False)
for param in vgg19_fixed.classifier[6].parameters():
    param.requires_grad = True

optimizer_fixed = optim.Adam(vgg19_fixed.classifier[6].parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
history_fixed = {'train_acc': [], 'val_acc': []}

print("Training with VGG-19 as fixed feature extractor...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(vgg19_fixed, train_loader, criterion, optimizer_fixed)
    val_loss, val_acc = evaluate(vgg19_fixed, test_loader, criterion)
    history_fixed['train_acc'].append(train_acc)
    history_fixed['val_acc'].append(val_acc)
    print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

# 2. Fine-tuning the whole VGG-19
vgg19_finetune = get_vgg19_model(finetune=True)
optimizer_finetune = optim.Adam(vgg19_finetune.parameters(), lr=LR)
history_finetune = {'train_acc': [], 'val_acc': []}

print("\nFine-tuning all VGG-19 layers...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train(vgg19_finetune, train_loader, criterion, optimizer_finetune)
    val_loss, val_acc = evaluate(vgg19_finetune, test_loader, criterion)
    history_finetune['train_acc'].append(train_acc)
    history_finetune['val_acc'].append(val_acc)
    print(f'Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')

# 3. Plot results
plt.figure(figsize=(8,5))
plot_history(history_fixed, "Feature Extractor")
plot_history(history_finetune, "Fine-tuned")
plt.show()

# 4. Print summary of best performance
print(f"\nBest Test Accuracy (Feature Extractor): {max(history_fixed['val_acc']):.2f}%")
print(f"Best Test Accuracy (Fine-tuned): {max(history_finetune['val_acc']):.2f}%")