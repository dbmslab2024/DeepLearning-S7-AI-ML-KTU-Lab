import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# CNN Architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))      # [batch, 32, 28, 28]
        x = self.pool(x)               # [batch, 32, 14, 14]
        x = F.relu(self.conv2(x))      # [batch, 64, 14, 14]
        x = self.pool(x)               # [batch, 64, 7, 7]
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

model = CNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch}: Training loss: {avg_loss:.4f}')

# Test function
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            # Display predictions for the first batch only
            if batch_idx == 0:
                print("Sample predictions:")
                for i in range(min(10, data.size(0))):
                    print(f"Image {i+1}: True label = {target[i].item()}, Predicted = {pred[i].item()}")
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    return accuracy

def show_random_predictions(model, device, dataset, num_samples=5):
    """Display random test images with their true and predicted labels."""
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        input_img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_img)
            pred = output.argmax(dim=1).item()
        img_np = image.squeeze().cpu().numpy()
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img_np, cmap='gray')
        plt.title(f'True: {label}\nPred: {pred}', color='green' if label==pred else 'red')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main loop
best_acc = 0
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, criterion, epoch)
    acc = test(model, DEVICE, test_loader)
    if acc > best_acc:
        best_acc = acc
        # Save the best model
        torch.save(model.state_dict(), 'best_mnist_cnn.pth')

print(f'Best Test Accuracy: {best_acc:.2f}%')

# Show one random prediction after training is complete
show_random_predictions(model, DEVICE, test_dataset, num_samples=4)