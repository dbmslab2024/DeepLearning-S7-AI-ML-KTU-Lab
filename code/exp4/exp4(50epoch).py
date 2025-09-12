import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class FeedforwardNN(nn.Module):
    def __init__(self, hidden_units, activation):
        super(FeedforwardNN, self).__init__()
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, hidden_units[0])
        self.act1 = activations[activation]
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.act2 = activations[activation]
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.act3 = activations[activation]
        self.fc4 = nn.Linear(hidden_units[2], 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.fc4(x)
        return x

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def predict_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))  # Ensure image is 32x32
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        class_name = CLASS_NAMES[class_idx]
    print(f"Predicted class: {class_name} (index {class_idx})")
    return class_name

def train_and_test(hidden_units, activation, run_id):
    print(f"\n--- Run {run_id} ---")
    print(f"Hidden units: {hidden_units}, Activation: {activation}")

    model = FeedforwardNN(hidden_units, activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(trainloader):.4f}")

    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    runs = [
        # (hidden_units, activation)
        ((512, 256, 128), "relu"),
        ((512, 256, 128), "tanh"),
        ((512, 256, 128), "sigmoid"),
        ((256, 128, 64), "relu"),
        ((256, 128, 64), "tanh"),
        ((256, 128, 64), "sigmoid"),
        ((1024, 512, 256), "relu"),
        ((1024, 512, 256), "tanh"),
        ((1024, 512, 256), "sigmoid"),
    ]
    results = []
    last_model = None
    last_hidden_units = None
    last_activation = None
    for i, (hidden_units, activation) in enumerate(runs, 1):
        model = FeedforwardNN(hidden_units, activation).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # Train
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(trainloader):.4f}")
        # Test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        acc = 100 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
        results.append((i, hidden_units, activation, acc))
        last_model = model
        last_hidden_units = hidden_units
        last_activation = activation

    print("\n--- Summary ---")
    for run_id, hidden_units, activation, acc in results:
        print(f"Run {run_id}: Hidden units={hidden_units}, Activation={activation}, Test Acc={acc:.2f}%")

    # Predict on 4 random images from the test set and display them
    n_images = 4
    indices = random.sample(range(len(testset)), n_images)
    images = []
    true_labels = []
    pred_labels = []
    for idx in indices:
        img, label = testset[idx]
        # Predict
        img_input = img.unsqueeze(0).to(device)
        last_model.eval()
        with torch.no_grad():
            outputs = last_model(img_input)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
        images.append(img)
        true_labels.append(CLASS_NAMES[label])
        pred_labels.append(CLASS_NAMES[class_idx])

    # Plot images with predicted and true labels
    fig, axs = plt.subplots(1, n_images, figsize=(12, 3))
    for i in range(n_images):
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 0.5) + 0.5  # unnormalize
        axs[i].imshow(img_np)
        axs[i].axis('off')
        axs[i].set_title(f"GT: {true_labels[i]}\nPred: {pred_labels[i]}")
    plt.tight_layout()
    plt.show()