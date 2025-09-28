import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import time
import random

# Set seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading and preprocessing
tokenizer = get_tokenizer("basic_english")
max_len = 500  # Maximum sequence length

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# Get IMDB dataset
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Text processing function
def text_pipeline(text):
    tokens = tokenizer(text)
    return [vocab[token] for token in tokens][:max_len]

# Collate function for DataLoader
def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(label)
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
    
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list.to(device), label_list.to(device)

# DataLoaders
batch_size = 32
train_iter, test_iter = IMDB(split='train'), IMDB(split='test')
train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(list(test_iter), batch_size=batch_size, 
                             shuffle=False, collate_fn=collate_batch)

# Model definitions
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.gru(embedded)
        return self.fc(hidden.squeeze(0))

# Model parameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 2

# Initialize models
rnn_model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
lstm_model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
gru_model = GRU(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# Training function
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for text, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == labels).float().sum()
        
        epoch_loss += loss.item()
        epoch_acc += correct.item()
        
    return epoch_loss / len(dataloader), epoch_acc / (len(dataloader) * batch_size)

# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for text, labels in dataloader:
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            pred_class = torch.argmax(predictions, dim=1)
            correct = (pred_class == labels).float().sum()
            
            epoch_loss += loss.item()
            epoch_acc += correct.item()
            
            all_predictions.extend(pred_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return (epoch_loss / len(dataloader), 
            epoch_acc / (len(dataloader) * batch_size),
            all_predictions, all_labels)

# Training settings
n_epochs = 5
criterion = nn.CrossEntropyLoss()

# Optimizers
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

# Lists to store metrics
models = [("RNN", rnn_model, rnn_optimizer), 
          ("LSTM", lstm_model, lstm_optimizer), 
          ("GRU", gru_model, gru_optimizer)]

history = {model_name: {"train_loss": [], "train_acc": [], 
                        "val_loss": [], "val_acc": [], 
                        "training_time": []} for model_name, _, _ in models}

# Train all models
for model_name, model, optimizer in models:
    print(f"\nTraining {model_name} model...")
    best_val_acc = 0
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate_model(model, test_dataloader, criterion)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # Store metrics
        history[model_name]["train_loss"].append(train_loss)
        history[model_name]["train_acc"].append(train_acc)
        history[model_name]["val_loss"].append(val_loss)
        history[model_name]["val_acc"].append(val_acc)
        history[model_name]["training_time"].append(epoch_time)
        
        print(f"Epoch: {epoch+1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val. Loss: {val_loss:.4f} | Val. Acc: {val_acc*100:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model_name.lower()}_sentiment_model.pt')

# Make predictions with all models
def get_sample_predictions(models, n_samples=5):
    test_samples = []
    
    # Get some test samples
    test_iter = IMDB(split='test')
    for i, (label, text) in enumerate(test_iter):
        if i < n_samples:
            test_samples.append((label, text))
        else:
            break
    
    results = []
    
    for label, text in test_samples:
        sample_result = {"text": text[:200] + "..." if len(text) > 200 else text, 
                         "actual": "Positive" if label == 1 else "Negative"}
        
        # Process text
        processed_text = torch.tensor([text_pipeline(text)], dtype=torch.int64).to(device)
        
        # Get predictions from each model
        for model_name, model, _ in models:
            model.eval()
            with torch.no_grad():
                prediction = model(processed_text)
                pred_class = torch.argmax(prediction, dim=1).item()
                pred_label = "Positive" if pred_class == 1 else "Negative"
                sample_result[model_name] = pred_label
                
        results.append(sample_result)
    
    return results

# Load the best models
for model_name, model, _ in models:
    model.load_state_dict(torch.load(f'{model_name.lower()}_sentiment_model.pt'))

# Get sample predictions
sample_predictions = get_sample_predictions(models)

# Display sample predictions
print("\nSample Predictions:")
for i, sample in enumerate(sample_predictions):
    print(f"\nSample {i+1}:")
    print(f"Text: {sample['text']}")
    print(f"Actual: {sample['actual']}")
    
    for model_name, _, _ in models:
        print(f"{model_name} prediction: {sample[model_name]}")

# Visualize the results
plt.figure(figsize=(18, 12))

# Plot training & validation accuracy
plt.subplot(2, 2, 1)
for model_name, _, _ in models:
    plt.plot(history[model_name]["train_acc"], label=f"{model_name} Train")
    plt.plot(history[model_name]["val_acc"], linestyle='--', label=f"{model_name} Val")
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(2, 2, 2)
for model_name, _, _ in models:
    plt.plot(history[model_name]["train_loss"], label=f"{model_name} Train")
    plt.plot(history[model_name]["val_loss"], linestyle='--', label=f"{model_name} Val")
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot final validation accuracy
plt.subplot(2, 2, 3)
model_names = [name for name, _, _ in models]
final_val_acc = [history[name]["val_acc"][-1] * 100 for name, _, _ in models]
plt.bar(model_names, final_val_acc)
plt.title('Final Validation Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
for i, v in enumerate(final_val_acc):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center')

# Plot average training time per epoch
plt.subplot(2, 2, 4)
avg_times = [np.mean(history[name]["training_time"]) for name, _, _ in models]
plt.bar(model_names, avg_times)
plt.title('Average Training Time per Epoch')
plt.xlabel('Model')
plt.ylabel('Time (seconds)')
for i, v in enumerate(avg_times):
    plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

plt.tight_layout()
plt.savefig('rnn_comparison_results.png')
plt.show()

# Final test accuracy
print("\nFinal Test Accuracy:")
for model_name, model, _ in models:
    _, test_acc, predictions, labels = evaluate_model(model, test_dataloader, criterion)
    print(f"{model_name}: {test_acc*100:.2f}%")

# Compare the number of parameters in each model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Parameters:")
for model_name, model, _ in models:
    print(f"{model_name}: {count_parameters(model):,} parameters")