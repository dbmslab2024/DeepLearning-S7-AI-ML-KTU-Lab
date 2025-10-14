import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import requests
import os
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download IMDB dataset if not already downloaded
def download_imdb_dataset():
    dataset_dir = 'imdb_dataset'
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Download positive and negative reviews
    base_url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = 'aclImdb_v1.tar.gz'
    
    if not os.path.exists(f"{dataset_dir}/{filename}"):
        print(f"Downloading IMDB dataset...")
        r = requests.get(base_url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        with open(f"{dataset_dir}/{filename}", 'wb') as f:
            for data in tqdm(r.iter_content(chunk_size=1024), total=total_size//1024, unit='KB'):
                f.write(data)
        
        # Extract the tar.gz file
        import tarfile
        print("Extracting dataset...")
        with tarfile.open(f"{dataset_dir}/{filename}") as tar:
            tar.extractall(path=dataset_dir)
        
        print("Dataset downloaded and extracted successfully.")
    else:
        print("Dataset already exists.")

# Load and preprocess IMDB dataset
def load_imdb_data():
    try:
        # Try to download the dataset first
        download_imdb_dataset()
        
        # Load the dataset from the extracted files
        pos_files = ['imdb_dataset/aclImdb/train/pos/' + f for f in os.listdir('imdb_dataset/aclImdb/train/pos') if f.endswith('.txt')]
        neg_files = ['imdb_dataset/aclImdb/train/neg/' + f for f in os.listdir('imdb_dataset/aclImdb/train/neg') if f.endswith('.txt')]
        
        # Read the text files and create labels
        texts = []
        labels = []
        
        # Read positive reviews
        for file_path in tqdm(pos_files, desc="Loading positive reviews"):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(1)  # Positive sentiment
        
        # Read negative reviews
        for file_path in tqdm(neg_files, desc="Loading negative reviews"):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(0)  # Negative sentiment
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        return df
        
    except Exception as e:
        print(f"Error loading IMDB dataset: {e}")
        print("Falling back to loading a sample of the dataset...")
        
        # If the download fails, we'll create a small synthetic dataset for demo purposes
        texts = [
            "This movie was amazing! I loved every minute of it.",
            "Worst film I've ever seen. Complete waste of time.",
            "Great acting, captivating story, and beautiful cinematography.",
            "The plot was confusing and the characters were poorly developed.",
            "I was on the edge of my seat the entire time. Highly recommend!",
            "Boring and predictable. Don't waste your money."
        ]
        labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })
        
        return df

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabets and spaces
    return text

# Tokenization and vocabulary building
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: '<PAD>', 1: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2
        
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text):
        tokenized_text = text.split()
        return [self.stoi.get(token, self.stoi['<UNK>']) for token in tokenized_text]

# Custom Dataset class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        # Preprocess and numericalize
        text = preprocess_text(text)
        numericalized_text = self.vocab.numericalize(text)
        
        # Pad or truncate to max_length
        if len(numericalized_text) < self.max_length:
            numericalized_text = numericalized_text + [0] * (self.max_length - len(numericalized_text))
        else:
            numericalized_text = numericalized_text[:self.max_length]
        
        return torch.tensor(numericalized_text), torch.tensor([label], dtype=torch.float)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        # hidden shape: [n_layers, batch_size, hidden_dim]
        
        hidden = hidden[-1, :, :]  # Get the last hidden state
        # hidden shape: [batch_size, hidden_dim]
        
        return self.fc(self.dropout(hidden))

# Training function
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        predictions = torch.argmax(F.softmax(predictions, dim=1), dim=1)
        epoch_acc += (predictions == labels).float().sum().item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset)

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            predictions = torch.argmax(F.softmax(predictions, dim=1), dim=1)
            epoch_acc += (predictions == labels).float().sum().item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset)

# Main function
def main():
    # Load data
    print("Loading IMDB dataset...")
    df = load_imdb_data()
    print(f"Dataset loaded with {len(df)} samples")
    
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}")
    
    # Preprocess text
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    valid_df['processed_text'] = valid_df['text'].apply(preprocess_text)
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)
    
    # Build vocabulary
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(train_df['processed_text'].tolist())
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = IMDBDataset(train_df['processed_text'].tolist(), train_df['label'].tolist(), vocab)
    valid_dataset = IMDBDataset(valid_df['processed_text'].tolist(), valid_df['label'].tolist(), vocab)
    test_dataset = IMDBDataset(test_df['processed_text'].tolist(), test_df['label'].tolist(), vocab)
    
    # Create dataloaders
    BATCH_SIZE = 64
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Model hyperparameters
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1  # Binary classification
    N_LAYERS = 2
    DROPOUT = 0.5
    
    # Initialize model
    model = RNNModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    N_EPOCHS = 5
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
        
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)
        
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc*100:.2f}%")
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'rnn_imdb_model.pt')
            print("Model saved!")
    
    # Load the best model
    model.load_state_dict(torch.load('rnn_imdb_model.pt'))
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)
    print(f"\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
    
    # Show some predictions on real IMDB test reviews
    print("\n===== Sample Predictions from IMDB Test Set =====")
    def predict_sentiment(model, text, vocab, max_length=100):
        model.eval()
        processed_text = preprocess_text(text)
        numericalized_text = vocab.numericalize(processed_text)
        if len(numericalized_text) < max_length:
            numericalized_text = numericalized_text + [0] * (max_length - len(numericalized_text))
        else:
            numericalized_text = numericalized_text[:max_length]
        tensor = torch.tensor(numericalized_text).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor))
            prediction = prediction.item()
        return prediction, "Positive" if prediction >= 0.5 else "Negative"

    # Sample up to 5 reviews, but not more than available in test set
    num_samples = min(5, len(test_df))
    sample_test = test_df.sample(n=num_samples, random_state=42)
    for idx, row in sample_test.iterrows():
        review = row['text']
        true_label = "Positive" if row['label'] == 1 else "Negative"
        prob, sentiment = predict_sentiment(model, review, vocab)
        print(f"Review: {review[:200]}{'...' if len(review) > 200 else ''}")
        print(f"True Label: {true_label}")
        print(f"Predicted: {sentiment} (probability: {prob:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()