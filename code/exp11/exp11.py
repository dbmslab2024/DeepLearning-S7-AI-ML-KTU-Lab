import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
import unicodedata
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)

# Download dataset from Kaggle
# You'll need to download the English to Hindi dataset from Kaggle
# https://www.kaggle.com/datasets/aiswaryaramachandran/hindienglish-corpora
file_path="DeepLearning-S7-AI-ML-KTU-Lab/code/exp11/Hindi_English_Truncated_Corpus.csv"

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    return df

# Preprocessing functions
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z0-9.!?०-९अ-ह\s]', r'', s)
    s = re.sub(r'\s+', r' ', s)
    return s

# Tokenization
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # Start count with special tokens

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, eng_sentences, hindi_sentences, eng_lang, hindi_lang):
        self.eng_sentences = eng_sentences
        self.hindi_sentences = hindi_sentences
        self.eng_lang = eng_lang
        self.hindi_lang = hindi_lang
    
    def __len__(self):
        return len(self.eng_sentences)
    
    def __getitem__(self, idx):
        eng_sentence = self.eng_sentences[idx]
        hindi_sentence = self.hindi_sentences[idx]
        
        # Convert words to indices
        eng_indices = [self.eng_lang.word2index.get(word, self.eng_lang.word2index["<UNK>"]) for word in eng_sentence.split()]
        hindi_indices = [self.hindi_lang.word2index.get(word, self.hindi_lang.word2index["<UNK>"]) for word in hindi_sentence.split()]
        
        # Add SOS and EOS tokens
        eng_indices = [self.eng_lang.word2index["<SOS>"]] + eng_indices + [self.eng_lang.word2index["<EOS>"]]
        hindi_indices = [self.hindi_lang.word2index["<SOS>"]] + hindi_indices + [self.hindi_lang.word2index["<EOS>"]]
        
        return torch.tensor(eng_indices), torch.tensor(hindi_indices)

# Collate function for padding sequences in batch
def collate_fn(batch):
    eng_batch, hindi_batch = [], []
    for eng_item, hindi_item in batch:
        eng_batch.append(eng_item)
        hindi_batch.append(hindi_item)
    
    # Pad sequences
    eng_batch = pad_sequence(eng_batch, batch_first=True, padding_value=0)
    hindi_batch = pad_sequence(hindi_batch, batch_first=True, padding_value=0)
    
    return eng_batch, hindi_batch

# Encoder model
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# Decoder model
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input, hidden):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output)
        return output, hidden

# Training function
def train_step(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, eng_batch, hindi_batch, teacher_forcing_ratio=0.5):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    batch_size = eng_batch.size(0)
    target_length = hindi_batch.size(1)
    
    # Run encoder
    encoder_output, encoder_hidden = encoder(eng_batch)
    
    # Prepare decoder input - start with SOS token for each sentence
    decoder_input = torch.tensor([[1] * batch_size]).view(batch_size, 1).to(eng_batch.device)
    
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
    
    loss = 0
    
    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input
        for i in range(1, target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output.squeeze(1), hindi_batch[:, i])
            decoder_input = hindi_batch[:, i].unsqueeze(1)  # Next input is current target
    else:
        # Without teacher forcing: use network's own predictions as the next input
        for i in range(1, target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1)  # Detach from history as input
            loss += criterion(decoder_output.squeeze(1), hindi_batch[:, i])
    
    loss.backward()
    
    # Clip gradients to prevent exploding gradient issues
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

# Main training loop
def train(encoder, decoder, train_dataloader, n_epochs, learning_rate=0.001):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for i, (eng_batch, hindi_batch) in enumerate(train_dataloader):
            loss = train_step(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, eng_batch, hindi_batch)
            total_loss += loss
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}, Batch {i}, Loss: {loss:.4f}')
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}')

# Translation function
def translate(encoder, decoder, sentence, eng_lang, hindi_lang, max_length=50):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Process input sentence
        sentence = normalize_string(sentence)
        words = sentence.split()
        indices = [eng_lang.word2index.get(word, eng_lang.word2index["<UNK>"]) for word in words]
        indices = [eng_lang.word2index["<SOS>"]] + indices + [eng_lang.word2index["<EOS>"]]
        input_tensor = torch.tensor([indices]).long()
        
        # Encode
        encoder_output, encoder_hidden = encoder(input_tensor)
        
        # Prepare decoder input
        decoder_input = torch.tensor([[hindi_lang.word2index["<SOS>"]]])
        
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        
        # Decode
        for i in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            
            if topi.item() == hindi_lang.word2index["<EOS>"]:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(hindi_lang.index2word[topi.item()])
            
            decoder_input = topi.detach()
        
        # Remove SOS and EOS tokens
        if "<EOS>" in decoded_words:
            decoded_words = decoded_words[:decoded_words.index("<EOS>")]
        
        return ' '.join(decoded_words)

# Main function
def main():
    # Load dataset
    # Assume you've downloaded the dataset from Kaggle
    df = load_dataset(file_path)
    print(f"Dataset loaded with {len(df)} samples")
    
    # Preprocess data
    df['english'] = df['english'].apply(normalize_string)
    df['hindi'] = df['hindi'].apply(normalize_string)
    
    # Create language instances
    eng_lang = Lang('english')
    hindi_lang = Lang('hindi')
    
    # Add words to vocabularies
    for eng, hindi in zip(df['english'], df['hindi']):
        eng_lang.add_sentence(eng)
        hindi_lang.add_sentence(hindi)
    
    print(f"English vocabulary size: {eng_lang.n_words}")
    print(f"Hindi vocabulary size: {hindi_lang.n_words}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Create datasets
    train_dataset = TranslationDataset(train_df['english'].values, train_df['hindi'].values, eng_lang, hindi_lang)
    val_dataset = TranslationDataset(val_df['english'].values, val_df['hindi'].values, eng_lang, hindi_lang)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize models
    hidden_size = 256
    encoder = EncoderRNN(eng_lang.n_words, hidden_size)
    decoder = DecoderRNN(hidden_size, hindi_lang.n_words)
    
    # Train the model
    n_epochs = 10
    train(encoder, decoder, train_dataloader, n_epochs)
    
    # Save models
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')
    
    # Test translation
    test_sentences = [
        "Hello how are you?",
        "I love learning about artificial intelligence",
        "The weather is beautiful today"
    ]
    
    for sentence in test_sentences:
        translation = translate(encoder, decoder, sentence, eng_lang, hindi_lang)
        print(f"English: {sentence}")
        print(f"Hindi: {translation}")
        print()

if __name__ == "__main__":
    main()