import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import yfinance as yf

df = yf.download('^NSEI', start='2024-09-29', end='2025-09-29', progress=False)
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load and preprocess data
#df = pd.read_csv('DeepLearning-S7-AI-ML-KTU-Lab/code/exp10/NIFTY 50-29-09-2024-to-29-09-2025.csv')
# No need to strip columns or parse 'Date' for yfinance DataFrame

# Extract the closing price data
data = df['Close'].values.astype(float)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

# 2. Create sequences for time series prediction
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Use a longer sequence length for better context
seq_length = 5

# Create sequences
X, y = create_sequences(data_normalized, seq_length)

# 3. Split data into training and testing sets (80-20 split)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# 5. Create PyTorch Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 8
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 6. Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 8. Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch = X_batch.view(-1, seq_length, 1)
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {running_loss/len(train_loader):.6f}')

# 9. Evaluate on test set
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.view(-1, seq_length, 1)
        y_pred = model(X_batch)
        predictions.extend(y_pred.numpy())
        actuals.extend(y_batch.numpy())

# 10. Inverse transform to get actual values
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# 11. Calculate metrics
mse = mean_squared_error(actuals, predictions)
rmse = math.sqrt(mse)
mae = mean_absolute_error(actuals, predictions)

print(f'Test RMSE: {rmse:.2f}')
print(f'Test MAE: {mae:.2f}')

# 12. Visualize predictions vs. actual values
plt.figure(figsize=(14, 10))

# Plot 1: Full test set - Actual vs Predicted
plt.subplot(2, 1, 1)
plt.plot(actuals, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='orange')
plt.title('NIFTY-50 Test Set: Actual vs Predicted')
plt.xlabel('Time Index')
plt.ylabel('NIFTY-50 Index Value')
plt.legend()
plt.grid(True)

# Plot 2: Zoomed-in (last 30 test points)
plt.subplot(2, 1, 2)
zoom_points = 30
plt.plot(range(len(actuals)-zoom_points, len(actuals)), actuals[-zoom_points:], label='Actual', color='blue')
plt.plot(range(len(predictions)-zoom_points, len(predictions)), predictions[-zoom_points:], label='Predicted', color='orange')
plt.title(f'NIFTY-50 Test Set: Last {zoom_points} Points (Zoomed In)')
plt.xlabel('Time Index')
plt.ylabel('NIFTY-50 Index Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()