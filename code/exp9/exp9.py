import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load IMDB dataset
max_features = 10000  # Only consider top 10k words
maxlen = 200  # Only consider first 200 words of each review

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# 2. Model builder function
def build_model(rnn_layer):
    model = keras.Sequential([
        layers.Embedding(max_features, 128),
        rnn_layer,
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Build models
rnn_models = {
    'Simple RNN': build_model(layers.SimpleRNN(32)),
    'LSTM': build_model(layers.LSTM(32)),
    'GRU': build_model(layers.GRU(32))
}

# 4. Train and evaluate
history_dict = {}
for name, model in rnn_models.items():
    print(f"\nTraining {name}...")
    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=64,
        validation_split=0.2,
        verbose=2
    )
    history_dict[name] = history

# 5. Visualize results
plt.figure(figsize=(12,6))
# Accuracy plots
for name, history in history_dict.items():
    plt.plot(history.history['accuracy'], linestyle='-', label=f'{name} Train Acc')
    plt.plot(history.history['val_accuracy'], linestyle='--', label=f'{name} Val Acc')
plt.title('Training and Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
# Loss plots
for name, history in history_dict.items():
    plt.plot(history.history['loss'], linestyle='-', label=f'{name} Train Loss')
    plt.plot(history.history['val_loss'], linestyle='--', label=f'{name} Val Loss')
plt.title('Training and Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 6. Final test accuracy and loss
results = {}
for name, model in rnn_models.items():
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = {'test_acc': test_acc, 'test_loss': test_loss}
    print(f"{name} - Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Bar chart for test accuracy and test loss
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(results.keys(), [v['test_acc'] for v in results.values()], color=['#1f77b4','#ff7f0e','#2ca02c'])
plt.title('Test Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0,1)
for i, v in enumerate([v['test_acc'] for v in results.values()]):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center')
plt.subplot(1,2,2)
plt.bar(results.keys(), [v['test_loss'] for v in results.values()], color=['#1f77b4','#ff7f0e','#2ca02c'])
plt.title('Test Loss')
plt.ylabel('Loss')
for i, v in enumerate([v['test_loss'] for v in results.values()]):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()