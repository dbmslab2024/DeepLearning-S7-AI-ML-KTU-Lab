"""
Experiment 7: Digit Classification using VGG-19 with Transfer Learning
This script explores both fixed feature extraction and fine-tuning approaches
Optimized for CPU execution
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Set memory growth and CPU usage
tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
print("Using CPU for computation")

class MNISTTransferLearning:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.history_fixed = None
        self.history_finetuned = None
        self.model_fixed = None
        self.model_finetuned = None
        
    def load_and_preprocess_data(self):
        """Load MNIST dataset and preprocess for VGG-19"""
        print("Loading MNIST dataset...")
        
        # Load MNIST
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Use subset for faster CPU training
        train_samples = 10000  # Reduced for CPU efficiency
        test_samples = 2000
        
        x_train = x_train[:train_samples]
        y_train = y_train[:train_samples]
        x_test = x_test[:test_samples]
        y_test = y_test[:test_samples]
        
        # Convert grayscale to RGB and resize to 32x32 (minimum for VGG)
        print("Preprocessing images for VGG-19...")
        x_train_rgb = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test_rgb = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
        
        # Resize to 32x32 (smaller than standard 224x224 for CPU efficiency)
        x_train_resized = tf.image.resize(x_train_rgb, (32, 32))
        x_test_resized = tf.image.resize(x_test_rgb, (32, 32))
        
        # Normalize
        self.x_train = tf.keras.applications.vgg19.preprocess_input(x_train_resized)
        self.x_test = tf.keras.applications.vgg19.preprocess_input(x_test_resized)
        
        # One-hot encode labels
        self.y_train = keras.utils.to_categorical(y_train, 10)
        self.y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training samples: {len(self.x_train)}")
        print(f"Test samples: {len(self.x_test)}")
        print(f"Input shape: {self.x_train.shape[1:]}")
        
    def create_base_model(self):
        """Create VGG-19 base model"""
        # Load VGG-19 without top layers
        base_model = VGG19(
            input_shape=(32, 32, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        return base_model
    
    def build_fixed_feature_model(self):
        """Build model using VGG-19 as fixed feature extractor"""
        print("\n" + "="*50)
        print("Building Fixed Feature Extractor Model...")
        print("="*50)
        
        base_model = self.create_base_model()
        
        # Freeze all layers in base model
        base_model.trainable = False
        
        # Create new model
        inputs = keras.Input(shape=(32, 32, 3))
        x = base_model(inputs, training=False)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        self.model_fixed = keras.Model(inputs, outputs)
        
        # Compile model
        self.model_fixed.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Total layers: {len(self.model_fixed.layers)}")
        print(f"Trainable parameters: {self.count_params(self.model_fixed, trainable=True):,}")
        print(f"Non-trainable parameters: {self.count_params(self.model_fixed, trainable=False):,}")
        
    def build_finetuned_model(self):
        """Build model with fine-tuning of top VGG-19 layers"""
        print("\n" + "="*50)
        print("Building Fine-Tuned Model...")
        print("="*50)
        
        base_model = self.create_base_model()
        
        # Unfreeze top layers for fine-tuning
        base_model.trainable = True
        
        # Freeze all layers except last 4
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Create new model
        inputs = keras.Input(shape=(32, 32, 3))
        x = base_model(inputs, training=True)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        self.model_finetuned = keras.Model(inputs, outputs)
        
        # Compile with lower learning rate for fine-tuning
        self.model_finetuned.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Total layers: {len(self.model_finetuned.layers)}")
        print(f"Trainable parameters: {self.count_params(self.model_finetuned, trainable=True):,}")
        print(f"Non-trainable parameters: {self.count_params(self.model_finetuned, trainable=False):,}")
        
    def count_params(self, model, trainable=True):
        """Count model parameters"""
        if trainable:
            return sum([tf.size(w).numpy() for w in model.trainable_weights])
        else:
            return sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    def train_model(self, model, model_name, epochs=10):
        """Train a model and return history"""
        print(f"\nTraining {model_name}...")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
        
        start_time = time.time()
        
        history = model.fit(
            self.x_train, self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return history, training_time
    
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Get test accuracy
        test_loss, test_acc = model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Get predictions
        predictions = model.predict(self.x_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test, axis=1)
        
        print(f"\n{model_name} Results:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_acc, test_loss, pred_classes, true_classes
    
    def visualize_training_history(self):
        """Plot training history for both models"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Fixed Feature Extractor - Accuracy
        axes[0, 0].plot(self.history_fixed.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history_fixed.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Fixed Feature Extractor - Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Fixed Feature Extractor - Loss
        axes[0, 1].plot(self.history_fixed.history['loss'], label='Train')
        axes[0, 1].plot(self.history_fixed.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Fixed Feature Extractor - Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Fine-tuned Model - Accuracy
        axes[1, 0].plot(self.history_finetuned.history['accuracy'], label='Train')
        axes[1, 0].plot(self.history_finetuned.history['val_accuracy'], label='Validation')
        axes[1, 0].set_title('Fine-tuned Model - Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Fine-tuned Model - Loss
        axes[1, 1].plot(self.history_finetuned.history['loss'], label='Train')
        axes[1, 1].plot(self.history_finetuned.history['val_loss'], label='Validation')
        axes[1, 1].set_title('Fine-tuned Model - Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, model, model_name, num_samples=10):
        """Visualize model predictions on test samples"""
        # Get random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f'{model_name} - Sample Predictions', fontsize=14)
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Get prediction
            img = self.x_test[idx:idx+1]
            pred = model.predict(img, verbose=0)
            pred_class = np.argmax(pred)
            true_class = np.argmax(self.y_test[idx])
            confidence = np.max(pred) * 100
            
            # Display image (convert back from VGG preprocessing)
            display_img = np.array(img[0])  # Convert tensor to numpy array
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            
            axes[i].imshow(display_img)
            axes[i].axis('off')
            
            color = 'green' if pred_class == true_class else 'red'
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)', 
                             color=color, fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_classification_details(self, model, model_name, num_samples=6):
        """Detailed visualization showing probability distribution for each prediction"""
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
        fig.suptitle(f'{model_name} - Detailed Classification Analysis', fontsize=16, y=1.02)
        
        for i, idx in enumerate(indices):
            # Get prediction
            img = self.x_test[idx:idx+1]
            pred = model.predict(img, verbose=0)[0]
            pred_class = np.argmax(pred)
            true_class = np.argmax(self.y_test[idx])
            
            # Display image
            display_img = np.array(img[0])
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            
            axes[i, 0].imshow(display_img)
            axes[i, 0].axis('off')
            
            result_text = "✓ Correct" if pred_class == true_class else "✗ Wrong"
            color = 'green' if pred_class == true_class else 'red'
            axes[i, 0].set_title(f'True: {true_class}, Pred: {pred_class} - {result_text}', 
                                 color=color, fontsize=12, fontweight='bold')
            
            # Plot probability distribution
            bars = axes[i, 1].bar(range(10), pred, color='lightblue', edgecolor='navy')
            bars[pred_class].set_color('green' if pred_class == true_class else 'red')
            if pred_class != true_class:
                bars[true_class].set_color('orange')  # Highlight true class in orange
            
            axes[i, 1].set_xlabel('Digit Class')
            axes[i, 1].set_ylabel('Probability')
            axes[i, 1].set_title(f'Confidence: {pred[pred_class]*100:.2f}%')
            axes[i, 1].set_xticks(range(10))
            axes[i, 1].set_ylim([0, 1])
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add probability values on top of bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.01:  # Only show if probability > 1%
                    axes[i, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_misclassified(self, model, pred_classes, true_classes, model_name):
        """Show examples of misclassified images"""
        misclassified_indices = np.where(pred_classes != true_classes)[0]
        
        if len(misclassified_indices) == 0:
            print(f"No misclassified samples found for {model_name}!")
            return
        
        num_samples = min(8, len(misclassified_indices))
        sample_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(f'{model_name} - Misclassified Examples', fontsize=14)
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            # Get image and predictions
            img = self.x_test[idx:idx+1]
            pred = model.predict(img, verbose=0)[0]
            pred_class = pred_classes[idx]
            true_class = true_classes[idx]
            
            # Display image
            display_img = np.array(img[0])
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            
            axes[i].imshow(display_img)
            axes[i].axis('off')
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {pred[pred_class]*100:.1f}%', 
                             color='red', fontsize=10)
        
        # Hide unused subplots
        for i in range(num_samples, 8):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, true_classes, pred_classes, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_classes, pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    
    def compare_models(self, fixed_acc, fixed_time, finetuned_acc, finetuned_time):
        """Create comparison visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        models = ['Fixed Feature\nExtractor', 'Fine-tuned\nModel']
        accuracies = [fixed_acc, finetuned_acc]
        colors = ['#3498db', '#e74c3c']
        
        bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.7)
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.4f}', ha='center', va='bottom')
        
        # Training time comparison
        times = [fixed_time, finetuned_time]
        bars2 = axes[1].bar(models, times, color=colors, alpha=0.7)
        axes[1].set_ylabel('Training Time (seconds)')
        axes[1].set_title('Training Time Comparison')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, t in zip(bars2, times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{t:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print improvement analysis
        print("\n" + "="*50)
        print("PERFORMANCE IMPROVEMENT ANALYSIS")
        print("="*50)
        
        acc_improvement = ((finetuned_acc - fixed_acc) / fixed_acc) * 100
        print(f"\nAccuracy Improvement: {acc_improvement:.2f}%")
        print(f"Fixed Feature Extractor Accuracy: {fixed_acc:.4f}")
        print(f"Fine-tuned Model Accuracy: {finetuned_acc:.4f}")
        
        time_difference = finetuned_time - fixed_time
        print(f"\nTraining Time Difference: {time_difference:.2f} seconds")
        print(f"Fixed Feature Extractor Time: {fixed_time:.2f} seconds")
        print(f"Fine-tuned Model Time: {finetuned_time:.2f} seconds")
        
        if acc_improvement > 0:
            print(f"\n✅ Fine-tuning improved accuracy by {acc_improvement:.2f}%")
        else:
            print(f"\n❌ Fixed feature extraction performed better by {abs(acc_improvement):.2f}%")
    
    def run_experiment(self):
        """Run the complete experiment"""
        print("\n" + "="*60)
        print("MNIST DIGIT CLASSIFICATION WITH VGG-19 TRANSFER LEARNING")
        print("="*60)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Experiment 1: Fixed Feature Extractor
        print("\n" + "-"*50)
        print("EXPERIMENT 1: FIXED FEATURE EXTRACTOR")
        print("-"*50)
        self.build_fixed_feature_model()
        self.history_fixed, fixed_time = self.train_model(
            self.model_fixed, 
            "Fixed Feature Extractor",
            epochs=10
        )
        fixed_acc, fixed_loss, fixed_pred, fixed_true = self.evaluate_model(
            self.model_fixed,
            "Fixed Feature Extractor"
        )
        
        # Experiment 2: Fine-tuned Model
        print("\n" + "-"*50)
        print("EXPERIMENT 2: FINE-TUNED MODEL")
        print("-"*50)
        self.build_finetuned_model()
        self.history_finetuned, finetuned_time = self.train_model(
            self.model_finetuned,
            "Fine-tuned Model",
            epochs=10
        )
        finetuned_acc, finetuned_loss, finetuned_pred, finetuned_true = self.evaluate_model(
            self.model_finetuned,
            "Fine-tuned Model"
        )
        
        # Visualizations
        print("\n" + "="*50)
        print("VISUALIZATION AND ANALYSIS")
        print("="*50)
        
        # Training history
        self.visualize_training_history()
        
        # Sample predictions for both models
        self.visualize_predictions(self.model_fixed, "Fixed Feature Extractor")
        self.visualize_predictions(self.model_finetuned, "Fine-tuned Model")
        
        # Detailed classification analysis
        self.visualize_classification_details(self.model_fixed, "Fixed Feature Extractor")
        self.visualize_classification_details(self.model_finetuned, "Fine-tuned Model")
        
        # Misclassified examples
        self.visualize_misclassified(self.model_fixed, fixed_pred, fixed_true, 
                                    "Fixed Feature Extractor")
        self.visualize_misclassified(self.model_finetuned, finetuned_pred, finetuned_true,
                                    "Fine-tuned Model")
        
        # Confusion matrices
        self.plot_confusion_matrix(fixed_true, fixed_pred, "Fixed Feature Extractor")
        self.plot_confusion_matrix(finetuned_true, finetuned_pred, "Fine-tuned Model")
        
        # Model comparison
        self.compare_models(fixed_acc, fixed_time, finetuned_acc, finetuned_time)
        
        # Detailed classification reports
        print("\n" + "="*50)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*50)
        
        print("\nFixed Feature Extractor - Classification Report:")
        print(classification_report(fixed_true, fixed_pred, 
                                   target_names=[str(i) for i in range(10)]))
        
        print("\nFine-tuned Model - Classification Report:")
        print(classification_report(finetuned_true, finetuned_pred,
                                   target_names=[str(i) for i in range(10)]))
        
        return {
            'fixed_accuracy': fixed_acc,
            'finetuned_accuracy': finetuned_acc,
            'fixed_time': fixed_time,
            'finetuned_time': finetuned_time
        }

# Main execution
if __name__ == "__main__":
    # Create experiment instance
    experiment = MNISTTransferLearning()
    
    # Run the complete experiment
    results = experiment.run_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFinal Results Summary:")
    print(f"Fixed Feature Extractor: {results['fixed_accuracy']:.4f} accuracy")
    print(f"Fine-tuned Model: {results['finetuned_accuracy']:.4f} accuracy")
    print(f"Best performing model: {'Fine-tuned' if results['finetuned_accuracy'] > results['fixed_accuracy'] else 'Fixed Feature'}")