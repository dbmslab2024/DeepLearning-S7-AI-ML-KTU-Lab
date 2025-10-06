Deep Learning Lab Instructions & Program List
============================================

General Instructions:
---------------------
- Each student must bring a rough record with the algorithm written in it.
- Kindly convey messages to the entire class and ensure students follow instructions as per their lab slots.
- Programs should be done in VS Code using PyTorch (unless otherwise specified).
- Submit rough records up to the third experiment along with given program algorithms.

Batch Info:
-----------
- Tomorrow DL LAB for Batch 1 (3/07/2025)
- Naale lab is for batch 2 (27/07/2025)
- Lab for BATCH 2 (24/08/2025)
- Batch 2: Last lab for deep learning (28/09/2025)

Lab Programs & Experiments:
--------------------------

**Lab Date: 3/07/2025**
1. Decision Tree (ID3)  
   - Write a program to demonstrate the working of the decision tree based ID3 algorithm.
   - Use play tennis data set for building the decision tree and apply this knowledge to classify a new sample.

2. Linear SVM Classifier  
   - Implement and evaluate a linear Support Vector Machine (SVM) classifier for the Iris dataset.

3. Gradient Descent for Linear Regression  
   - Implement and visualize the Gradient Descent algorithm to find the optimal parameters (slope and intercept) for a simple linear regression model.

**Lab Date: 17/07/2025**
1. Simple Linear Regression  
   - Implement Simple Linear Regression with Synthetic Data.

2. Basic Image Enhancement  
   - Implement basic image enhancement operations such as histogram equalization, morphological operations.

**Lab Date: 3/08/2025**
1. Feed Forward Neural Network (CIFAR-10)  
   - Implement Feed forward neural network with three hidden layers for classification on CIFAR-10 dataset.
   - Design and train a neural network that achieves high accuracy in classifying the images.
   - Various runs with different hidden units and activations:
     - Run 1: (512, 256, 128), relu
     - Run 2: (512, 256, 128), tanh
     - Run 3: (512, 256, 128), sigmoid
     - Run 4: (256, 128, 64), relu
     - Run 5: (256, 128, 64), tanh
     - Run 6: (256, 128, 64), sigmoid
     - Run 7: (1024, 512, 256), relu
     - Run 8: (1024, 512, 256), tanh
     - Run 9: (1024, 512, 256), sigmoid

2. Optimization & Regularization Studies  
   - Train the network using Adam optimizer (baseline).
   - (a) Train with Xavier initialization; compare accuracy/convergence.
   - (b) Train with Kaiming initialization; compare accuracy/convergence.
   - (c) Apply dropout regularization; analyze effect on accuracy and overfitting.
   - (d) Apply L2 regularization; analyze effect on accuracy and prevention of overfitting.

**Lab Date: 24/08/2025**
Exp 6: Convolutional Neural Network (MNIST)  
   - Implement a CNN architecture for digit classification on MNIST dataset.

Exp 7: Transfer Learning (MNIST, VGGnet-19)  
   - Digit classification using pre-trained networks like VGGnet-19.
   - Analyze and visualize performance improvement.
   - Explore transfer learning using ConvNets as fixed feature extractors and fine-tuning.
   - Compare fixed feature extractor approach with fine-tuning on new image classification tasks.

**Lab Date: 18/09/2025**
expt 8: RNN for Sentiment Classification (IMDB)  
   - Implement a Recurrent Neural Network (RNN) for review classification on IMDB dataset (positive/negative).

exp 9: LSTM/GRU vs RNN (IMDB)  
   - Analyze and visualize performance changes using LSTM and GRU instead of standard RNN.
   - Compare performance of different architectures on sentiment classification.

**Lab Date: 28/09/2025**
Exp 10: Time Series Forecasting (NIFTY-50)  
   - Implement time series forecasting for the NIFTY-50 dataset. Predict future values based on historical data.

Exp 11: Autoencoder for Machine Translation (Kaggle Eng-Hindi)  
   - Implement a shallow autoencoder and decoder network for machine translation using Kaggle English to Hindi Neural Translation Dataset.
   - Design and train a model to translate English sentences to Hindi using autoencoders and decoders.

-------------------------------------------
End of Instructions
