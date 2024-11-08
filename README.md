# Sign-language-NN-from-Scratch

Neural Network for Sign Language Classification
This project implements a neural network from scratch to classify images of American Sign Language (ASL) letters using Python and basic libraries (NumPy, Pandas, Matplotlib). The model aims to recognize hand gestures representing different letters, helping to bridge communication for those using ASL.

Table of Contents
1. Dataset
2. Code Structure
3. Model Architecture
4. Training Logic
5. Dependencies
6. Run Instructions

## 1. Dataset:
Kaggle Link : https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data
The dataset consists of labeled images of ASL hand gestures represented as 28x28 pixel grayscale images, flattened into 784-feature rows in a CSV file format. Each row corresponds to one image, with:

Label: The ASL letter being represented, converted to a numeric class.
Pixels: Grayscale pixel values, normalized to improve model performance.
The dataset is split into a training set and a validation set to evaluate model generalization.

## 2. Code Structure
Data Loading and Preprocessing:

The dataset is loaded as a CSV, shuffled, normalized, and split into training and validation sets.
Data is transposed to facilitate matrix operations for neural network computations.

Functions:
init_params(): Initializes weights and biases.
forward_prop(): Implements the forward pass of the neural network.
backward_prop(): Computes gradients of the loss with respect to parameters.
update_params(): Updates weights and biases based on gradients.
make_predictions(): Generates predictions from the model.
get_accuracy(): Computes the accuracy of predictions.
Training with Gradient Descent:

The model is trained using gradient descent, with options for learning rate decay and early stopping.
Evaluation:
The evaluate_test_set function assesses the model on test data.

## 3. Model Architecture
The neural network has a simple two-layer architecture:

Input Layer: 784 neurons for each pixel in the image.
Hidden Layer: 128 neurons with ReLU activation, capturing essential features.
Output Layer: 25 neurons representing each ASL letter (one-hot encoded).
This setup was chosen for simplicity and the ease of interpretability while achieving reasonable accuracy for ASL letter classification tasks wihtout using any ML library.
Architecture: Input Layer (784 nodes) -> Hidden Layer (128 nodes, ReLU) -> Output Layer (25 nodes, Softmax)

## 4. Training Logic
The model uses mini-batch gradient descent with learning rate decay and early stopping.
  Learning Rate Decay: Reduces the learning rate at fixed intervals, preventing overshooting and improving convergence.
  Early Stopping: Stops training if validation accuracy doesn’t improve, avoiding overfitting.
  Early Stopping and Learning Rate Decay
  Learning Rate Decay: Every set number of epochs, the learning rate is decayed by a fixed rate.
  Early Stopping: Monitors validation accuracy to stop training if improvement stalls.
  Mathematics Behind Activation Functions:
    ReLU (Rectified Linear Unit): Converts output into probability distributions, making it suitable for multi-class classification.
    Softmax : Converts output into probability distributions, making it suitable for multi-class classification.
Results and Evaluation: After training, the model achieved an accuracy of approximately 79% on the training set and 59% on the test set. The disparity indicates some room 
for improvement, possibly with a deeper architecture or further hyperparameter tuning.

## 5. Dependencies
  Python 3.x
  Libraries: NumPy, Pandas, Matplotlib

## 6. Run Instructions
  Clone this repository.
  Run the notebook (.ipynb file) step-by-step to execute the code and observe model training and evaluation.
  Visualize training progress through accuracy plots generated after training.
