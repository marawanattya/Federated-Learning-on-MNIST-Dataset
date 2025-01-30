# Federated Learning on MNIST Dataset

This repository implements a **Federated Learning** system using PyTorch to train a simple neural network on the MNIST dataset. The system simulates multiple clients (devices) that train locally on their data and then aggregate their model updates into a global model. This approach is particularly useful for privacy-preserving machine learning, where data cannot be centralized.

## Table of Contents

1. [Introduction](#introduction)
2. [Federated Learning Overview](#federated-learning-overview)
3. [Dataset](#dataset)
4. [Implementation Details](#implementation-details)
   - [Model Architecture](#model-architecture)
   - [Federated Averaging](#federated-averaging)
   - [Local Training](#local-training)
   - [Global Model Testing](#global-model-testing)
5. [Results](#results)
6. [Usage](#usage)
7. [Contributing](#contributing)


---

## Introduction

Federated Learning is a decentralized machine learning approach where multiple clients (e.g., devices) collaboratively train a shared model without sharing their raw data. This repository demonstrates a simple implementation of Federated Learning using the MNIST dataset. The system consists of:

- A **global model** that aggregates updates from multiple clients.
- **Local models** that train on their respective subsets of the data.
- **Federated Averaging** to combine the updates from local models into the global model.

---

## Federated Learning Overview

Federated Learning involves the following steps:

1. **Initialization**: A global model is initialized and distributed to all clients.
2. **Local Training**: Each client trains the model on its local data for a few epochs.
3. **Model Aggregation**: The global model aggregates the updates from all clients using Federated Averaging.
4. **Global Testing**: The global model is evaluated on a centralized test dataset.
5. **Iteration**: Steps 2-4 are repeated for multiple global epochs.

This process ensures that the global model improves over time while keeping the data decentralized.

---

## Dataset

The **MNIST dataset** is used for this project. It consists of 70,000 grayscale images of handwritten digits (0-9), split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels.

- **Training Data**: Split among multiple clients for local training.
- **Test Data**: Used to evaluate the global model's performance.

---

## Implementation Details

### Model Architecture

A simple feedforward neural network is used for this project:

- **Input Layer**: 28x28 = 784 neurons (flattened image).
- **Hidden Layer**: 128 neurons with ReLU activation.
- **Output Layer**: 10 neurons (one for each digit) with softmax activation.

### Federated Averaging

The global model aggregates updates from all clients by averaging their model parameters.

### Local Training

Each client trains the model on its local data using Stochastic Gradient Descent (SGD).

### Global Model Testing

The global model is evaluated on the test dataset to measure its accuracy.

---

## Results

The global model's accuracy improves over multiple global epochs. Here are the results from a sample run:

- **Global Epoch 1/10**: Test Accuracy: 86.23%
- **Global Epoch 2/10**: Test Accuracy: 88.95%
- **Global Epoch 3/10**: Test Accuracy: 89.80%
- **Global Epoch 4/10**: Test Accuracy: 90.49%
- **Global Epoch 5/10**: Test Accuracy: 91.00%

The model achieves **91.93% accuracy** after 5 global epochs with 3 clients and **91.00% accuracy** after 5 global epochs with 5 clients.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/federated-learning-mnist.git
   cd federated-learning-mnist
   ```

2. Install the required dependencies:
   ```bash
   pip install torch torchvision
   ```

3. Run the script:
   ```bash
   python federated_learning.py
   ```

4. Adjust the hyperparameters (e.g., number of clients, local epochs, global epochs, learning rate) in the `main()` function.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

This README provides an overview of the Federated Learning implementation, including setup instructions, usage, and results. For more details, refer to the code and comments in the script.
