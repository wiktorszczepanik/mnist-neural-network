## Artificial Neural Network

A simple neural network implementation written from scratch in Go and trained to recognize handwritten digits from the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

The project implements the core components of a feed-forward neural network, including forward propagation, backpropagation, weight initialization, and gradient-based learning, without relying on external machine learning libraries.

The model can achieve around **98% accuracy** on the MNIST test dataset with current configuration.

## Default network architecture

* 784 ~ input layer (28 × 28 pixels)
* 128 ~ first hidden layer
* 64 ~ second hidden layer
* 10 ~ output layer representing digits 0-9

The current implementation uses `ReLU` activation function.

## Dataset

The project uses the standard MNIST dataset:

* 60,000 training images
* 10,000 test images
* 28 × 28 pixels per image
* 784 input values per image
* grayscale pixel values from 0 to 255

## Setup

Before running, download the MNIST dataset and place the following files in the *data/* directory:

```
data/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

The application expects these files at the paths defined in `main.go`
You can download the MNIST dataset from the official MNIST website.