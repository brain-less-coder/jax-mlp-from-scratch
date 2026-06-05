# Build an MLP in JAX from Scratch

A complete implementation of a Multi-Layer Perceptron (MLP) in JAX, built from first principles. This project covers the entire training pipeline, from random number generation and parameter initialization to forward propagation, loss computation, automatic differentiation, and optimization using Stochastic Gradient Descent (SGD).

## Overview

The objective of this project was to understand how neural networks work under the hood by implementing every major component manually using JAX's functional programming paradigm.

Unlike high-level frameworks that hide training details behind a single API call, this implementation explicitly handles:

* PRNG key management
* Weight initialization
* Data generation
* Forward propagation
* Activation functions
* Softmax and Log-Softmax
* Cross-Entropy Loss
* Automatic differentiation with `jax.grad`
* SGD parameter updates
* Training loops
* Prediction and evaluation

---

## Features

### Data Generation

* Explicit PRNG key creation and splitting
* Standard normal feature sampling
* Synthetic label generation
* One-hot encoding

### Neural Network Components

* Fully connected (linear) layers
* ReLU activation
* Multi-layer perceptron architecture
* Logits computation

### Training Pipeline

* Numerically stable Softmax
* Numerically stable Log-Softmax
* Cross-Entropy Loss
* Automatic gradient computation using `jax.grad`
* SGD optimization
* Full-batch training

### Evaluation

* Classification accuracy
* Class prediction from logits

---

## Concepts Implemented

### Random Number Generation

* `make_prng_key`
* `split_prng_key`

### Data Utilities

* `sample_normal_matrix`
* `sample_input_features`
* `assign_class_labels`
* `one_hot_encode_labels`

### Parameter Initialization

* `init_linear_layer`
* `init_mlp_params`

### Forward Pass

* `linear_forward`
* `relu_activation`
* `softmax_probabilities`
* `log_softmax_logits`
* `mlp_forward`

### Training Components

* `cross_entropy_loss`
* `loss_fn_of_params`
* `compute_param_grads`
* `sgd_update_params`
* `training_step`
* `train_mlp`

### Evaluation Components

* `classification_accuracy`
* `predict_classes`

---

## Technologies Used

* Python
* JAX
* JAX NumPy

---

## Learning Outcomes

Through this project I learned:

* Functional programming concepts in JAX
* Explicit random number handling with PRNG keys
* Neural network parameter initialization
* Forward and backward propagation
* Automatic differentiation using `jax.grad`
* Optimization using Stochastic Gradient Descent
* Numerical stability techniques for Softmax and Cross-Entropy
* End-to-end neural network training

---

## Example Workflow

```python
key = make_prng_key(0)

params = init_mlp_params(
    key,
    [4, 8, 3]
)

x = sample_input_features(
    make_prng_key(1),
    batch_size=64,
    num_features=4
)

labels = assign_class_labels(x, 3)
targets = one_hot_encode_labels(labels, 3)

params = train_mlp(
    params,
    x,
    targets,
    learning_rate=0.1,
    num_epochs=100
)

predictions = predict_classes(params, x)
```

---

## Acknowledgment

This project was completed as part of the Deep-ML learning exercises.

Deep-ML provides hands-on machine learning and deep learning exercises focused on understanding concepts through implementation.

Source: https://www.deep-ml.com/

The code in this repository represents my own implementation and solutions developed while working through the Deep-ML curriculum.

---

## License

This project is intended for educational and learning purposes.
