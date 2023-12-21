# Multilayer Perceptron Neural Network for classification of XOR, Spiral, Circle and Gaussian Datasets

In this project I created a multi-layer neural network learner for XOR, Gaussian and Circle dataset. 

### Goal
- For a given dataset,construct and train neural network classifier using provided labeled training data using nothing other than numpy library.
- Use the learned classifier to classify the unlabeled test data
- Output the predictions of your classifier on the test data into a file in the same directory for XOR, Gaussian and Circle dataset


### Model description
- The model implements a feed-forward neural network, with 3 hidden layers. 
- Model Body:
    - Input: 4 Nodes, Output Layer: 20 nodes, Activation Function: Relu
    - Input: 20 Nodes, Output Layer: 10 nodes, Activation Function: Relu
    - Input: 10 Nodes, Output Layer: 1 nodes, Activation Function: Sigmoid

- The loss function used is cross-entropy and for back propogation, it uses gradient descent algorithm.

### Results

- XOR: 93.2%
- Circle: 91%
- Gaussian: 96.6%
- Spiral: 97%