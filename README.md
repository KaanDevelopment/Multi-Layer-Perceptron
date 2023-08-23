Introduction
This code implements an MLP for digit classification, which can be trained and tested on datasets. The code supports different training configurations and provides options for two-fold testing. You can customize the network architecture, initialization, cost functions, optimizers, and more.

Usage
To use this code, follow these steps:
Clone the repository or download the code files.
Compile the Java code if needed.
Run the TrainPerceptrons class, which serves as the main entry point for training and testing the MLP.
Review and customize the configuration settings in the code to suit your specific dataset and requirements.

Features
MLP architecture with customizable layers and activations.
Options for cost functions and optimizers.
Support for two-fold testing for evaluating the MLP.
Training process with configurable batch size and epochs.
Calculation of precision, recall, and confusion matrix for evaluation.

Configuration
You can customize various aspects of the MLP and training process by modifying the code:
Adjust the network architecture by adding or modifying layers in the MultiLayerPerceptron.Builder.
Set the batch size, learning rate, and other hyperparameters in the trainFold method.
Choose the cost function and optimizer for backpropagation.
Configure the number of epochs and stopping criteria.

Testing Modes
The code supports two testing modes:
Normal Testing: In this mode, the MLP is trained and tested on the provided datasets conventionally.
Two-Fold Testing: This mode evaluates the MLP using a two-fold testing approach, where it trains on one dataset and tests on another. This can help assess the model's generalization.

Training Process
The training process consists of the following steps:
Initialize the MLP with the specified architecture, weights, and hyperparameters.
Perform training iterations, updating weights using backpropagation.
Optionally, evaluate the model's performance on the test dataset without updating weights.
Stop training when the specified criteria are met, such as achieving a low error rate or a maximum number of epochs.

Confusion Matrix
The code calculates a confusion matrix to assess the model's classification performance. It provides precision and recall values for each class, as well as overall accuracy.

Contributors
Kaan Gulsel

License
MIT license
Feel free to modify and extend this code to suit your needs. If you have any questions or encounter issues, please reach out to the contributors for assistance.
