# Kaggle--Digit-Recognizer
This repository contains code file for Handwritten Digit Recognizer(Kaggle Competition) in Python using TF/NumPy/Pandas. LeNet-5 was used as the network architecture for recognizing 28x28 images of Handwritten digits from 0 to 9. A training accuracy of about 97.2% was obtained and prediction accuracy of about 97.6% was obtained on Kaggle submission portal. 

The architecture used for the network was:

Conv(5x5,32 channels stride=1) --> max-pool( stride=2) --> conv (5x5, 28 channels stride=1) --> max-pool(stride=2) --> flattened layer --> 3 fully connected layers(with tanh or relu) --> softmax layer (log probs) --> predicted output

The network was trained for about 10 epoch, with each epoch loading about (total number of training samples/batch size) amount of samples usinng minibatch SGD. 

Files used:
* helper.py: This Python code file consists of helper functions necessary to run the neural network model. The helper functions include data pre-processing methods, reading datasets, mini-batch SGD function, functions to convert Dense to one-hot and vice-versa.

* imshow.py: This file consists of methods which can be used to display labelled and predicted images

* NN_main: Contains functions to initialize parameters(weights and biases) of the network, model and to train the NeuralNet

Instructions to run on command Line(Ubuntu) bash:

Download or clone from the github repopsitory
```
$wget https://github.com/kulka193/Kaggle--Digit-Recognizer/archive/master.zip
```
unzip to extract files
```
$unzip master.zip
```
change directory to your file path
```
$cd "Your downloaded directory"
```
Run the main file
```
$python NN_main.py
```
