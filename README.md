# Kaggle--Digit-Recognizer
This repository contains code file for Handwritten Digit Recognizer(Kaggle Competition) in Python using TF/NumPy/Pandas. LeNet-5 was used as the network architecture for recognizing 28x28 images of Handwritten digits from 0 to 9. A training accuracy of about 97.2% was obtained and prediction accuracy of about 97.6% was obtained on Kaggle submission portal. 

The architecture used for the network was:

Conv(5x5,32 channels stride=1) --> max-pool( stride=2) --> conv (5x5, 28 channels stride=1) --> max-pool(stride=2) --> flattened layer --> 3 fully connected layers with 400, 120 and 80 neurons in each of the layer (with tanh or relu) --> softmax layer (log probs) --> predicted output

The network was trained for about 10 epoch, with each epoch loading about (total number of training samples/batch size) amount of samples usinng minibatch SGD. 

Files used:
* helper.py: This Python code file consists of helper functions necessary to run the neural network model. The helper functions include data pre-processing methods, reading datasets, mini-batch SGD function, functions to convert Dense to one-hot and vice-versa.

* imshow.py: This file consists of methods which can be used to display labelled and predicted images

* NN_main: Contains functions to initialize parameters(weights and biases) of the network, to prepare the NN model and to train the network

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
Results:
=========

The below snapshot shows the cost function in eacch epoch and displays the training accuracy at the end of training phase:
![training](https://user-images.githubusercontent.com/30439795/37257059-d2f756da-2531-11e8-81d0-6d6d606e01ae.PNG)

Predicted labels from the test data and corresponding predicted value by our LeNet-5 for that particular index:

![3](https://user-images.githubusercontent.com/30439795/37257137-f8abd5ee-2532-11e8-9d7f-6b55c1213239.PNG)
=========
![5](https://user-images.githubusercontent.com/30439795/37257138-f8cc7be6-2532-11e8-9dbb-fcc7d87e378a.PNG)
=========
![8](https://user-images.githubusercontent.com/30439795/37257139-f8ed5302-2532-11e8-885c-7b85f0611d93.PNG)
=========
![6](https://user-images.githubusercontent.com/30439795/37257140-f9500ba0-2532-11e8-82ed-2f4eb99ba6ed.PNG)
=========
![](https://user-images.githubusercontent.com/30439795/37257141-f9714c5c-2532-11e8-8124-a801df7a17b7.PNG)
=========
