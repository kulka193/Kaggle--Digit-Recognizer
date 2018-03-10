# Kaggle--Digit-Recognizer
This repository contains code file for Handwritten Digit Recognizer(Kaggle Competition) in Python using TF/NumPy/Pandas. LeNet-5 was used as the network architecture for recognizing 28x28 images of Handwritten digits from 0 to 9. A training accuracy of about 97.2% was obtained and prediction accuracy of about 97.6% was obtained on Kaggle submission portal. The architecture used was:

Conv(5x5,32 channels stride=1)--max-pool( stride=2) -- conv (5x5, 28 channels stride=1) -- max-pool(stride=2) -- flattened layer --3 fully connected layers(with tanh or relu) -- softmax layer (log probs) -- predicted output
