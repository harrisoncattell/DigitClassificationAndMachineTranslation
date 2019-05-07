## Digital Classification and Natural Language Processing

### Description

This is a Python-based application that carries out digit classification and natural language processing on the MNIST dataset and a dataset of English-German sentences using the Scikit-Learn library and Tensorflow/Keras
The application also carries out analysis on the results

### Prerequisites

When running this application, Scikit-Learn is needed for the digit classificion, Tensorflow and Keras is needed for the natural language processing section.

### Compiling the code

Compiling the code requires Python and a number of libraries that can be found in the code. Use the following code in the terminal:

```
py DigitClassificationAndNLP.py
```

The training, validation and testing sets are included within the downloaded files.

### Properties of the program

The 3 models trained in the digit classification are: K-Nearest Neighbour, Random Decision Forest, and Convolutional Neural Network.

- The KNN model trains with neighbour values from 1 to 10
- The Random Decision Forest trains with n_iterator values of 10, 100, and 1000
- The convolutional neural network trains a standard 1 convolutional layer model and an implementation of [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).

The natural language processing model uses LSTM model provided by Keras. The 3 different implementations have a varying number of hidden from 256 to 1024.


Created by Harrison Cattell, 2019
