#PROGRAM DEVELOPED BY HARRISON CATTELL CAT15562670 for CMP9137M

import matplotlib as mat
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as reviewMetrics
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Embedding, RepeatVector
from keras.layers import Conv2D, MaxPool2D, AvgPool2D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras import optimizers
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from matplotlib import pyplot as plt
from tabulate import tabulate
import statistics
import pickle
import string 
import re 
from numpy import array, argmax, random, take 
import pandas as pd
import keras
import time
import tensorflow as tf
import numpy as np
import os
import sys
import time
import math

#Custom function to stop tensorflow to showing deprecated messages from https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
def tensorflow_shutup():
 
        try:
                # noinspection PyPackageRequirements
                import os
                from tensorflow import logging
                logging.set_verbosity(logging.ERROR)
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
                # noinspection PyUnusedLocal
                def deprecated(date, instructions, warn_once=True):
                        
                        def deprecated_wrapper(func):
                                return func
                                
                        return deprecated_wrapper

                from tensorflow.python.util import deprecation
                
                deprecation.deprecated = deprecated

        except ImportError:
                
                pass

#Digit Classification
class Digit_Classifier:

        #Main arrays for training data
        #2D arrays
        trainingData = np.empty([0, 784])
        validationData = np.empty([0, 784])
        testingData = np.empty([0, 784])

        #1D arrays
        trainingLabelData = []
        validationLabelData = []   
        testingLabelData = []                                                                                
                                                                                                                

        def ProcessData(self):

                #----- PROCESSING TRAINING DATA -----#
                #Variable for interating through image files
                index = 1

                print()
                print("\tPROCESSING TRAINING AND VALIDATION IMAGES")
                print()
                print("\t\tImporting training images from file...")
                start = time.time()

                #Loops through directory to read each image
                for i in os.listdir("Task1Datasets\\TrainingDigits5000"):

                        #Imports image from file
                        inputImage = img.imread("Task1Datasets\\TrainingDigits5000\\" + i)
                
                        #Splits filename to obtain image label                
                        label = i.split("_")
                        self.trainingLabelData.append(int(label[0]))
        
                        #Reshapes image to a 1D array
                        reshapedInputImage = np.reshape(inputImage, 784)

                        #Adds the 1D array to array of images
                        self.trainingData = np.vstack((self.trainingData, reshapedInputImage))

                        index+=1

                end = time.time()
                print()
                print("\t\tImporting training images completed in " + str(math.ceil(end-start)) + "s")
                print()
                

                #----- PROCESSING VALIDATION DATA -----#
                #Variable for interating through image files
                index = 1

                print("\t\tImporting validation images from file...")
                print()

                start = time.time()

                #Loops through directory to read each image
                for i in os.listdir("Task1Datasets\\ValidationDigits1000"):

                        #Imports image from file
                        inputImage = img.imread("Task1Datasets\\ValidationDigits1000\\" + i)
                        
                        #Splits filename to obtain image label
                        label = str(i).split("_")
                        self.validationLabelData.append(int(label[0]))

                        #Reshapes image to a 1D array
                        reshapedInputImage = np.reshape(inputImage, 784)

                        #Adds the 1D array to array of images
                        self.validationData = np.vstack((self.validationData, reshapedInputImage))

                        index+=1

                end = time.time()

                print("\t\tImporting validation images completed in " + (str(math.ceil(end-start))) + "s")

                 #----- PROCESSING TESTING DATA -----#
                #Variable for interating through image files
                index = 1

                print("\t\tImporting testing images from file...")
                print()

                start = time.time()

                #Loops through directory to read each image
                for i in os.listdir("Task1Datasets\\TestingDigits500"):

                        #Imports image from file
                        inputImage = img.imread("Task1Datasets\\TestingDigits500\\" + i)
                        
                        #Splits filename to obtain image label
                        label = str(i).split("_")
                        self.testingLabelData.append(int(label[0]))

                        #Reshapes image to a 1D array
                        reshapedInputImage = np.reshape(inputImage, 784)

                        #Adds the 1D array to array of images
                        self.testingData = np.vstack((self.testingData, reshapedInputImage))

                        index+=1

                end = time.time()

                print("\t\tImporting testing images completed in " + (str(math.ceil(end-start))) + "s")       

        def KNNClassifier(self):

                print()
                print("\tTRAINING K-NEAREST NEIGHBOUR CLASSIFIER")
                print()

                #Sets neighbourhood range
                testingNeighbours = range(1,11)
                validationAccuracy = []
                trainingTime = []
                testingTime = []

                print("\t\tTraining classifier with neighbour range of 1-10...")
                print()

                #Training a number of KNN classifier with 
                for i in testingNeighbours:

                        print("\t\t\tTraining with " + str(i) + " neighbour(s)")

                        #Measuring time taken to train classifier
                        start = time.time()

                        #Initiate variable with classifier type KNN with i neighbour range
                        knnClf = KNeighborsClassifier(n_neighbors=i)

                        #Trains classifer using training data and labels
                        knnClf.fit(self.trainingData, self.trainingLabelData)

                        #Appending this time to an array
                        end = time.time()
                        trainingTime.append(end-start)

                        #Measuring time taken to test classifier
                        start = time.time()

                        #Computes accuracy of model using validation data and labels
                        validationAccuracy.append(round(knnClf.score(self.validationData,self.validationLabelData)*100,4))

                        #Appending this time to an array
                        end = time.time()
                        testingTime.append(end-start)
                        

                print()
                print("\t\tTraining classifiers completed")
                print()

                #Plots graph showing change in accuracy as number of neighbours increases
                plt.title("K-Nearests Neighbour: Accuracy of the classifier using the Validation Set")
                plt.plot(testingNeighbours, validationAccuracy)
                plt.ylabel('Accuracy')
                plt.xlabel('Number of Neighbors')
                plt.legend()
                plt.show()

                #Plots graph showing change in training time as number of neighbours increases
                plt.title("K-Nearests Neighbour: Time taken to train classifier using Training Set")
                plt.plot(testingNeighbours, trainingTime)
                plt.ylabel('Time taken (s)')
                plt.xlabel('Number of Neighbors')
                plt.legend()
                plt.show()

                #Plots graph showing change in testing time as number of neighbours increases
                plt.title("K-Nearests Neighbour: Time taken to test classifier using Validation Set")
                plt.plot(testingNeighbours, testingTime)
                plt.ylabel('Time taken (s)')
                plt.xlabel('Number of Neighbors')
                plt.legend()
                plt.show()

                #Prints accuracy values for each neighbour incrementation in console
                for i in testingNeighbours:

                        print("\t\tNumber of neighbours: " + str(i) + " Accuracy: " + str(validationAccuracy[i-1]))

                print()

        def RandomForest(self):

                print()
                print("\tTRAINING RANDOM FOREST CLASSIFIER")
                print()

                nEstimators = [10,100,1000]
                validationAccuracy = []
                trainingTime = []
                validationTime = []

                print("\t\tTraining classifier with n_estimators equaling 10,100,1000...")
                print()

                for i in nEstimators:
                
                        print("\t\t\tTraining with n_estimators equalling " + str(i))

                        start = time.time()

                        randomForestClf = RandomForestClassifier(n_estimators=i)

                        randomForestClf.fit(self.trainingData, self.trainingLabelData)

                        end = time.time()

                        trainingTime.append(end-start)

                        start = time.time()

                        validationAccuracy.append(round(randomForestClf.score(self.validationData,self.validationLabelData)*100,4))

                        end = time.time()

                        validationTime.append(end-start)

                print()
                print("\t\tTraining classifiers completed")
                print()

                #Plots graph showing change in accuracy as number of neighbours increases
                plt.title("RandomForest Classifier: Accuracy of the classifier using the Validation Set")
                plt.plot(nEstimators, validationAccuracy)
                plt.ylabel('Accuracy')
                plt.xlabel('Number of n_estimators')
                plt.legend()
                plt.show()

                #Plots graph showing change in training time as number of neighbours increases
                plt.title("Random Forest: Time taken to train classifier using Training Set")
                plt.plot(nEstimators, trainingTime)
                plt.ylabel('Time taken (s)')
                plt.xlabel('Number of n_estimators')
                plt.legend()
                plt.show()   

                #Plots graph showing change in testing time as number of neighbours increases
                plt.title("RandomForest Classifier: Time taken to test classifier using Validation Set")
                plt.plot(nEstimators, validationTime)
                plt.ylabel('Time taken (s)')
                plt.xlabel('Number of n_estimators')
                plt.legend()
                plt.show()             

                index = 0

                #Prints accuracy of classifier with different n_estimator values in console
                for i in nEstimators:

                        print("\t\tNumber of n_estimators: " + str(i) + " Accuracy: " + str(validationAccuracy[index]))
                        index+=1

                print()

        def CNeuralNetwork(self):

                print()
                print("\tTRAINING CONVOLUTIONAL NEURAL NETWORKS")
                print()
                
                #Call tensorflow shutup function
                tensorflow_shutup()

                trainingTime = []
                validationTime = []
                validationAccuracy = []

                print("\tReshaping data...")
                print()

                #Reshapes data back to 2D
                X_train = self.trainingData.reshape(self.trainingData.shape[0], 28, 28, 1)
                X_val = self.validationData.reshape(self.validationData.shape[0], 28, 28, 1)
                inputShape = (28, 28, 1)

                #Turns training and validation label data into categorical form
                Y_train = np_utils.to_categorical(self.trainingLabelData, 10)
                Y_val = np_utils.to_categorical(self.validationLabelData, 10)

                # --- MODEL 1 ---
                #Sequential model for a standard 1 convultional layer CNN
                print("\tBuilding model for a standard 1 convolutional layered network...")
                print()
                stdCNN = Sequential()
                stdCNN.add(Conv2D(6, kernel_size=(3,3), input_shape=inputShape))
                stdCNN.add(MaxPool2D(pool_size=(2, 2)))
                stdCNN.add(Flatten()) # Flattening the 2D arrays for fully connected layers
                stdCNN.add(Dense(128, activation=tf.nn.relu))
                stdCNN.add(Dropout(0.2))
                stdCNN.add(Dense(10,activation=tf.nn.softmax))

                print("\tCompiling and training model...")
                print()

                #Compiles model with loss function and optimizer
                stdCNN.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

                #Fits model and calcuates time
                start = time.time()
                historyStdCNN = stdCNN.fit(x=X_train,y=Y_train, epochs=30)
                end = time.time()
                trainingTime.append(end-start)
                

                start = time.time()
                #Prints accuracy score of validation
                scores = stdCNN.evaluate(X_val, Y_val, verbose=0)
                end = time.time()
                validationTime.append(end-start)
                validationAccuracy.append(scores[1]*100)
                
                print()
                print("\tTraining model completed")
                print()

                #Pickle classifer for later use
                modelName = "CNNStandard.pkl"
                with open(modelName, 'wb') as model:
                        pickle.dump(stdCNN,model)

                # --- MODEL 2 ---
                #Sequential model for LeNet-5 architecture
                print("\tBuilding model for a LeNet implementation of a convolutional neural network...")
                print()
                leNET5 = Sequential()
                leNET5.add(Conv2D(6, kernel_size=(5,5), input_shape=inputShape, padding="same", activation='relu'))
                leNET5.add(AvgPool2D(pool_size=(2, 2), strides=2))
                leNET5.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
                leNET5.add(AvgPool2D(pool_size=(2, 2), strides=2))
                leNET5.add(Flatten())
                leNET5.add(Dense(units=120, activation='relu'))
                leNET5.add(Dense(units=84, activation='relu'))
                leNET5.add(Dense(units=10, activation = 'softmax'))

                print("\tCompiling and training model...")
                print()

                #Compiles model with loss function and optimizer
                leNET5.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                #Fits model and calcuates time
                start = time.time()
                historyLeNet = leNET5.fit(x=X_train,y=Y_train, epochs=30)
                end = time.time() 
                trainingTime.append(end-start)

                #Prints accuracy score of validation
                start = time.time()
                scores = leNET5.evaluate(X_val, Y_val, verbose=0)
                end = time.time()
                validationTime.append(end-start)
                validationAccuracy.append(scores[1]*100)

                print()
                print("\tTraining models completed")

                #Pickle classifer for later use
                modelName = "LeNet-5.pkl"
                with open(modelName, 'wb') as model:
                        pickle.dump(leNET5,model)

                #Plots loss graph
                plt.plot(historyStdCNN.history['acc'])
                plt.plot(historyLeNet.history['acc'])
                plt.title('CNN: Accuracy of classifier using validation data')
                plt.ylabel('acc')
                plt.xlabel('epoch')
                plt.legend(['Standard','LeNet-5'], loc='upper left')
                plt.show()

                #Plots accuracy grpah
                plt.plot(historyStdCNN.history['loss'])
                plt.plot(historyLeNet.history['loss'])
                plt.title('CNN: Loss values')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['Standard','LeNet-5'], loc='upper left')
                plt.show()

                objects = ('Standard CNN', 'LeNet-5')
                y_pos = np.arange(len(objects))
                plt.bar(y_pos, trainingTime, align='center', alpha=0.5)
                plt.xticks(y_pos, objects)
                plt.ylabel('Time (s)')
                plt.title('CNN: Time taken to train classifier')
                plt.show()

                objects = ('Standard CNN', 'LeNet-5')
                y_pos = np.arange(len(objects))
                plt.bar(y_pos, validationTime, align='center', alpha=0.5)
                plt.xticks(y_pos, objects)
                plt.ylabel('Time (s)')
                plt.title('CNN: Time take to test classifier using validation data')
                plt.show()

                print()

                #Plots accuracy of validation data
                objects = ('Standard CNN', 'LeNet-5')
                for i in range(len(validationAccuracy)):

                        print("\t\tAccuray of " + str(objects[i]) + ": " + str(validationAccuracy[i]) + "%")

                print()

        def TestingBestPerformingClassifier(self):

                tensorflow_shutup()

                print() 
                print("\tTRAINING BEST CLASSIFIER: LeNet-5")
                print()

                #Best classifier used to test on testing set -> LeNet-5

                print("\tReshaping data...")
                print()

                #Reshapes training and testing data back to 28x28
                X_train = self.trainingData.reshape(self.trainingData.shape[0], 28, 28, 1)
                X_test = self.testingData.reshape(self.testingData.shape[0], 28, 28, 1)
                inputShape = (28, 28, 1)

                #Turns training and testing label data into categorical form
                Y_train = np_utils.to_categorical(self.trainingLabelData, 10)
                Y_test = np_utils.to_categorical(self.testingLabelData, 10)

                #Import picked classifier
                modelName = "LeNet-5.pkl"

                with open(modelName, 'rb') as model:  
                        leNET5Final = pickle.load(model)

                #Predict values using model
                predictedValues = leNET5Final.predict(X_test)

                #Turn labels and predicted labels back into normal non-categorical view
                normalPredY = np.argmax(predictedValues, axis=1)
                normalYTest = np.argmax(Y_test, axis=1)

                #Plots Confusion Matrix
                self.plot_confusion_matrix(normalYTest,normalPredY)       
                plt.show()

                print()
                print("\tPrinting metrics for analysis of testing data")
                print()

                #Calcuating precision, recall, F-score and support
                precision, recall, fscore, support = reviewMetrics(normalYTest, normalPredY)
                
                #Display values in a table
                print(tabulate([['0', precision[0],recall[0],fscore[0]], ['1', precision[1],recall[1],fscore[1]], ['2', precision[2],recall[2],fscore[2]], ['3', precision[3],recall[3],fscore[3]], ['4', precision[4],recall[4],fscore[4]],['5', precision[5],recall[5],fscore[5]], ['6', precision[6],recall[6],fscore[6]],['7', precision[7],recall[7],fscore[7]],['8', precision[8],recall[8],fscore[8]],['9', precision[9],recall[9],fscore[9]]], headers=['Label', 'Precision','Recall','F-Score']))


        ## -------- CUSTOM EXTRA FUNCTIONS -------- ##
        #Custom confusion matrix function from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        def plot_confusion_matrix(self, y_true, y_pred):

                cmap = plt.cm.Blues

                normalize = True

                title = "Confusion matrix for LeNet-5"

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                # Only use the labels that appear in the data
                classes = unique_labels(y_true, y_pred)
                if normalize:
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        print("\tPrinting normalized confusion matrix")
                else:
                        print('\tPrinting Confusion matrix without normalization')

                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
                ax.figure.colorbar(im, ax=ax)
                # We want to show all ticks...
                ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes, yticklabels=classes,title=title,ylabel='True label',xlabel='Predicted label')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                fmt = '.2f' if normalize else 'd'
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")
                fig.tight_layout()
                
        #Main function
        def Main(self):

                #Function for processing data
                self.ProcessData()

                isLoop1 = True

                while isLoop1:

                        print()
                        print("\tPlease select operation")
                        print()
                        print("\t\t1. Train classifiers")
                        print("\t\t2. Test best performaing classifier")
                        print("\t\t3. EXIT")   
                        print()
                        selc = int(input("\tSelection: ")) 

                        #Train classifiers
                        if selc == 1:

                                isLoop2 = True

                                while isLoop2:
                                        
                                        print()
                                        print("\tPlease select a classifier to train")
                                        print()
                                        print("\t\t1. K-Nearest Neighbour")
                                        print("\t\t2. Random Forest")
                                        print("\t\t3. Convolutional Neural Network")
                                        print("\t\t4. Previous Menu")
                                        print("\t\t5. EXIT")   
                                        print()
                                        selc = int(input("\tSelection: "))

                                        if selc == 1:

                                                self.KNNClassifier()

                                        elif selc == 2:

                                                self.RandomForest()

                                        elif selc == 3:
                                                
                                                self.CNeuralNetwork()

                                        elif selc == 4:

                                                isLoop2 = False

                                        elif selc == 5:

                                                isLoop2 = False
                                                isLoop1 = False

                                        else:
                                                pass

                        elif selc == 2:

                                self.TestingBestPerformingClassifier()

                        elif selc == 3:

                                isLoop1 = False

                        else:
                                pass

#Machine Translation  
class MachineTranslation:
    
        def ReadFile(self, filename):

                # open the file 
                file = open(filename, mode='rt', encoding='utf-8')
        
                # read all text 
                text = file.read() 
                file.close() 

                # returns text
                return text

        def SplitIntoLines(self, text): 

                indvSentences = text.strip().split('\n') 
                indvSentences = [i.split('\t') for i in indvSentences] 
                return indvSentences

        # function to build a tokenizer     
        def tokenization(self, lines): 
                tokenizer = Tokenizer() 
                tokenizer.fit_on_texts(lines) 
                return tokenizer

        # Encode and pad sequences 
        def EncodeSquences(self, tokenizer, length, lines):       

                # integer encode sequences          
                seq = tokenizer.texts_to_sequences(lines)          
                # pad sequences with 0 values          
                seq = pad_sequences(seq, maxlen=length, padding='post')           
                return seq

        # Builds model
        def build_model(self, in_vocab, out_vocab, in_timesteps, out_timesteps,n):

                model = Sequential() 
                model.add(Embedding(in_vocab, n, input_length=in_timesteps,   
                mask_zero=True)) 
                model.add(LSTM(n)) 
                model.add(RepeatVector(out_timesteps)) 
                model.add(LSTM(n, return_sequences=True))  
                model.add(Dense(out_vocab, activation='softmax')) 
                return model

        # Gets word in exchange for token
        def get_word(self, n, tokenizer):  
        
                for word, index in tokenizer.word_index.items():                       
                
                        if index == n: 
                                return word

                return None

        #Training Seq2Seq Models Function
        def TrainingSeq2Seq(self):

                print()
                print("\tPROCESSING TRAINING AND VALIDATION TEXT")
                print()

                print("\t\tReading Training and Validation text...")
                print()

                trainingText = self.ReadFile("Task2Datasets\\trainingdata.txt") #Training data
                validationText = self.ReadFile("Task2Datasets\\validationdata.txt") #Validation data
                
                #Split training data into lines
                trainingData = self.SplitIntoLines(trainingText) 
                trainingData = array(trainingData)
                
                validationData = self.SplitIntoLines(validationText) 
                validationData = array(trainingData)
                
                print("\t\tProcessing Training and Validation data...")
                print()

                # Remove punctuation 
                trainingData[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in trainingData[:,0]] 
                trainingData[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in trainingData[:,1]] 
                validationData[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in validationData[:,0]] 
                validationData[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in validationData[:,1]] 

                # convert text to lowercase 
                for i in range(len(trainingData)): 

                        trainingData[i,0] = trainingData[i,0].lower() 
                        trainingData[i,1] = trainingData[i,1].lower()

                # convert text to lowercase 
                for i in range(len(validationData)): 

                        validationData[i,0] = validationData[i,0].lower() 
                        validationData[i,1] = validationData[i,1].lower()

                # prepare english tokenizer
                eng_tokenizer = self.tokenization(trainingData[:, 0]) 
                eng_vocab_size = len(eng_tokenizer.word_index) + 1 
                eng_length = 8
                
                val_eng_tokenizer = self.tokenization(validationData[:, 0]) 
                val_eng_vocab_size = len(val_eng_tokenizer.word_index) + 1 
                val_eng_length = 8 

                # prepare Deutch tokenizer 
                deu_tokenizer = self.tokenization(trainingData[:, 1]) 
                deu_vocab_size = len(deu_tokenizer.word_index) + 1 
                deu_length = 8
                val_deu_tokenizer = self.tokenization(validationData[:, 1]) 
                val_deu_vocab_size = len(val_deu_tokenizer.word_index) + 1 
                val_deu_length = 8

                # prepare training data 
                trainX = self.EncodeSquences(deu_tokenizer, deu_length, trainingData[:, 1]) 
                trainY = self.EncodeSquences(eng_tokenizer, eng_length, trainingData[:, 0]) 

                # prepare validation data 
                valX = self.EncodeSquences(val_deu_tokenizer, val_deu_length, validationData[:, 1]) 
                valY = self.EncodeSquences(val_eng_tokenizer, val_eng_length, validationData[:, 0])
                
                print("\tTRAINING LSTM Model")
                print()

                print("\tTraining Seq2Seq models with 256, 512 and 1024 hidden units..")
                print()

                trainingTime = []

                #Training model with 256 hidden units
                #----------------------------------------------------------------
                print("\t\tCompiling model with 256 hidden units...")
                print()

                # model compilation (with i hidden units)
                NLPModel256 = self.build_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 256)


                rms = optimizers.RMSprop(lr=0.001) 
                NLPModel256.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

                start = time.time()

                history256 = NLPModel256.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), validation_data=(valX, valY.reshape(valY.shape[0],valY.shape[1], 1)),epochs=40, verbose=1)

                end = time.time()
                        
                trainingTime.append(end-start)

                modelName = "NLPModelWith256Unit.pkl"
                        
                with open(modelName, 'wb') as model:
                        pickle.dump(NLPModel256,model)

                #----------------------------------------------------------------

                #Training model with 512 hidden units
                #----------------------------------------------------------------
                print()
                print("\t\tCompiling model with 512 hidden units...")
                print()

                # model compilation (with i hidden units)
                NLPModel512 = self.build_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 512)

                rms = optimizers.RMSprop(lr=0.001) 
                NLPModel512.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

                start = time.time()

                history512 = NLPModel256.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), validation_data=(valX, valY.reshape(valY.shape[0],valY.shape[1], 1)),epochs=40, verbose=1)

                end = time.time()
                        
                trainingTime.append(end-start)

                modelName = "NLPModelWith512Unit.pkl"
                        
                with open(modelName, 'wb') as model:
                        pickle.dump(NLPModel512,model)

                #----------------------------------------------------------------

                #Training model with 1024 hidden units
                #----------------------------------------------------------------
                print()
                print("\t\tCompiling model with 1025 hidden units...")
                print()

                # model compilation (with i hidden units)
                NLPModel1024 = self.build_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 1024)

                rms = optimizers.RMSprop(lr=0.001) 
                NLPModel1024.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

                start = time.time()

                history1024 = NLPModel1024.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), validation_data=(valX, valY.reshape(valY.shape[0],valY.shape[1], 1)),epochs=40, verbose=1)

                end = time.time()
                        
                trainingTime.append(end-start)

                modelName = "NLPModelWith1024Unit.pkl"
                        
                with open(modelName, 'wb') as model:
                        pickle.dump(NLPModel1024,model)

                #----------------------------------------------------------------

                #Plot loss for all models
                plt.title("Loss rate for NLP models")
                plt.plot(history256.history['loss']) 
                plt.plot(history512.history['loss']) 
                plt.plot(history1024.history['loss'])
                plt.legend(['256','512', '1024']) 
                plt.show()

                #Plot validation loss for all models
                plt.title("Validation loss rate for NLP models")
                plt.plot(history256.history['val_loss']) 
                plt.plot(history512.history['val_loss']) 
                plt.plot(history1024.history['val_loss'])
                plt.legend(['256','512', '1024']) 
                plt.show()


                #Plot time taken to train for each model
                objects = ('256', '512', '1024')
                y_pos = np.arange(len(objects))
                plt.bar(y_pos, trainingTime, align='center', alpha=0.5)
                plt.xticks(y_pos, objects)
                plt.ylabel('Time (s)')
                plt.title('Time taken to train each model')
                plt.show()              

        def TestBestPerformingModel(self):
                
                print()
                print("\tTESTING BEST NATURAL LANGAUGE PROCESSING CLASSIFIER: LSTM WITH 1024 HIDDEN UNITS")
                print()

                print("\t\tReading Testing Test...")
                print()

                testingText = self.ReadFile("Task2Datasets\\testingdata.txt") #Training data
                
                #Split training data into lines
                testingData = self.SplitIntoLines(testingText) 
                testingData = array(testingData)
                
                print("\t\tProcessing Testing data...")
                print()

                # Remove punctuation 
                testingData[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in testingData[:,0]] 
                testingData[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in testingData[:,1]] 

                # convert text to lowercase 
                for i in range(len(testingData)): 

                        testingData[i,0] = testingData[i,0].lower() 
                        testingData[i,1] = testingData[i,1].lower()

                # prepare english tokenizer
                test_eng_tokenizer = self.tokenization(testingData[:, 0]) 
                test_eng_vocab_size = len(test_eng_tokenizer.word_index) + 1 
                test_eng_length = 8 

                # prepare Deutch tokenizer 
                test_deu_tokenizer = self.tokenization(testingData[:, 1]) 
                test_deu_vocab_size = len(test_deu_tokenizer.word_index) + 1 
                test_deu_length = 8

                # prepare  data 
                testX = self.EncodeSquences(test_deu_tokenizer, test_deu_length, testingData[:, 1]) 
                testY = self.EncodeSquences(test_eng_tokenizer, test_eng_length, testingData[:, 0])

                # Loading best performing model
                modelName = "NLPModelWith1024Unit.pkl"
                
                with open(modelName, 'rb') as model:  
                        pickle_model = pickle.load(model)

                #Predicting using model
                preds = pickle_model.predict_classes(testX.reshape((testX.shape[0], testX.shape[1])))

                #Turn back to text
                preds_text = [] 
                for i in preds:
                        temp = []        
                        for j in range(len(i)):             
                                t = self.get_word(i[j], test_eng_tokenizer)             
                                
                                if j > 0:                 
                                        if (t==self.get_word(i[j-1], test_eng_tokenizer))or(t== None):                       
                                                temp.append('')                 
                                        else:                      
                                                temp.append(t)             
                                else:                    
                                        if(t == None):                                   
                                                        
                                                temp.append('')                    
                                        else:   

                                                temp.append(t)        
                                
                        preds_text.append(' '.join(temp))

                #Calculating BLEU Score
                bleuScores = []

                testingEng = testingData[:,0]

                smoothing = SmoothingFunction()

                print("\tCalculating average BLEU score for predictions...")
                print()

                for i in range(0, 499):

                        bleuScores.append(sentence_bleu(testingEng[i],preds_text[i], smoothing_function=smoothing.method1))
                
                print("\tAverage BLEU score for predicted dataset: {}".format(round(statistics.mean(bleuScores)),4))
                
                

        #Main function
        def Main(self):

                isLoop1 = True

                while isLoop1:

                        print()
                        print("\tPlease select operation")
                        print()
                        print("\t\t1. Train Seq2Seq models")
                        print("\t\t2. Test best performing model")
                        print("\t\t3. EXIT")   
                        print()
                        selc = int(input("\tSelection: ")) 

                        #Train classifiers
                        if selc == 1:

                                self.TrainingSeq2Seq()

                        elif selc == 2:

                                self.TestBestPerformingModel()

                        elif selc == 3:

                                isLoop1 = False

                        else:
                                pass

class ProgramStart:

        def Title(self):
                
                print()
                print("\t __  __ _        __  __  ____  _____  ______ _        _______ _____            _____ _   _ _____ ______ _____")  
                print("\t|  \\/  | |      |  \\/  |/ __ \\|  __ \\|  ____| |      |__   __|  __ \     /\\   |_   _| \ | |_   _|  ____|  __ \\") 
                print("\t| \  / | |      | \  / | |  | | |  | | |__  | |         | |  | |__) |   /  \    | | |  \| | | | | |__  | |__) |")
                print("\t| |\\/| | |      | |\\/| | |  | | |  | |  __| | |         | |  |  _  /   / /\\ \\   | | | . ` | | | |  __| |  _  /") 
                print("\t| |  | | |____  | |  | | |__| | |__| | |____| |____     | |  | | \ \  / ____ \\ _| |_| |\\  |_| |_| |____| | \\ \\") 
                print("\t|_|  |_|______| |_|  |_|\\____/|_____/|______|______|    |_|  |_|  \\_\\/_/    \\_\\_____|_| \\_|_____|______|_|  \\_\\")
                print()
                print("\t----------------------- Python Application To Train 3 Types Of Machine Learning Models ------------------------")
                print()

        def Menu(self):

                self.Title()

                isLoop = True

                while(isLoop):

                        print()
                        print("\tPlease select operation")
                        print()
                        print("\t\t1. Train classifiers for Digit Recognition")
                        print("\t\t2. Train classifiers for Natural Langauge Processing")
                        print("\t\t3. EXIT")   
                        print()
                        selc = int(input("\tSelection: "))

                        if selc == 1:

                                DC = Digit_Classifier()
                                DC.Main()

                        elif selc == 2:

                                MT = MachineTranslation()
                                MT.Main()
                                
                        elif selc == 3:

                                isLoop = False


#Initiates class and calls main function
os.system("cls")
print()

tensorflow_shutup()

Main = ProgramStart()
Main.Menu()