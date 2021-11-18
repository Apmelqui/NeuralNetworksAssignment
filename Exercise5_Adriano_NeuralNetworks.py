# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:10:23 2021

@author: apmel
"""

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

#Generating the dataset
data_set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)
data_set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
data_set3 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)


#Concatenating the dataset into one
input_Adriano = np.concatenate((data_set1, data_set2, data_set3), axis=1)
input_Adriano.shape

#print(input_Adriano)

output_Adriano = data_set1 + data_set2 + data_set3
#print(output_Adriano)

type(output_Adriano)

#Setting the seed
np.random.seed(1)


nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]], [6, 1])


#Train the neural network
error_progress = nn.train(input_Adriano, output_Adriano, show=15, goal=0.00001)
#print(error_progress)

#Plotting the trainning error
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error process')

#Testing the data set
testing_data = nn.sim([[0.2, 0.1, 0.2]])
print(testing_data)


############################################################
print('###########################################################')


import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

#Generating the dataset
data_set4 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)
data_set5 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)
data_set6 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)

#Concatenating the dataset into one
input_Adriano = np.concatenate((data_set4, data_set5, data_set6), axis=1)
#print(input_Adriano)

output_Adriano = data_set4 + data_set5 + data_set6
#print(output_Adriano)

#Setting the seed
np.random.seed(1)

#Creating the neural network with:
#2 input neurons
#two hidden layers: first with 5 neurons and second with 3 neurons
#1 output neuron
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]], [5, 3, 1])

#Set the trainning algorithm to gradient descent
nn.trainf = nl.train.train_gd

#Train the neural network
error_progress = nn.train(input_Adriano, output_Adriano, epochs=1000, show=100, goal=0.00001)

#Plotting the trainning error
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error process')

#Testing the dataset:
testing_dataset = nn.sim([[0.2, 0.1, 0.2]])
print(testing_dataset)















