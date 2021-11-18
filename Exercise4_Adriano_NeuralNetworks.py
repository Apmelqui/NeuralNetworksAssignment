# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:20:48 2021

@author: apmel
"""

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt


#Generating the dataset
data_set1 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)
data_set2 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)

#Concatenating the dataset into one
input_Adriano = np.concatenate((data_set1, data_set2), axis=1)
#print(input_Adriano)

output_Adriano = data_set1 + data_set2
#print(output_Adriano)

#Setting the seed
np.random.seed(1)

#Creating the neural network with:
#2 input neurons
#two hidden layers: first with 5 neurons and second with 3 neurons
#1 output neuron
nn = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6]], [5, 3, 1])

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
testing_dataset = nn.sim([[0.1, 0.2]])
print(testing_dataset)






