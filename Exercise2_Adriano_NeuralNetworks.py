# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:11:55 2021

@author: apmel
"""
import numpy as np
import neurolab as nl


#Features data:
data_set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)
#print(data_set1)
data_set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
#print(data_set2)
#print(type(data_set1))

input_Adriano = np.concatenate((data_set1, data_set2), axis=1)
input_Adriano.shape

#print(input_Adriano)

#output layer:
output_Adriano = data_set1 + data_set2
#print(output_Adriano)

type(output_Adriano)

#Setting the seed
np.random.seed(1)

#Define a multi-layer neutal network with two hidden layers.
#Fisrt layer with 5 neurons and second layer with 3 neurons 
#1 output
nn = nl.net.newff([[-0.6, 0.6], [-.6, 0.6]], [5, 3, 1])

#Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

#Train the neural network
error_progress = nn.train(input_Adriano, output_Adriano, epochs=1000, show=100, goal=0.00001)
#print(error_progress)

#Testing the dataset
testing_data1 = nn.sim([[0.1, 0.2]])
print(testing_data1)












