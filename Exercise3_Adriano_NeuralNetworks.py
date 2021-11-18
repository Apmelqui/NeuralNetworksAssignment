# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:57:27 2021

@author: apmel
"""
import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt


data_set1 = np.random.uniform(-0.6, 0.6, 100).reshape(100,1)
#print(data_set1)
data_set2 = np.random.uniform(-0.6, 0.6, 100).reshape(100, 1)
#print(data_set2)
print(type(data_set1))

input_Adriano = np.concatenate((data_set1, data_set2), axis=1)
input_Adriano.shape

#print(input_Adriano)

#Feature - input data
output_Adriano = data_set1 + data_set2
#print(output_Adriano)

type(output_Adriano)

#Setting the seed
np.random.seed(1)

#Defining min and max values (PS> it is not necessary in this exercise since it was already given (-0.6 to 0.6))
#Minimum and Maximum values for each dimention:
data_set1_min = data_set1[:,0].min()
#print(dim1_min)   
data_set1_max = data_set1[:,0].max()
#print(dim1_max) 
data_set2_min = data_set2[:,0].min()
data_set2_max = data_set2[:,0].max()

#Define a single-layer neutal network with 6 neurons and 1 output
set1 = [data_set1_min, data_set1_max]
set2 = [data_set2_min, data_set2_max]
nn = nl.net.newff([set1, set2], [6, 1])

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
testing_data = nn.sim([[0.1, 0.2]])
print(testing_data)
