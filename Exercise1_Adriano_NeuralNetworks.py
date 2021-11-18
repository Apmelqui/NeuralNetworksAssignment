# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:05:43 2021

@author: apmel
"""

import numpy as np
import neurolab as nl

#Generating the dataset
data_set1 = np.random.uniform(-0.6, 0.6, 10).reshape(10,1)
#print(data_set1)
data_set2 = np.random.uniform(-0.6, 0.6, 10).reshape(10, 1)
#print(data_set2)
print(type(data_set1))
data_set1.shape

#Concatenating the dataset into one
input_Adriano = np.concatenate((data_set1, data_set2), axis=1)
input_Adriano.shape

#print(input_Adriano)

output_Adriano = data_set1 + data_set2
output_Adriano.shape
#print(output_Adriano)

#Setting the seed
np.random.seed(1)

#Defining min and max values (PS> it is not necessary in this exercise since it was already given (-0.6 to 0.6))
#Minimum and Maximum values for each dimention:
data_set1_min = data_set1[:].min()
#print(dim1_min)   
data_set1_max = data_set1[:].max()
#print(dim1_max) 
data_set2_min = data_set2[:].min()
data_set2_max = data_set2[:].max()

#Define a single-layer neutal network with 6 neurons and 1 output
set1 = [data_set1_min, data_set1_max]
set2 = [data_set2_min, data_set2_max]
nn = nl.net.newff([set1, set2], [6, 1])

#Train the neural network
error_progress = nn.train(input_Adriano, output_Adriano, show=15, goal=0.00001)
#print(error_progress)

#Testing the data set
testing_data1 = nn.sim([[0.1, 0.2]])
print(testing_data1)


