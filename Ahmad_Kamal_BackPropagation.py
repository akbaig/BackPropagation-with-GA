#Assignment of Ahmad Kamal Baig (BS CIS 18-22) Implementation of Neural Networks using Backpropagation

import numpy as np
import random

inputs = np.array([[0.4, -0.7],[0.3, -0.5],[0.6, 0.1],[0.2, 0.4],[0.1, -0.2]])
outputs = np.array([[0.1],[0.05],[0.3],[0.25],[0.12]])
learning_rate = 0.1
eta = 0.5
error = 1

def SigmoidFunction(n):
    return (1/ (1 + np.exp(-n)))
    
def ApplySigmoidFunc(ray, new_ray):
    for i in range(np.shape(ray)[0]):
        for j in range(np.shape(ray)[1]):
            new_ray[i][j] = SigmoidFunction(ray[i][j])
    return new_ray
    
def CalculateHiddenInput(row):
        temp = np.matmul(weights_1.transpose(), inputs[row].transpose())
        return temp[np.newaxis].T   
    
def CalculateOutputLayerInput(weight, sig_out):
    return np.matmul(weight.transpose(), sig_out)


weights_1 = np.empty([2,2])
weights_2 = np.empty([2,1])
outputHidden = np.empty([2, 1])

for i in range(np.shape(weights_1)[0]):
    for j in range(np.shape(weights_1)[1]):
        weights_1[i][j] = random.uniform(-0.5, 0.5)

for i in range(np.shape(weights_2)[0]):
    for j in range(np.shape(weights_2)[1]):
        weights_2[i][j] = random.uniform(-0.5, 0.5)

for index in range(np.shape(inputs)[0]):
    epoch = 0
    while error > 0.000001:
        epoch += 1
        deltaW = 0.0
        deltaV = 0.0
        inputHidden = CalculateHiddenInput(index)

        outputHidden = ApplySigmoidFunc(inputHidden, outputHidden)
        inputForOutputlayer = CalculateOutputLayerInput(weights_2, outputHidden)
        calculatedValue = SigmoidFunction(inputForOutputlayer)
        
        error = np.square(outputs[index] - calculatedValue)
        
        dFactor = (outputs[index] - calculatedValue)*(calculatedValue)*(1-calculatedValue)
        Y = outputHidden * dFactor
        deltaW = (deltaW * learning_rate) + (eta * Y)
        eFactor = weights_2 * dFactor
        
        dStar = np.array([eFactor[0]* outputHidden[0] * (1 - outputHidden[0]), eFactor[1]* outputHidden[1] * (1 - outputHidden[1])])
        dStar = np.reshape(dStar,[1,2])
        
        input_temp = inputs[index]
        input_temp = input_temp[np.newaxis].T
        
        X = np.matmul(input_temp, dStar)
        deltaV = (deltaV * learning_rate) + (eta * X)
        
        weights_1 = weights_1 + deltaV
        weights_2 = weights_2 + deltaW
        
        if error <= 0.000001:
            print("Input values are ", inputs[index])
            print("Output values are ", np.around(calculatedValue,2))
    print("Number of epochs = ", epoch,"\n")
    epoch = 0
    error = 1  
    
