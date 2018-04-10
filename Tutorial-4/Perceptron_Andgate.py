# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:54:28 2018

@author: Jatin
"""

# and gate using perceptron
import numpy as np
import random
import sys
and_gate = [
    [(1, 1), 1],
    [(1, -1), -1],
    [(-1, 1), -1],
    [(-1, -1), -1]
]
def activation_function(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1

def run_perceptron(gate):
    bias = (1,) # the bias is always one
    learning_constant = 0.1
    n = 50 # how many times the machine learns

    weights = []

    # initialize with 3 random weights between -1 and 1, one for each input and one for the bias
    for i in range(3):
        weights.append(random.uniform(-1, 1))

    for i in range(n):
        inputs, expected_output = random.choice(gate)
        inputs = inputs + bias # add the bias here
        weighted_sum = np.dot(inputs, weights)
        guess = activation_function(weighted_sum) # find the sign of the weighted sum
        error = expected_output - guess
        weights += learning_constant * error * np.asarray(inputs) 

    for i in range(5):
        inputs, expected_output = random.choice(gate)
        inputs = inputs + bias
        w=np.dot(inputs,weights)
        
        print("inputs: "+str(inputs))
        print ("weights: "+str(w))
        print ("expected output: " +str(expected_output))
        print ("perceptron output: " + str(activation_function(w)) +'\n')   

if __name__=="__main__":
    run_perceptron(and_gate)
    
