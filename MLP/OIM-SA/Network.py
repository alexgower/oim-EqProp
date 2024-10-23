import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import torch

class Network(nn.Module):

    def __init__(self, args):

        super(Network, self).__init__()

        with torch.no_grad():
            N_inputs, N_hidden, N_output =  args.layersList[0], args.layersList[1], args.layersList[2]

            # Initialise weights with this distribution
            self.weights_0 = 2*(np.random.rand(N_inputs, N_hidden)-0.5)*math.sqrt(1/N_inputs)
            self.weights_1 = 2*(np.random.rand(N_hidden, N_output)-0.5)*math.sqrt(1/N_hidden)

            # Multiply weights by gain
            self.weights_0 = args.gain_weight0 * self.weights_0
            self.weights_1 = args.gain_weight1 * self.weights_1

            # Initialise biases to zero
            self.bias_0 = np.zeros(N_hidden)
            self.bias_1 = np.zeros(N_output)





    def computeLossAcc(self, seq, target, args):
        '''
        compute the loss and the error from target and seq (for any kind of sampler)
        '''
        # Note that loss is the sum of the squared differences between the target and the output neuron activations
        # Note that pred is the number of correct predictions (i.e. the number of times the predicted i.e. most activated class is the same as the target class)


        # Note seq[1] is the output layer of the network for the free phase (s[0] is the hidden layer)

        # Note target and seq contain MANY samples

        with torch.no_grad():
            # Get number of output neurons per class
            expand_output = int(args.layersList[2]/10)

            #sanity check
            assert seq[1].shape == target.shape

            # Compute loss # TODO think if we need to change this for continuous loss function
        
            loss = (((target-seq[1])**2).sum()/2).item()


            # Get the predicted average of the output neurons for each class and stack them
            pred_ave   = np.stack([item.sum(1) for item in np.split(seq[1], int(args.layersList[2]/expand_output), axis = 1)], 1)/expand_output
            # Get the target average of the output neurons for each class and stack them
            target_red = np.stack([item.sum(1) for item in np.split(target, int(args.layersList[2]/expand_output), axis = 1)], 1)/expand_output

            assert pred_ave.shape == target_red.shape

            # Compute the number of correct predictions 
            # (i.e. the number of times the predicted class with maximum average activation over multiple output neurons per class is the same as the target class)
            pred = ((np.argmax(target_red, axis = 1) == np.argmax(pred_ave, axis = 1))*1).sum()

        return loss, pred





    def computeGrads(self, data, s, seq, args):

        # s = nudged output
        # seq = free output

        with torch.no_grad():
            coef = args.beta*args.batch_size # TODO why are we multiplying by batch_size?
            
            gradsW, gradsB = [], []

            # Compute gradients for weights between hidden and output layer
            gradsW.append(-(np.matmul(s[0].T, s[1]) - np.matmul(seq[0].T, seq[1])) /coef)
            # Compute gradients for weights between input and hidden layer
            gradsW.append(-(np.matmul(data.numpy().T, s[0]) - np.matmul(data.numpy().T, seq[0])) /coef)

            # Compute gradients for biases in output layer
            gradsB.append(-(s[1] - seq[1]).sum(0) /coef)
            # Compute gradients for biases in hidden layer
            gradsB.append(-(s[0] - seq[0]).sum(0) /coef)

            # TODO figure out why we use .sum(0) for biases but not weights

            return gradsW, gradsB




    def updateParams(self, data, s, seq, args):
        '''
        Update the weights and biases of the network using the gradients computed from the data and the nudged and free outputs and the learning rates
        '''

        with torch.no_grad():
            ## Compute gradients and update weights from simulated sampling
            gradsW, gradsB = self.computeGrads(data, s, seq, args)


            # Update weights and biases with their individual learning rates
            # Clip them to be between -1 and 1 # TODO see what happens if we change this

            #weights
            assert self.weights_1.shape == gradsW[0].shape
            self.weights_1 += args.lrW0 * gradsW[0]
            self.weights_1 = self.weights_1.clip(-1,1)

            assert self.weights_0.shape == gradsW[1].shape
            self.weights_0 += args.lrW1 * gradsW[1]
            self.weights_0 = self.weights_0.clip(-1,1)

            #biases
            assert self.bias_1.shape == gradsB[0].shape
            self.bias_1 += args.lrB0 * gradsB[0]
            self.bias_1 = self.bias_1.clip(-1,1)

            assert self.bias_0.shape == gradsB[1].shape
            self.bias_0 += args.lrB1 * gradsB[1]
            self.bias_0 = self.bias_0.clip(-1,1)

            del gradsW, gradsB

