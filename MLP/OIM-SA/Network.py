import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import torch

class Network(nn.Module):

    def __init__(self, args):

        super(Network, self).__init__()

        with torch.no_grad(): # TODO maybe optimise this initialisation at some point
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


        # Note seq[1] is the output layer of the network for the free phase (s[0] is the hidden layer, and there is no input layer as it is always fixed)

        # Note target and seq contain MANY samples

        with torch.no_grad():
            # Get number of output neurons per class
            expand_output = int(args.layersList[2]/10)

            #sanity check
            assert seq[1].shape == target.shape

            # Compute loss 
            # Note target = [-1,+1,-1,-1,-1,-1,-1,-1,-1,-1] if target is 1 etc
            # Note this just implements a simple mean squared error loss between the output neurons between +-1 and the onehot encoded target using +1 and -1
            # This used identically for rounded and non-rounded outputs
            # TODO maybe implement a cross entropy loss here soon
        
            loss = (((target-seq[1])**2).sum()/2).item()


            # Get the predicted AVERAGE of the output neurons for each CLASS and stack them
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
            coef = args.beta*args.batch_size # Note we must divide by the batch size to get the average gradient over the batch
            
            gradsW, gradsB = [], []

            if args.rounding == 1: 

                # Compute gradients for weights between hidden and output layer
                gradsW.append(-(np.matmul(s[0].T, s[1]) - np.matmul(seq[0].T, seq[1])) /coef)
                # Compute gradients for weights between input and hidden layer
                gradsW.append(-(np.matmul(data.numpy().T, s[0]) - np.matmul(data.numpy().T, seq[0])) /coef)

                # Compute gradients for biases in output layer
                gradsB.append(-(s[1] - seq[1]).sum(0) /coef)
                # Compute gradients for biases in hidden layer
                gradsB.append(-(s[0] - seq[0]).sum(0) /coef)

                # Note that we sum over the batch dimension to get the average gradient over the batch
                # This is done automatically in the matrix multiplication for the weights but not for the biases hence the .sum(0) for biases
        

            elif args.rounding == 0:

                # Note this assumes K=1 in the OIM (else need xK factor in the gradients)
                # Note this also assumes s = cos(\phi) from OIMs
                # TODO change this when if we change cos(\phi) to tanh etc

                # For s = cos(arccos(s)), sin(arccos(s)) = sqrt(1-s^2)            
                # Compute sin terms
                sin_s0 = np.sqrt(1 - np.square(s[0]))
                sin_s1 = np.sqrt(1 - np.square(s[1]))
                sin_seq0 = np.sqrt(1 - np.square(seq[0]))
                sin_seq1 = np.sqrt(1 - np.square(seq[1]))
                sin_data = np.sqrt(1 - np.square(data.numpy()))

                # Hidden-output weights: 
                # cos(x-y) = cos(x)cos(y) + sin(x)sin(y)
                grad_W1 = -(np.matmul(s[0].T, s[1]) + np.matmul(sin_s0.T, sin_s1) - np.matmul(seq[0].T, seq[1]) - np.matmul(sin_seq0.T, sin_seq1)) / coef
                gradsW.append(grad_W1)

                # Input-hidden weights
                grad_W0 = -(np.matmul(data.numpy().T, s[0]) + np.matmul(sin_data.T, sin_s0) -np.matmul(data.numpy().T, seq[0]) - np.matmul(sin_data.T, sin_seq0)) / coef
                gradsW.append(grad_W0)

                # Biases with corrected signs to match discrete case
                gradsB.append(-(s[1] - seq[1]).sum(0) / coef)  # Output layer biases
                gradsB.append(-(s[0] - seq[0]).sum(0) / coef)  # Hidden layer biases

            return gradsW, gradsB




    def updateParams(self, data, s, seq, args):
        '''
        Update the weights and biases of the network using the gradients computed from the data and the nudged and free outputs and the learning rates
        '''

        with torch.no_grad():
            ## Compute gradients and update weights from simulated sampling
            gradsW, gradsB = self.computeGrads(data, s, seq, args)


            # Update weights and biases with their individual learning rates
            # Clip them to be between -J_clip and J_clip

            #weights
            assert self.weights_1.shape == gradsW[0].shape
            self.weights_1 += args.lrW0 * gradsW[0]
            self.weights_1 = self.weights_1.clip(-args.J_clip,args.J_clip)

            assert self.weights_0.shape == gradsW[1].shape
            self.weights_0 += args.lrW1 * gradsW[1]
            self.weights_0 = self.weights_0.clip(-args.J_clip,args.J_clip)

            #biases
            assert self.bias_1.shape == gradsB[0].shape
            self.bias_1 += args.lrB0 * gradsB[0]
            self.bias_1 = self.bias_1.clip(-args.J_clip,args.J_clip)

            assert self.bias_0.shape == gradsB[1].shape
            self.bias_0 += args.lrB1 * gradsB[1]
            self.bias_0 = self.bias_0.clip(-args.J_clip,args.J_clip)

            del gradsW, gradsB

