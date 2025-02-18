import torch.nn as nn
import numpy as np
from numpy import cos  # Import cos from numpy for vectorization
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import torch

class Network(nn.Module):

    def __init__(self, args):

        super(Network, self).__init__()

        with torch.no_grad(): 
            # TODO make more flexible for multiple hidden layers
            N_inputs, N_hidden, N_output =  args.layersList[0], args.layersList[1], args.layersList[2]

            # Initialise weights with this distribution
            # TODO maybe optimise this initialisation at some point
            self.weights_0 = 2*(np.random.rand(N_inputs, N_hidden)-0.5)*math.sqrt(1/N_inputs)
            self.weights_1 = 2*(np.random.rand(N_hidden, N_output)-0.5)*math.sqrt(1/N_hidden)

            # Multiply weights by gain
            self.weights_0 = args.gain_weight0 * self.weights_0
            self.weights_1 = args.gain_weight1 * self.weights_1

            # Initialise biases to zero
            self.bias_0 = np.zeros(N_hidden)
            self.bias_1 = np.zeros(N_output)

            # Initialise syncs to zero
            self.sync_0 = np.zeros(N_hidden)
            self.sync_1 = np.zeros(N_output)





    def computeLossAcc(self, s, target, args):
        '''
        compute the loss and the error from target and seq (for any kind of sampler)
        '''
        # Note that loss is the sum of the squared differences between the target and the output neuron activations
        # Note that pred is the number of correct predictions (i.e. the number of times the predicted i.e. most activated class is the same as the target class)


        # Note s[1] is the output layer of the network for the free phase (s[0] is the hidden layer, and there is no input layer as it is always fixed)

        # Note target and s contain MANY samples

        with torch.no_grad():
            # Get number of output neurons per class
            expand_output = int(args.layersList[2]/10)

            #sanity check
            assert s[1].shape == target.shape

            # Convert phases to cosine values before computing loss and accuracy
            s_cos = np.cos(s[1])

            # Compute loss 
            # Note target = [-1,+1,-1,-1,-1,-1,-1,-1,-1,-1] if target is 1 etc
            # Note this just implements a simple mean squared error loss between the output neurons between +-1 and the onehot encoded target using +1 and -1
            # This used identically for rounded and non-rounded outputs
            loss = (((target-s_cos)**2).sum()/2).item()

            # Get the predicted AVERAGE of the output neurons for each CLASS and stack them
            # Note int(args.layersList[2]/expand_output) should always equal 10
            pred_ave   = np.stack([item.sum(1) for item in np.split(s_cos, int(args.layersList[2]/expand_output), axis = 1)], 1)/expand_output
            # Get the target average of the output neurons for each class and stack them
            target_avg = np.stack([item.sum(1) for item in np.split(target, int(args.layersList[2]/expand_output), axis = 1)], 1)/expand_output

            assert pred_ave.shape == target_avg.shape

            # Compute the number of correct predictions 
            # (i.e. the number of times the predicted class with maximum average activation over multiple output neurons per class is the same as the target class)
            pred = ((np.argmax(target_avg, axis = 1) == np.argmax(pred_ave, axis = 1))*1).sum()

        return loss, pred





    def computeGrads(self, data, s_pos, s_neg, args):

        # Pass to computeExactGrads if exact_grads is set to 1
        if args.exact_grads == 1:
            return computeExactGrads(self, data, s_pos, s_neg, args)

        # s_pos = positive beta nudged sample
        # s_neg = negative beta nudged sample

        with torch.no_grad():
            # Note we must divide by the batch size to get the average gradient over the batch
            # Note also we use factor of 2 becuase we are using symmetric +beta and -beta average
            coef = 2*args.beta*args.batch_size 
            
            gradsW, gradsB, gradsSYNC = [], [], []


            if args.simulation_type == 0: # i.e. for continuous OIM case

                # Note this also assumes s = cos(\phi) from OIMs
                # Note that grad_B and grad_SYNC use .sum(0) to sum over the batch dimension
                # but grad_W uses .T to get the transpose of the input matrix so does automatically 

                ### HIDDEN LAYER LEARNABLE PARAMETERS
                # Note for hidden layer weights since we use input matrix trick:
                # If input s_i = cos(\phi_i) then implicitly we are assuming J_{ij} cos(\phi_i) cos(\phi_j)
                # Therefore we must account for this in learning rule for hidden layer weights
                grad_W0 = -(-np.matmul(data.numpy().T, np.cos(s_pos[0])) + np.matmul(data.numpy().T, np.cos(s_neg[0]))) / coef
                gradsW.append(grad_W0)

                grad_B0 = -(-np.cos(s_pos[0]) + np.cos(s_neg[0])).sum(0) / coef
                gradsB.append(grad_B0)
                
                grad_SYNC0 = -0.5 * ( -np.cos(2*s_pos[0]) + np.cos(2*s_neg[0])).sum(0) / coef
                gradsSYNC.append(grad_SYNC0)



                ### OUTPUT LAYER LEARNABLE PARAMETERS
                # Note we need to reshape s_pos[1] and s_neg[1] to be column vectors and s_pos[0] and s_neg[0] to be row vectors to use broadcasting 
                # to get all combinations of s_pos[1] and s_pos[0] and s_neg[1] and s_neg[0]
                # Note we use .T to get the transpose of the input matrix so does automatically
                grad_W1 = -(-np.cos(s_pos[1][..., None] - s_pos[0][..., None, :]) + 
                           np.cos(s_neg[1][..., None] - s_neg[0][..., None, :])).sum(0).T / coef
                gradsW.append(grad_W1)

                grad_B1 = -(-np.cos(s_pos[1]) + np.cos(s_neg[1])).sum(0) / coef
                gradsB.append(grad_B1)
                
                grad_SYNC1 = -0.5 * (-np.cos(2*s_pos[1]) + np.cos(2*s_neg[1])).sum(0) / coef
                gradsSYNC.append(grad_SYNC1)


            elif args.simulation_type == 1: # i.e. for Scellier case

                # TODO do this
                pass



            if args.debug:
                # Print relative changes and ranges for each parameter type and layer
                print("\nParameter Update Analysis:")
            
                # Hidden layer (layer 0)
                print("\nHidden Layer:")
                print("Weights:")
                rel_change_w0 = (args.lrW0 * gradsW[0]) / (self.weights_0 + 1e-10)  # Add small epsilon to avoid div by 0
                print(f"  Relative change (min/max): {np.min(rel_change_w0):.6f} / {np.max(rel_change_w0):.6f}")
                print(f"  Parameter range (min/max): {np.min(self.weights_0):.6f} / {np.max(self.weights_0):.6f}")
                # print(" All relative changes:")
                # for row in rel_change_w0:
                #     print(' '.join(f'{x:8.6f}' for x in row))
                
                print("Biases:")
                rel_change_b0 = (args.lrB0 * gradsB[0]) / (self.bias_0 + 1e-10)
                print(f"  Relative change (min/max): {np.min(rel_change_b0):.6f} / {np.max(rel_change_b0):.6f}")
                print(f"  Parameter range (min/max): {np.min(self.bias_0):.6f} / {np.max(self.bias_0):.6f}")
                
                print("Sync terms:")
                rel_change_s0 = (args.lrSYNC0 * gradsSYNC[0]) / (self.sync_0 + 1e-10)
                print(f"  Relative change (min/max): {np.min(rel_change_s0):.6f} / {np.max(rel_change_s0):.6f}")
                print(f"  Parameter range (min/max): {np.min(self.sync_0):.6f} / {np.max(self.sync_0):.6f}")
                
                # Output layer (layer 1) 
                print("\nOutput Layer:")
                print("Weights:")
                rel_change_w1 = (args.lrW1 * gradsW[1]) / (self.weights_1 + 1e-10)
                print(f"  Relative change (min/max): {np.min(rel_change_w1):.6f} / {np.max(rel_change_w1):.6f}")
                print(f"  Parameter range (min/max): {np.min(self.weights_1):.6f} / {np.max(self.weights_1):.6f}")
                
                print("Biases:")
                rel_change_b1 = (args.lrB1 * gradsB[1]) / (self.bias_1 + 1e-10)
                print(f"  Relative change (min/max): {np.min(rel_change_b1):.6f} / {np.max(rel_change_b1):.6f}")
                print(f"  Parameter range (min/max): {np.min(self.bias_1):.6f} / {np.max(self.bias_1):.6f}")
                
                print("Sync terms:")
                rel_change_s1 = (args.lrSYNC1 * gradsSYNC[1]) / (self.sync_1 + 1e-10)
                print(f"  Relative change (min/max): {np.min(rel_change_s1):.6f} / {np.max(rel_change_s1):.6f}")
                print(f"  Parameter range (min/max): {np.min(self.sync_1):.6f} / {np.max(self.sync_1):.6f}")

            return gradsW, gradsB, gradsSYNC
        

    def computeExactGrads(self, data, s, seq, args):
        # TODO do this code
        return [0], [0] 




    def updateParams(self, data, s_pos, s_neg, args):
        '''
        Update the weights and biases and syncs of the network using the gradients computed from the data and the nudged outputs and the learning rates.
        Indexing convention:
        - Index 0 corresponds to layer 0 (hidden layer)
        - Index 1 corresponds to layer 1 (output layer)
        '''

        with torch.no_grad():
            ## Compute gradients and update weights from simulated sampling
            gradsW, gradsB, gradsSYNC = self.computeGrads(data, s_pos, s_neg, args)

            # Update weights and biases with their individual learning rates and clip them

            # weights_0 (hidden layer) uses gradsW[0] and lrW0
            assert self.weights_0.shape == gradsW[0].shape
            self.weights_0 += args.lrW0 * gradsW[0]
            self.weights_0 = np.clip(self.weights_0, -args.J_clip, args.J_clip)

            # weights_1 (output layer) uses gradsW[1] and lrW1
            assert self.weights_1.shape == gradsW[1].shape
            self.weights_1 += args.lrW1 * gradsW[1]
            self.weights_1 = np.clip(self.weights_1, -args.J_clip, args.J_clip)

            # bias_0 (hidden layer) uses gradsB[0] and lrB0
            assert self.bias_0.shape == gradsB[0].shape
            self.bias_0 += args.lrB0 * gradsB[0]
            self.bias_0 = np.clip(self.bias_0, -args.h_clip, args.h_clip)

            # bias_1 (output layer) uses gradsB[1] and lrB1
            assert self.bias_1.shape == gradsB[1].shape
            self.bias_1 += args.lrB1 * gradsB[1]
            self.bias_1 = np.clip(self.bias_1, -args.h_clip, args.h_clip)

            # syncs_0 (hidden layer) uses gradsSYNC[0] and lrSYNC0
            assert self.sync_0.shape == gradsSYNC[0].shape
            self.sync_0 += args.lrSYNC0 * gradsSYNC[0]
            self.sync_0 = np.clip(self.sync_0, -args.sync_clip, args.sync_clip)

            # syncs_1 (output layer) uses gradsSYNC[1] and lrSYNC1
            assert self.sync_1.shape == gradsSYNC[1].shape
            self.sync_1 += args.lrSYNC1 * gradsSYNC[1]
            self.sync_1 = np.clip(self.sync_1, -args.sync_clip, args.sync_clip)

            del gradsW, gradsB, gradsSYNC

