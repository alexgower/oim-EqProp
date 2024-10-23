import matplotlib.pyplot as plt
import numpy as np
import argparse # For parsing arguments from command line
from math import*
from tqdm import tqdm as progressbars # For progress bar
import os # For file management
from random import*

from Tools import*
from Network import*


# Used to compare to Laydevant D-Wave based Simulated Annealing
from simulated_sampler import SimulatedAnnealingSampler 


import torch


# Parse arguments from command line
parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation with D-Wave support')
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='Dataset we use for training (default=mnist, others: digits)')
parser.add_argument(
    '--simulated',
    type=int,
    default=1,
    help='specify if we use simulated annealing (=1) or OIM annealing (=0) (default=1, else = 0)') # TODO implement our own too?
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    # default=[784, 10, 10],
    default=[784, 120, 40],
    help='List of layer sizes (default: [784, 120, 10])') # TODO change back for OIM
parser.add_argument(
    '--n_iter_free',
    type=int,
    default=10,
    help='Times to iterate for the OIM on a single data point for getting the minimal energy state of the free phase (default=10)')
parser.add_argument(
    '--n_iter_nudge',
    type=int,
    default=10, 
    help='Times to iterate for the OIM on a single data point for getting the minimal energy state of the nudge phase (default=10)')
parser.add_argument(
    '--Ks_max_nudge',
    type=float,
    default=0.25,
    help='Maximum value of the Ks for the nudge phase (default=0.25)') # TODO implement
parser.add_argument(
    '--N_data_train',
    type=int,
    # default=10,
    default=1000,
    help='Number of data points for training (default=1000)') 
parser.add_argument(
    '--N_data_test',
    type=int,
    # default=2,
    default=100, 
    help='Number of data points for testing (default=100)')
parser.add_argument(
    '--beta',
    type=float,
    default=5,
    help='Beta - hyperparameter of EP (default=5)')
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Size of mini-batches we use (for training only) (default=1)')
parser.add_argument(
    '--lrW0',
    type=float,
    default=0.01,
    help='Learning rate for weights - input-hidden  (default=0.01)')
parser.add_argument(
    '--lrW1',
    type=float,
    default=0.01,
    help='Learning rate for weights - hidden-output (default=0.01)')
parser.add_argument(
    '--lrB0',
    type=float,
    default=0.001,
    help='Learning rate for biases - hidden (default=0.001)')
parser.add_argument(
    '--lrB1',
    type=float,
    default=0.001,
    help='Learning rate for biases - output (default=0.001)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10, # TODO was 50 before
    help='Number of epochs (default=10)')
parser.add_argument(
    '--load_model',
    type=int,
    default=0,
    help='If we load the parameters from a previously trained model to continue the training (default=0, else = 1)')
parser.add_argument(
    '--gain_weight0',
    type=float,
    default=0.5,
    help='Gain for initialization of the weights - input-hidden (default=1)')
parser.add_argument(
    '--gain_weight1',
    type=float,
    default=0.25,
    help='Gain for initialization of the weights  - hidden-output (default=1)')
parser.add_argument( # TODO remove or at least use consistently everywhere unlike now??
    '--bias_lim',
    type=float,
    default=4.0,
    help='Max limit for the amplitude of the local biases applied to the qbits, either for free or nudge phase - (default=1)')
parser.add_argument(
    '--data_augmentation',
    type=int,
    default=0,
    help='Set data augmentation or not for the problems - (default=False)')




parser.add_argument( # TODO remove once replaced simulated annealing
    '--frac_anneal_nudge',
    type=float,
    default=0.25,
    help='fraction of system non-annealed (default=0.5, if <0.5: more annealing, if > 0.5, less annealing)')

args = parser.parse_args()






with torch.no_grad():

    ## SAMPLERS
    if args.simulated == 1:
        sampler = SimulatedAnnealingSampler() 
    else:
        # TODO change to OIM
        sampler = SimulatedAnnealingSampler() 




    ## Files saving: create a folder for the simulation and save simulation's parameters
    BASE_PATH = createPath(args) # Create path to S-i folder
    dataframe = initDataframe(BASE_PATH) # Create dataframe to store results in results.csv
    print(BASE_PATH)


    ## Generate DATA
    if args.dataset == "digits":
        train_loader, test_loader = generate_digits(args)
    elif args.dataset == "mnist":
        train_loader, test_loader, dataset = generate_mnist(args)


    ## Create the network
    if args.load_model == 0:
        net = Network(args)
        saveHyperparameters(BASE_PATH, args)
    else:
        net = load_model_numpy(BASE_PATH)






    ## Monitor loss and prediction error
    train_loss_tab, train_error_tab = [], []
    test_loss_tab, test_error_tab = [], []

    for epoch in progressbars(range(args.epochs)):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train the network
        loss, error = train(net, args, train_loader, sampler)
        train_loss_tab.append(loss)
        train_error_tab.append(error)

        # Test the network
        loss, error = test(net, args, test_loader, sampler)
        test_loss_tab.append(loss)
        test_error_tab.append(error)

        # Store error and loss at each epoch
        dataframe = updateDataframe(BASE_PATH, dataframe, np.array(train_error_tab)[-1]/len(train_loader.dataset)*100, np.array(test_error_tab)[-1]/len(test_loader.dataset)*100, train_loss_tab, test_loss_tab)

        save_model_numpy(BASE_PATH, net)

