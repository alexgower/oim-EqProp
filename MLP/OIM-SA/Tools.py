import os
import os.path
import datetime
import shutil
from tqdm import tqdm as progressbars

import pandas as pd
import numpy as np 

import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import*
from copy import*
import pickle

import matplotlib.pyplot as plt

import dimod

# Import the Julia OIM module
from julia.api import Julia
from julia import Main

# Initialize Julia
Julia(compiled_modules=False)

# Activate the Julia environment
from julia import Pkg
Pkg.activate(".")
Pkg.instantiate()

# Include the Julia module
module_path = os.path.join("..", "oim-simulator", "code", "core", "simulations", "oim_simulations.jl")
Main.include(module_path)

# Access the module via Main
OIMSimulations = Main.OIMSimulations

# Access the function directly
oim_problem_solver = OIMSimulations.oim_problem_solver
simple_oim_dynamics = OIMSimulations.simple_oim_dynamics






########## GENERIC DATA SAVING AND MANIPULATION FUNCTIONS ##########

def createPath(args):
    '''
    Create path to save data

    File structure:
    DATA/YYYY-MM-DD/S-i
    (where YYYY-MM-DD is the date of the simulation and i is the number of the simulation of the day)

    Also move plotFunction.py to the created folder of the day
    '''

    prefix = '/'
    BASE_PATH = '' + os.getcwd()
    BASE_PATH += prefix + 'DATA'
    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")


    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)


    script_dir = os.path.dirname(__file__)
    plot_function_path = os.path.join(script_dir, 'plotFunction.py')
    filePath = shutil.copy(plot_function_path, BASE_PATH)


    files = os.listdir(BASE_PATH)

    # Also pop all files ending in .png or .py or .DS_Store
    files_to_keep = []
    for file in files:
        if not file.endswith('.png') and not file.endswith('.py') and not file.endswith('.DS_Store'):
            files_to_keep.append(file)

    files = files_to_keep

    print(files)
    # Calculate the S-i number
    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        for names in files:
            # Get the number of all previous simulations
            tab.append(int(names.split('-')[1]))
        if args.load_model == 0:
            # If we start a new simulation, increment the number of the last simulation
            BASE_PATH += prefix + 'S-' + str(max(tab)+1)
        elif args.load_model > 0:
            # If we load a model, the number of the last simulation is the same
            BASE_PATH += prefix + 'S-' + str(args.load_model)



    # If we start a new simulation, create these extra folders
    if args.load_model == 0:
        os.mkdir(BASE_PATH)

    # Get the i of the S-i folder
    name = BASE_PATH.split(prefix)[-1]

    return BASE_PATH



def initDataframe(path, dataframe_to_init = 'results.csv'):
    prefix = '/'

    if os.path.isfile(path + prefix + dataframe_to_init):
        dataframe = pd.read_csv(path + prefix + dataframe_to_init, sep = ',', index_col = 'Epoch')
    else:
        columns_header = ['Train_Acc', 'Test_Acc', 'Train_Loss', 'Test_Loss']
        
        # Create the dataframe with 'Epoch' as index
        dataframe = pd.DataFrame(columns = columns_header)
        dataframe.index.name = 'Epoch'
        dataframe.to_csv(path + prefix + 'results.csv')

    return dataframe


def updateDataframe(BASE_PATH, dataframe, train_acc, test_acc, train_loss, test_loss):
    prefix = '/'

    data = [train_acc, test_acc, train_loss[-1], test_loss[-1]]

    # Get the next epoch number
    next_epoch = len(dataframe) + 1

    new_data = pd.DataFrame([data], index=[next_epoch], columns=dataframe.columns)
    new_data.index.name = 'Epoch'


    
    dataframe = pd.concat([dataframe, new_data])
    dataframe.to_csv(BASE_PATH + prefix + 'results.csv')

    return dataframe



def saveHyperparameters(BASE_PATH, args):
    '''
    Save all hyperparameters in the path provided
    '''
    prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write("Equilibrium Propagation on OIM \n")
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__.keys():
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()


def save_model_numpy(path, net):
    '''
    Save the parameters of the model as a dictionary in a pickle file
    (Pickle is Python's built-in serialization module)
    '''

    prefix = '/'

    # We use 'wb' (write binary) to save the model so no text encoding issues
    with open(path + prefix + 'model_parameters.pickle', 'wb') as f:
            pickle.dump(net, f)


    # By convention, the function should return 0 on success
    return 0


def load_model_numpy(path):
    '''
    Load the parameters of the model from a dictionary in a pickle file
    '''
    
    prefix = '/'

    with open(path + prefix + 'model_parameters.pickle', 'rb') as f:
            net = pickle.load(f)

    return net

















########## ML DATASET GENERATION ##########

class CustomDataset(Dataset):
    # Class to create a custom dataset

    def __init__(self, images, labels=None):
        self.x = images
        self.y = labels

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        return (data, target)

    def __len__(self):
        return (len(self.x))


class ReshapeTransform:
    # Class to reshape the data
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)



class ReshapeTransformTarget:
    # Class to reshape the target labels to one-hot encoding on multiple output neurons per class if needed
    # __init__ stores the number of classes and the number of output neurons, __call__ does the one-hot encoding with this stored structure

    def __init__(self, number_classes, args):
        self.number_classes = number_classes
        self.outputlayer = args.layersList[2]

    def __call__(self, target):
        # e.g. target = 3

        target = torch.tensor(target).unsqueeze(0).unsqueeze(1)


        target_onehot = -1*torch.ones((1, self.number_classes))

        # target.long() is used to convert the target to a long (int) tensor
        # e.g. target_onehot.scatter_(1, target.long(), 1) = tensor([[0., 1., 0., 0,.0 0., 0., 0., 0., 0., 0.]]) if target = 1
        # e.g. target_onehot.scatter_(1, target.long(), 1).repeat_interleave(int(self.outputlayer/self.number_classes)) = tensor([[0.,0.,0.,0.,1.,1.,1.,1.,0.,0.,.....,0.,0.]]) if target = 1
        # i.e. we repeat 0. 4 times for class 0 and 1. 4 times for class 1, ...
        return target_onehot.scatter_(1, target.long(), 1).repeat_interleave(int(self.outputlayer/self.number_classes)).squeeze(0)



class DefineDataset(Dataset):
    '''
    Class to hold data and labels for the dataset, with the possibility to apply transformations to the data and labels too
    '''
    
    def __init__(self, images, labels=None, transforms=None, target_transforms=None):
        self.x = images
        self.y = labels
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, i):
        data = self.x[i, :]
        target = self.y[i]

        if self.transforms:
            data = self.transforms(data)

        if self.target_transforms:
            target = self.target_transforms(target)

        if self.y is not None:
            return (data, target)
        else:
            return data

    def __len__(self):
        return (len(self.x))









def generate_digits(args):
    '''
    Generate the dataloaders for digits dataset
    '''

    digits = load_digits()

    # Random_state sets reproducible seed for shuffle
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.1, random_state=10, shuffle=True)

    # TODO check if normalization needs to be changed
    normalisation = 8
    x_train, x_test = x_train / normalisation, x_test / normalisation

    # Use ReshapeTransformTarget to reshape the target labels to one-hot encoding on multiple output neurons per class if needed
    train_data = DefineDataset(x_train, labels=y_train, target_transforms=ReshapeTransformTarget(10, args))
    test_data = DefineDataset(x_test, labels=y_test, target_transforms=ReshapeTransformTarget(10, args))

    ## Data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader








def generate_mnist(args):
    '''
    Generate mnist dataloaders - 1000 training images, 100 testing images
    '''


    N_class = 10

    # Use custom training and test data size
    N_data_train = args.N_data_train
    N_data_test = args.N_data_test

    with torch.no_grad():

        # Augment data if needed, otherwise just convert to pytorch tensor and flatten to input layer
        if args.data_augmentation:
            transforms_train=[torchvision.transforms.ToTensor(), torchvision.transforms.RandomAffine(10, translate=(0.04, 0.04), scale=None, shear=None, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=0), ReshapeTransform((-1,))]
        else:
            transforms_train=[torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

        transforms_test=[torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]



        ### Training data

        # Load the MNIST dataset, apply the transformations if appropraite from above, and target transformations to multiple output neurons per class if needed
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                transform=torchvision.transforms.Compose(transforms_train),
                                                target_transform=ReshapeTransformTarget(10, args))

        # Now reduce the number of training data points to N_data_train, but keep the same number of data points per class
        mnist_train_data, mnist_train_targets, comp = torch.empty(N_data_train,28,28,dtype=mnist_train.data.dtype), torch.empty(N_data_train,dtype=mnist_train.targets.dtype), torch.zeros(N_class)
        idx_0, idx_1 = 0, 0
        while idx_1 < N_data_train:
            class_data = mnist_train.targets[idx_0]
            if comp[class_data] < int(N_data_train/N_class):
                mnist_train_data[idx_1,:,:] = mnist_train.data[idx_0,:,:].clone()
                mnist_train_targets[idx_1] = class_data.clone()
                comp[class_data] += 1
                idx_1 += 1
            idx_0 += 1
        mnist_train.data, mnist_train.targets = mnist_train_data, mnist_train_targets

        # Create the training data loader
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = args.batch_size, shuffle=True)





        ### Testing data

        # Load the MNIST dataset, apply the transformations if appropraite from above, and target transformations to multiple output neurons per class if needed
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                                transform=torchvision.transforms.Compose(transforms_test),
                                                target_transform=ReshapeTransformTarget(10, args))


        # Now reduce the number of testing data points to N_data_test, but keep the same number of data points per class
        mnist_test_data, mnist_test_targets, comp = torch.empty(N_data_test,28,28,dtype=mnist_test.data.dtype), torch.empty(N_data_test,dtype=mnist_test.targets.dtype), torch.zeros(N_class)
        idx_0, idx_1 = 0, 0
        while idx_1 < N_data_test:
            class_data = mnist_test.targets[idx_0]
            if comp[class_data] < int(N_data_test/N_class):
                mnist_test_data[idx_1,:,:] = mnist_test.data[idx_0,:,:].clone()
                mnist_test_targets[idx_1] = class_data.clone()
                comp[class_data] += 1
                idx_1 += 1
            idx_0 += 1

        mnist_test.data, mnist_test.targets = mnist_test_data, mnist_test_targets

        # Create the testing data loader
        test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = 1, shuffle=False)



        return train_loader, test_loader, mnist_train
    







########## BINARY QUADRATIC MODEL (BQM) FUNCTIONS ##########

# def createIsingProblem(net, args, input, beta = 0, target = None, simulated = 1): # TODO delete if new method works

#     with torch.no_grad():

#         ### BIASES

#         # Use bias trick to encode the input layer 
#         bias_input = np.matmul(input, net.weights_0)
#         bias_lim = args.bias_lim
#         h = {idx_loc: (bias + bias_input[idx_loc]).clip(-bias_lim,bias_lim).item() for idx_loc, bias in enumerate(net.bias_0)} # .clip(.,.) from numpy and .item() converts to python scalar

#         # Calculate nudge biases for the output layer if target is provided or use the original biases but clipped to bias_lim if not
#         if target is not None:
#             bias_nudge = -beta*target

#             # Note that the output layer biases are indexed from the second layer (hidden) onwards, remembering that the input layer is encoded in the of the hidden layer and don't need biases themselves
#             h.update({idx_loc + args.layersList[1]: (bias + bias_nudge[idx_loc]).clip(-bias_lim,bias_lim).item() for idx_loc, bias in enumerate(net.bias_1)})
#         else:
#             h.update({idx_loc + args.layersList[1]: bias.clip(-bias_lim,bias_lim).item() for idx_loc, bias in enumerate(net.bias_1)})


#         ### WEIGHTS

#         J = {}

#         # Weights between hidden and output layer
#         for k in range(args.layersList[1]):
#             for j in range(args.layersList[2]):
#                 # Again the output layer weights are indexed from the second layer (hidden) onwards hence the +args.layersList[1]
#                 # Here we just clip the weights to -1,1 to turn into a QUBO problem
#                 J.update({(k,j+args.layersList[1]): net.weights_1[k][j].clip(-1,1)}) # TODO think if want to use clipping or not

#         return h, J
    


def createIsingProblem(net, args, input, beta=0, target=None, simulated=1):

    with torch.no_grad():
        ### BIASES

        # Use bias trick to encode the input layer
        bias_input = np.matmul(input, net.weights_0)
        bias_lim = args.bias_lim

        # Hidden layer biases
        h_hidden = (net.bias_0 + bias_input).clip(-bias_lim, bias_lim)

        # Output layer biases
        if target is not None:
            bias_nudge = -beta * target
            h_output = (net.bias_1 + bias_nudge).clip(-bias_lim, bias_lim)
        else:
            h_output = net.bias_1.clip(-bias_lim, bias_lim)

        # Combine biases
        h = np.concatenate((h_hidden, h_output))

        ### WEIGHTS

        # Weights between hidden and output layers
        weights = net.weights_1.clip(-1, 1)

        # Number of spins
        num_hidden = args.layersList[1]
        num_output = args.layersList[2]
        num_spins = num_hidden + num_output

        if simulated == 0:
            # Generate h and J as arrays
            # Initialize J as a zero matrix
            J = np.zeros((num_spins, num_spins))

            # Fill in the J matrix with symmetric weights
            J[:num_hidden, num_hidden:] = weights
            J[num_hidden:, :num_hidden] = weights.T  # Ensure symmetry

            # Return h and J as numpy arrays
            return h, J

        else:
            # For simulated == 1, generate h and J as dictionaries
            h_dict = {idx: h_val.item() for idx, h_val in enumerate(h)}
            J_dict = {}

            # Create J as a dictionary
            for k in range(num_hidden):
                for j in range(num_output):
                    idx1 = k
                    idx2 = num_hidden + j
                    weight = weights[k, j]
                    if weight != 0:
                        J_dict[(idx1, idx2)] = weight

            return h_dict, J_dict












########## TRAINING AND TESTING FUNCTIONS ##########

def train(net, args, train_loader, sa_sampler):
    '''
    function to train the network for 1 epoch
    '''

    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Training")
        # Iterate over the training data
        for idx, (DATA, TARGET) in enumerate(progressbars(train_loader, position=0, leave=True)): # position=0 i.e. progress bar at top, leave=True i.e. progress bar remains after completion

            # These variables are used to store all free/nudge states for the batch (used to calculate loss and error later)
            store_seq = None
            store_s = None

            # Iterate over the batch
            for k in range(DATA.size()[0]):
                data, target = DATA[k].numpy(), TARGET[k].numpy() # Convert pytorch tensor to numpy array



                ### FREE PHASE
                # Create the BQM model for the free phase with the input data
                h, J = createIsingProblem(net, args, data, simulated=args.simulated)


                # Simulated annealing sampling
                if args.simulated == 1: 

                    model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)
                    oim_seq = sa_sampler.sample(model, num_reads = args.n_iter_free, num_sweeps = 100)

                    # Print energies of the samples
                    # energies = model.energies(oim_seq.record["sample"])
                    # print(f"SA Energies: {sorted(energies.tolist())}")

                    # Print samples with those energies
                    # print(f"SA Samples: {oim_seq.record['sample']}")




                # OIM sampling
                else: # TODO readd

                    # TODO speed up
                    oim_seq = []
                    oim_seq_energies = []

                    for i in range(args.n_iter_free):
                        # result = oim_problem_solver(-2*J, 20.0, 0.01, h=-h, oim_dynamics_function=simple_oim_dynamics) # TODO fiddle with duration and timestep and everything else
                        result = oim_problem_solver(-2*J, 20.0, 0.01, h=-h, noise=True) 
                        oim_seq.append(result[0])
                        oim_seq_energies.append(result[1])

                    oim_seq = [x for _, x in sorted(zip(oim_seq_energies, oim_seq), key=lambda pair: pair[0])]
                    oim_seq_energies = sorted(oim_seq_energies)

                    # print(f"OIM Energies: {oim_seq_energies}")

                    










                ## Nudge phase: same system except bias for the output layer
                h, J = createIsingProblem(net, args, data, beta = args.beta, target = target, simulated=args.simulated)                


                # First sample
                if args.simulated == 1:
                    first_seq_sample = oim_seq.record["sample"][0]
                else:
                    first_seq_sample = oim_seq[0]




                # TODO check this works for OIMs too
                if args.simulated == 1 and np.array_equal(first_seq_sample.reshape(1,-1)[:,args.layersList[1]:][0], target): # oim_seq.record["sample"] is an array of 10 simple arrays for each sample from the 10 in the sampler, [0] is the first one
                
                    oim_s = oim_seq # i.e. set the nudged state to be the same as the free state in this case 

                else:

                    # Simulated reverse annealing
                    if args.simulated == 1:

                        model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)
                    
                        # TODO can we make this use first_seq_sample instead of oim_seq.record["sample"][0]?
                        oim_s = sa_sampler.sample(model, num_reads = args.n_iter_nudge, num_sweeps = 100, initial_states = oim_seq.first.sample, reverse = True, fraction_annealed = args.frac_anneal_nudge)

                        # print("SA Nudge Energies")
                        # print(model.energies(oim_s.record["sample"]))

                    # OIM reverse annealing
                    else:
                        oim_s = []
                        oim_s_energies = []

                        for i in range(args.n_iter_nudge):
                            # TODO currently we just re-run OIM as usual for the nudge phase, but maybe some sort of more minimal 'reverse annealing' could be done
                            # TODO speed up
                            # result = oim_problem_solver(-2*J, 20.0, 0.01, h=-h, oim_dynamics_function=simple_oim_dynamics, initial_spins=oim_seq[1]) # oim_seq[0] should be lowest energy # TODO fiddle with duration and timestep and everything else
                            result = oim_problem_solver(-2*J, 20.0, 0.01, h=-h, noise=True, initial_spins=first_seq_sample) # oim_seq[0] should be lowest energy # TODO fiddle with duration and timestep and everything else
                            oim_s.append(result[0])
                            oim_s_energies.append(result[1])


                        # Sort the samples and energies by increasing energy
                        oim_s = [x for _, x in sorted(zip(oim_s_energies, oim_s), key=lambda pair: pair[0])]
                        oim_s_energies = sorted(oim_s_energies)

                        # print("OIM Nudge Energies")
                        # print(oim_s_energies)
                    



                # First sample
                if args.simulated == 1:
                    first_s_sample = oim_s.record["sample"][0]
                else:
                    first_s_sample = oim_s[0]




                # Store all the free and nudged states of all the samples of the batch
                if store_seq is None:
                    store_seq = first_seq_sample.reshape(1, first_seq_sample.shape[0]) #qpu_seq
                    store_s = first_s_sample.reshape(1, first_s_sample.shape[0]) #qpu_s
                else:
                    store_seq = np.concatenate((store_seq, first_seq_sample.reshape(1, first_seq_sample.shape[0])),0)
                    store_s = np.concatenate((store_s, first_s_sample.reshape(1, first_s_sample.shape[0])),0)




                # Delete variables to free up memory
                del oim_seq, oim_s
                del data, target


            # Separate into [hidden layer, output layer] for the free and nudged states for all samples of the batch
            seq = [store_seq[:,:args.layersList[1]], store_seq[:,args.layersList[1]:]]
            s   = [store_s[:,:args.layersList[1]], store_s[:,args.layersList[1]:]]


            ## Compute loss and error for combined all samples of the batch
            loss, pred = net.computeLossAcc(seq, TARGET, args)

            # Add loss and error to the total for the training set
            total_pred += pred
            total_loss += loss


            net.updateParams(DATA, s, seq, args)

            del seq, s
            del loss, pred 
            del DATA, TARGET

    print(f"Total Loss: {total_loss} (Normalised: {total_loss/len(train_loader.dataset)})")
    print(f"Total Pred: {total_pred} (Normalised: {total_pred/len(train_loader.dataset)})")
    return total_loss, total_pred






def test(net, args, test_loader, oim_sampler):
    '''
    function to test the network
    '''
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Testing")
        for idx, (data, target) in enumerate(progressbars(test_loader, position=0, leave=True)): # position=0 i.e. progress bar at top, leave=True i.e. progress bar remains after completion
            data, target = data.numpy()[0], target.numpy()[0]

            ## Free phase
            h, J = createIsingProblem(net, args, data, simulated=args.simulated)
            if args.simulated == 1:
                    model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)

            # Simulated annealing sampling
            if args.simulated == 1:
                actual_seq = oim_sampler.sample(model, num_reads = args.n_iter_free, num_sweeps = 100)

            # QPU sampling
            else:
                actual_seq = []
                actual_seq_energies = []

                for i in range(args.n_iter_free):
                    # result = oim_problem_solver(-2*J, 20.0, 0.1, h=-h, oim_dynamics_function=simple_oim_dynamics)
                    result = oim_problem_solver(-2*J, 20.0, 0.1, h=-h, noise=True)
                    actual_seq.append(result[0])
                    actual_seq_energies.append(result[1])

                
                # Sort the samples by increasing energy
                actual_seq = [x for _, x in sorted(zip(actual_seq_energies, actual_seq), key=lambda pair: pair[0])]
                actual_seq_energies = sorted(actual_seq_energies)

                # print(f"OIM Energies: {actual_seq_energies}")

            



            # First sample
            if args.simulated == 1:
                actual_seq = actual_seq.record["sample"][0].reshape(1, actual_seq.record["sample"][0].shape[0]) 
            else:
                actual_seq = actual_seq[0].reshape(1, actual_seq[0].shape[0])


            ## Compute loss and error for QPU sampling
            seq = [actual_seq[:, :args.layersList[1]], actual_seq[:, args.layersList[1]:]] # Again separate into [hidden layer, output layer]

            loss, pred = net.computeLossAcc(seq, target.reshape(1,target.shape[0]), args)

            # Add loss and error to the total for the test set
            total_pred += pred
            total_loss += loss

            del actual_seq, seq
            del pred, loss 
            del data, target

    print(f"Total Loss: {total_loss} (Normalised: {total_loss/len(test_loader.dataset)})")
    print(f"Total Pred: {total_pred} (Normalised: {total_pred/len(test_loader.dataset)})")
    return total_loss, total_pred









