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

    data = [train_acc, test_acc, train_loss, test_loss]

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

    normalisation = 8 # Apparently scikitlearn digits are in [0,16] so normalise by 8 to get [0,2] so mappable to [-1,1]
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

def createIsingProblem(net, args, input, beta=0, target=None, simulated=1):

    with torch.no_grad():
        ### BIASES

        # Use bias trick to encode the input layer
        bias_input = np.matmul(input, net.weights_0)
        h_clip = args.h_clip

        # Hidden layer biases
        h_hidden = (net.bias_0 + bias_input).clip(-h_clip, h_clip)

        # Output layer biases
        if target is not None:
            bias_nudge = -beta * target
            h_output = (net.bias_1 + bias_nudge).clip(-h_clip, h_clip)
        else:
            h_output = net.bias_1.clip(-h_clip, h_clip)

        # Combine biases
        h = np.concatenate((h_hidden, h_output))

        ### WEIGHTS

        # Weights between hidden and output layers
        J_clip = args.J_clip

        weights = net.weights_1.clip(-J_clip, J_clip) 

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

def train(net, args, train_loader, sampler):
    '''
    function to train the network for 1 epoch with memory optimization
    '''
    import gc  # Add garbage collector import
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Training")
        for idx, (DATA, TARGET) in enumerate(progressbars(train_loader, position=0, leave=True)):
            # Force garbage collection at start of each batch
            gc.collect()
            store_seq = None
            store_s = None

            # Pre-allocate numpy arrays for batch to avoid repeated allocations
            batch_size = DATA.size()[0]
            total_size = args.layersList[1] + args.layersList[2]  # Hidden + output size
            store_seq = np.zeros((batch_size, total_size))
            store_s = np.zeros((batch_size, total_size))

            # Iterate over the batch
            for k in range(batch_size):
                data, target = DATA[k].numpy(), TARGET[k].numpy()

                ### FREE PHASE
                h, J = createIsingProblem(net, args, data, simulation_type=args.simulation_type)

                if args.simulation_type == 0: # OIM annealing
                    if args.oim_dynamics == 0:
                        dynamics = sampler.wang_oim_dynamics
                        stochastic_dynamics = sampler.wang_oim_stochastic_dynamics
                    elif args.oim_dynamics == 1:
                        dynamics = sampler.simple_oim_dynamics
                        stochastic_dynamics = sampler.simple_oim_stochastic_dynamics

                    best_seq_sample, _ = sampler.oim_parallel_problem_solver(
                        -2*J, args.oim_duration, args.oim_dt, 
                        h=-h, 
                        noise=args.oim_noise==1,
                        runs=args.n_iter_free,
                        oim_dynamics_function=dynamics,
                        oim_stochastic_dynamics_function=stochastic_dynamics,
                        rounding=args.rounding==1,
                        simple_rounding_only=args.oim_simple_rounding_only==1
                    )
                    best_seq_sample_to_store = best_seq_sample
                    del best_seq_sample  # Cleanup

                elif args.simulation_type== 1: # Simulated annealing
                    model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)
                    sa_seq = sampler.sample(model, num_reads=args.n_iter_free, num_sweeps=100)
                    best_seq_sample = sa_seq.first.sample
                    best_seq_sample_to_store = np.array([best_seq_sample[i] for i in range(len(best_seq_sample))])
                    del sa_seq, best_seq_sample  # Cleanup

                elif args.simulation_type == 2: # Scellier annealing
                    # TODO stuff here
                    print("Scellier annealing not implemented yet")
                else: 
                    print("Invalid simulation type")
                    raise ValueError("Invalid simulation type")
                    

                ### NUDGE PHASE
                h, J = createIsingProblem(net, args, data, beta=args.beta, target=target, simulation_type=args.simulation_type)

                if np.array_equal(best_seq_sample_to_store.reshape(1,-1)[:,args.layersList[1]:][0], target):
                    best_s_sample_to_store = best_seq_sample_to_store
                else:
                    if args.simulation_type == 0: # OIM annealing
                        if args.oim_dynamics == 0:
                            dynamics = sampler.wang_oim_dynamics
                            stochastic_dynamics = sampler.wang_oim_stochastic_dynamics
                        elif args.oim_dynamics == 1:
                            dynamics = sampler.simple_oim_dynamics
                            stochastic_dynamics = sampler.simple_oim_stochastic_dynamics
                    
                        best_s_sample, _ = sampler.oim_parallel_problem_solver(
                            -2*J, args.oim_duration, args.oim_dt,
                            h=-h,
                            noise=args.oim_noise==1,
                            initial_spins=best_seq_sample_to_store,
                            runs=args.n_iter_nudge,
                            oim_dynamics_function=dynamics,
                            oim_stochastic_dynamics_function=stochastic_dynamics,
                            rounding=args.rounding==1,
                            simple_rounding_only=args.oim_simple_rounding_only==1
                        )
                        best_s_sample_to_store = best_s_sample
                        del best_s_sample  # Cleanup

                    elif args.simulation_type == 1: # Simulated annealing
                        model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)
                        sa_s = sampler.sample(
                            model,
                            num_reads=args.n_iter_nudge,
                            num_sweeps=100,
                            initial_states=best_seq_sample_to_store,
                            reverse=True,
                            fraction_annealed=args.frac_anneal_nudge
                        )
                        best_s_sample = sa_s.first.sample
                        best_s_sample_to_store = np.array([best_s_sample[i] for i in range(len(best_s_sample))])
                        del sa_s, best_s_sample  # Cleanup

                    elif args.simulation_type == 2: # Scellier annealing
                        # TODO stuff here
                        print("Scellier annealing not implemented yet")
                    
                    else:
                        print("Invalid simulation type")
                        raise ValueError("Invalid simulation type")
                        

                # Store batch results
                store_seq[k] = best_seq_sample_to_store
                store_s[k] = best_s_sample_to_store

                # Clean up iteration variables
                del h, J, data, target
                del best_seq_sample_to_store, best_s_sample_to_store

            # Separate into [hidden layer, output layer] for the free and nudged states
            seq = [store_seq[:,:args.layersList[1]], store_seq[:,args.layersList[1]:]]
            s = [store_s[:,:args.layersList[1]], store_s[:,args.layersList[1]:]]

            loss, pred = net.computeLossAcc(seq, TARGET, args)
            total_pred += pred
            total_loss += loss

            net.updateParams(DATA, s, seq, args)

            # Clean up batch variables
            del seq, s, store_seq, store_s
            del loss, pred
            del DATA, TARGET
            
            # Force garbage collection at end of batch
            gc.collect()

    print(f"Total Loss: {total_loss} (Normalised: {total_loss/len(train_loader.dataset)})")
    print(f"Total Pred: {total_pred} (Normalised: {total_pred/len(train_loader.dataset)})")
    return total_loss, total_pred




def test(net, args, test_loader, sampler):
    '''
    function to test the network
    '''
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Testing")
        for idx, (data, target) in enumerate(progressbars(test_loader, position=0, leave=True)): # position=0 i.e. progress bar at top, leave=True i.e. progress bar remains after completion
            data, target = data.numpy()[0], target.numpy()[0]

            ## Free phase
            h, J = createIsingProblem(net, args, data, simulation_type=args.simulation_type)

            # Simulated sampling
            if args.simulation_type == 0: # OIM annealing
                if args.oim_dynamics == 0:
                    dynamics = sampler.wang_oim_dynamics
                    stochastic_dynamics = sampler.wang_oim_stochastic_dynamics
                elif args.oim_dynamics == 1:
                    dynamics = sampler.simple_oim_dynamics
                    stochastic_dynamics = sampler.simple_oim_stochastic_dynamics

                # Use Julia's parallel OIM solver for free phase
                actual_seq, _ = sampler.oim_parallel_problem_solver(
                    -2*J, args.oim_duration, args.oim_dt, 
                    h=-h, 
                    noise=args.oim_noise==1,
                    runs=args.n_iter_free,
                    oim_dynamics_function=dynamics,
                    oim_stochastic_dynamics_function=stochastic_dynamics,
                    rounding=args.rounding==1,
                    simple_rounding_only=args.oim_simple_rounding_only==1
                )

                # Reshaping
                actual_seq = actual_seq.reshape(1, actual_seq.shape[0])


            elif args.simulation_type == 1:
                    model = dimod.BinaryQuadraticModel.from_ising(h, J, 0)
                    actual_seq = sampler.sample(model, num_reads = args.n_iter_free, num_sweeps = 100)

                    # Reshaping
                    actual_seq = actual_seq.record["sample"][0].reshape(1, actual_seq.record["sample"][0].shape[0]) 


            elif args.simulation_type == 2: # Scellier annealing
                # TODO stuff here
                print("Scellier annealing not implemented yet")

            else:
                print("Invalid simulation type")
                raise ValueError("Invalid simulation type")
                
                
            


            ## Compute loss and error for 
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





########## BATCH FUNCTION FOR OIM JULIA ONLY ##########

def train_oim_julia_batch_parallel(net, args, train_loader, OIMSimulations):
    '''
    function to train the network for 1 epoch using Julia's batch parallel OIM solver,
    with memory optimization
    '''
    import gc  # Add garbage collector import
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Training")
        
        for idx, (DATA, TARGET) in enumerate(progressbars(train_loader, position=0, leave=True)):
            # Force garbage collection at start of each batch
            gc.collect()
            
            batch_size = DATA.size()[0]
            total_size = args.layersList[1] + args.layersList[2]  # Hidden + output size
            
            ### FREE PHASE
            # Pre-allocate batch data arrays
            J_batch = []
            h_batch = []
            for k in range(batch_size):
                h, J = createIsingProblem(net, args, DATA[k].numpy(), simulated=0)
                J_batch.append(-2*J)
                h_batch.append(-h)
                del h, J  # Cleanup intermediates
            
            # Select dynamics functions
            if args.oim_dynamics == 0:
                dynamics = OIMSimulations.wang_oim_dynamics
                stochastic_dynamics = OIMSimulations.wang_oim_stochastic_dynamics
            elif args.oim_dynamics == 1:
                dynamics = OIMSimulations.simple_oim_dynamics
                stochastic_dynamics = OIMSimulations.simple_oim_stochastic_dynamics
                
            # Batch solve free phase
            best_seq_configs, best_seq_energies = OIMSimulations.oim_batch_parallel_problem_solver(
                J_batch, h_batch,
                duration=args.oim_duration,
                timestep=args.oim_dt,
                noise=args.oim_noise==1,
                runs=args.n_iter_free,
                oim_dynamics_function=dynamics,
                oim_stochastic_dynamics_function=stochastic_dynamics,
                rounding=args.rounding==1,
                simple_rounding_only=args.oim_simple_rounding_only==1
            )

            if args.comparison==1:
                # If comprison, first print the best sequence configurations and energies
                print(f"Best sequence configurations: {best_seq_configs}")
                print(f"Best sequence energies: {best_seq_energies}")

                # Then also run another free phase (this will be hardcoded in))
                best_seq_configs, best_seq_energies = OIMSimulations.oim_batch_parallel_problem_solver(
                    J_batch, h_batch,
                    duration=args.oim_duration,
                    timestep=args.oim_dt,
                    noise=args.oim_noise==1,
                    runs=args.n_iter_free,
                    oim_dynamics_function=dynamics,
                    oim_stochastic_dynamics_function=stochastic_dynamics,
                    rounding=0==1,
                )

                # Print the best sequence configurations and energies
                print(f"Comparison sequence configurations: {best_seq_configs}")
                print(f"Comparison sequence energies: {best_seq_energies}")
                
            # Convert to list and cleanup
            best_seq_samples = list(best_seq_configs)
            del best_seq_configs, best_seq_energies, J_batch, h_batch

            ### NUDGE PHASE
            # Pre-allocate arrays for nudge phase
            need_nudge = []
            best_s_samples = [None] * batch_size  # Pre-allocate full result array
            J_batch = []
            h_batch = []
            initial_spins = []
            
            # Determine which samples need nudging
            for k in range(batch_size):
                data, target = DATA[k].numpy(), TARGET[k].numpy()
                if np.array_equal(best_seq_samples[k].reshape(1,-1)[:,args.layersList[1]:][0], target):
                    best_s_samples[k] = best_seq_samples[k]
                else:
                    need_nudge.append(k)
                    h, J = createIsingProblem(net, args, data, beta=args.beta, target=target, simulated=0)
                    J_batch.append(-2*J)
                    h_batch.append(-h)
                    initial_spins.append(best_seq_samples[k])
                    del h, J  # Cleanup

            # Only run nudge phase if needed
            if need_nudge:
                # Batch solve nudge phase
                best_s_configs, best_s_energies = OIMSimulations.oim_batch_parallel_problem_solver(
                    J_batch, h_batch,
                    initial_spins_batch=initial_spins,
                    duration=args.oim_duration,
                    timestep=args.oim_dt,
                    noise=args.oim_noise==1,
                    runs=args.n_iter_nudge,
                    oim_dynamics_function=dynamics,
                    oim_stochastic_dynamics_function=stochastic_dynamics,
                    rounding=args.rounding==1,
                    simple_rounding_only=args.oim_simple_rounding_only==1
                )
                
                # Insert nudged results back in correct positions
                for idx, batch_idx in enumerate(need_nudge):
                    best_s_samples[batch_idx] = best_s_configs[idx]
                
                # Cleanup
                del best_s_configs, best_s_energies
            
            del J_batch, h_batch, initial_spins, need_nudge
            
            # Stack all results
            store_seq = np.stack(best_seq_samples)
            store_s = np.stack(best_s_samples)
            del best_seq_samples, best_s_samples
            
            # Separate into [hidden layer, output layer]
            seq = [store_seq[:,:args.layersList[1]], store_seq[:,args.layersList[1]:]]
            s = [store_s[:,:args.layersList[1]], store_s[:,args.layersList[1]:]]
            del store_seq, store_s

            # Compute loss and accuracy
            loss, pred = net.computeLossAcc(seq, TARGET, args)
            total_pred += pred
            total_loss += loss

            # Update network parameters
            net.updateParams(DATA, s, seq, args)

            # Clean up all remaining batch variables
            del seq, s
            del loss, pred
            del DATA, TARGET
            
            # Force garbage collection at end of batch
            gc.collect()

    print(f"Total Loss: {total_loss} (Normalised: {total_loss/len(train_loader.dataset)})")
    print(f"Total Pred: {total_pred} (Normalised: {total_pred/len(train_loader.dataset)})")
    return total_loss, total_pred