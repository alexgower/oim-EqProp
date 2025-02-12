import os
import os.path
import datetime
import shutil

import pandas as pd

from scipy import*
from copy import*
import pickle




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




