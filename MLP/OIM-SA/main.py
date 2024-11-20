import matplotlib.pyplot as plt
import numpy as np
import argparse # For parsing arguments from command line
from math import*
from tqdm import tqdm as progressbars # For progress bar
import os # For file management
from random import*
import atexit
from pathlib import Path

from Tools import*
from Network import*


# Used to compare to Laydevant D-Wave based Simulated Annealing
from simulated_sampler import SimulatedAnnealingSampler 


import torch

import psutil
from julia.api import Julia
from julia import Main, Distributed
import gc




def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"Virtual Memory: {memory_info.vms / 1024 / 1024:.2f} MB")
    print(f"Percent: {process.memory_percent():.1f}%")

def should_reset_workers():
   """Check if memory usage is too high and workers need reset"""
   process = psutil.Process()
   memory_percent = process.memory_percent()
   print(f"Current memory usage: {memory_percent:.1f}%")
   return memory_percent > 70  # Reset if using >70% of available RAM

def reset_julia_workers(n_workers):
   """Reset Julia workers during runtime"""
   print("Memory threshold exceeded - resetting workers...")
   try:
       Distributed.rmprocs(Distributed.workers())
       Main.GC.gc()
       setup_julia_workers(n_workers)
       print("Worker reset complete")
   except Exception as e:
       print(f"Error resetting workers: {e}")

def cleanup_julia_workers():
    """Cleanup Julia workers on program exit"""
    try:
        Distributed.rmprocs(Distributed.workers())
        Main.GC.gc()
    except:
        pass

def setup_julia_workers(n_workers):
    """Setup Julia workers and load necessary modules"""
    print(f"Adding {n_workers} workers...")
    Distributed.addprocs(n_workers)

    print("Activating environment on workers...")
    Main.eval('''
        using Distributed;
        @everywhere begin
            using Pkg;
            Pkg.activate(".");
            ENV["GKSwstype"] = "100";  # Disable plotting display
        end
    ''')

    print("Loading OIM module...")
    module_path = os.path.join("..", "oim-simulator", "code", "core", "simulations", "oim_simulations.jl")
    abs_module_path = os.path.abspath(module_path)
    
    Main.eval(f'''
        include("{abs_module_path}");
        @everywhere include("{abs_module_path}");
        using .OIMSimulations;
        @everywhere using .OIMSimulations;
    ''')

def setup_julia(n_workers=40):
    """Initial Julia setup and environment configuration"""
    print("Initializing Julia...")
    
    # Set environment variables before importing Julia
    os.environ["PYTHON"] = ""  # Avoid conda python conflicts
    
    # Create local Julia depot path in the current working directory
    local_depot = os.path.join(os.getcwd(), ".julia_depot")
    if not os.path.exists(local_depot):
        os.makedirs(local_depot)
    os.environ["JULIA_DEPOT_PATH"] = f"{local_depot}:{os.environ.get('JULIA_DEPOT_PATH', '')}"
    
    # Register cleanup function
    atexit.register(cleanup_julia_workers)
    
    # Initialize Julia with custom system image if available
    sysimage_path = Path("oim_custom.so")
    if sysimage_path.exists():
        print("Using custom system image...")
        jl = Julia(compiled_modules=True,
                  sysimage=str(sysimage_path))
    else:
        print("No custom system image found, using default...")
        jl = Julia(compiled_modules=True)

    # Setup workers and load modules
    setup_julia_workers(n_workers)
    
    print("Julia setup complete!")
    return Main.OIMSimulations


# Parse arguments from command line
parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation with D-Wave support')\

parser.add_argument(
    '--comparison', 
    type=int,
    default=0,
    help='Comparison with hardcoded alternative annealing (default=0)')
parser.add_argument(
    '--simulation_type',
    type=int,
    default=0,
    help='specify if we use OIM annealing (=0) or simulated annealing (=1) or scellier dynamics (2) (default=0)') # TODO LATER implement our own SA too?


parser.add_argument(
    '--oim_duration',
    type=float,
    default=20.0,
    help='Duration of the OIM simulation (default=20.0)')
parser.add_argument(
    '--oim_dt',
    type=float,
    default=0.1, # TODO change?
    help='Time step of the OIM simulation (default=0.01)')
parser.add_argument(
    '--oim_noise',
    type=int,
    default=0,
    help='Noise in the OIM simulation (default=0)')
parser.add_argument(
    '--oim_dynamics',
    type=int,
    default=0,
    help='OIM Dynamics (Wang = 0, Simple = 1)')
parser.add_argument(
    '--rounding',
    type=int,
    default=0,
    help='OIM Rounding to Ising spins (1) or keeping as soft cos(\phi_i) continuous variables (0) after anneal (default=1)')
parser.add_argument(
    '--oim_simple_rounding_only',
    type=int,
    default=0,
    help='OIM Simple Rounding Only (default=0)')
parser.add_argument(
    '--N_data_train',
    type=int,
    default=1000, #60000 max
    # default=10000,
    help='Number of data points for training (default=1000)') 
parser.add_argument(
    '--N_data_test',
    type=int,
    default=100, #10000 max
    # default=1000,
    help='Number of data points for testing (default=100)')

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Number of epochs (default=10)')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    # default=[784, 10, 10],
    # default=[784, 120, 40],
    default=[784, 120, 10],
    # default=[784, 500, 40],
    help='List of layer sizes (default: [784, 120, 10])') # TODO NEXT try 10 with continuous
parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
    help='Size of mini-batches we use (for training only) (default=1)')
parser.add_argument(
    '--procs',
    type=int,
    default=40,
    help='Total number of processors to use (default=40)')
parser.add_argument(
    '--beta',
    type=float,
    default=5,
    help='Beta - hyperparameter of EP (default=5)') # TODO change this with continuous or not?
parser.add_argument( # TODO LATER think about this clipping
    '--h_clip',
    type=float,
    default=1.0, # Laydevant did 1.0
    # default=10.0, 
    help='Max limit for the amplitude of the local h applied to the oscillators, either for free or nudge phase - (default=1)')
parser.add_argument( # TODO LATER think about this clipping
    '--J_clip',
    type=float,
    default=1.0, # Laydevant did 1.0
    # default=10.0,
    help='Max limit for the amplitude of the local J applied to the oscillators, either for free or nudge phase - (default=1)')

parser.add_argument(
    '--Ks_max_nudge',
    type=float,
    default=0.25, # TODO maybe smaller beta factor or max nudge for continuous?
    help='Maximum value of the Ks for the nudge phase (default=0.25)') # TODO LATER implement reverse annealing
parser.add_argument(
    '--frac_anneal_nudge',
    type=float,
    default=0.25,
    help='fraction of SA system non-annealed (default=0.5, if <0.5: more annealing, if > 0.5, less annealing)')

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
    '--load_model',
    type=int,
    default=0,
    help='If we load the parameters from a previously trained model to continue the training (default=0, else = 1)')


parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='Dataset we use for training (default=mnist, others: digits)')
parser.add_argument(
    '--data_augmentation',
    type=int,
    default=0,
    help='Set data augmentation or not for the problems - (default=False)')
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
    '--gain_weight0',
    type=float,
    default=0.5,
    help='Gain for initialization of the weights - input-hidden (default=1)')
parser.add_argument(
    '--gain_weight1',
    type=float,
    default=0.25,
    help='Gain for initialization of the weights  - hidden-output (default=1)')



args = parser.parse_args()

# Print arguments
print(args)
print("Number of processors: ", args.procs)


# Setup julia stuff if using OIM
if args.simulated == 0:
    OIMSimulations = setup_julia(n_workers=args.procs)





with torch.no_grad():

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
    train_loss_tab = np.zeros(args.epochs)
    train_error_tab = np.zeros(args.epochs)
    test_loss_tab = np.zeros(args.epochs)
    test_error_tab = np.zeros(args.epochs)

    for epoch in progressbars(range(args.epochs)):
        print(f"Epoch {epoch+1}/{args.epochs}")
        log_memory_usage()

        if epoch > 0 and should_reset_workers():  # Skip first epoch check
            reset_julia_workers(args.procs)
            gc.collect()  # Force Python garbage collection after reset
            OIMSimulations = Main.OIMSimulations  # Re-establish OIMSimulations after reset


        # Train the network
        # loss, error = train(net, args, train_loader, sampler)
        if args.simulated == 1:
            loss, error = train(net, args, train_loader, SimulatedAnnealingSampler())
        else:
            loss, error = train_oim_julia_batch_parallel(net, args, train_loader, OIMSimulations)
        train_loss_tab[epoch] = loss
        train_error_tab[epoch] = error

        # Test the network
        if args.simulated == 1:
            loss, error = test(net, args, test_loader, SimulatedAnnealingSampler())
        else:
            loss, error = test(net, args, test_loader, OIMSimulations)
        test_loss_tab[epoch] = loss
        test_error_tab[epoch] = error

        # Store error and loss at each epoch
        dataframe = updateDataframe(BASE_PATH, dataframe, np.array(train_error_tab)[epoch]/len(train_loader.dataset)*100, np.array(test_error_tab)[epoch]/len(test_loader.dataset)*100, train_loss_tab[epoch], test_loss_tab[epoch])

        save_model_numpy(BASE_PATH, net)



