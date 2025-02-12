import matplotlib.pyplot as plt
import numpy as np
import argparse # For parsing arguments from command line
from math import*
from tqdm import tqdm as progressbars # For progress bar
import os # For file management
from random import*
import atexit
from pathlib import Path

from train_test import*
from Tools.dataset_generation_tools import *
from Tools.project_tools import *
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
    
    # Log per-worker memory usage
    try:
        for worker in Distributed.workers():
            worker_mem = Distributed.remotecall_fetch(
                lambda: int(round(Main.eval("Sys.total_memory() / 2^20"))),
                worker
            )
            print(f"Worker {worker} memory: {worker_mem} MB")
    except:
        pass

def should_reset_workers():
   """Check if memory usage is too high and workers need reset"""
   process = psutil.Process()
   memory_percent = process.memory_percent()
   print(f"Current memory usage: {memory_percent:.1f}%")
   
   # Also check individual worker memory
   try:
       for worker in Distributed.workers():
           worker_mem_percent = Distributed.remotecall_fetch(
               lambda: int(round(Main.eval("psutil.Process().memory_percent()"))),
               worker
           )
           if worker_mem_percent > 60:  # Reset if any worker uses >60%
               print(f"Worker {worker} memory usage high: {worker_mem_percent:.1f}%")
               return True
   except:
       pass
       
   return memory_percent > 70  # Reset if main process using >70%

def reset_julia_workers(n_workers):
   """Reset Julia workers during runtime"""
   print("Memory threshold exceeded - resetting workers...")
   try:
       # First try to clean memory on workers without removing them
       for worker in Distributed.workers():
           Distributed.remotecall_fetch(lambda: Main.eval("GC.gc()"), worker)
       
       # If still needed, remove and recreate workers
       if should_reset_workers():
           Distributed.rmprocs(Distributed.workers())
           Main.GC.gc()
           setup_julia_workers(n_workers)
           print("Worker reset complete")
   except Exception as e:
       print(f"Error resetting workers: {e}")

def cleanup_julia_workers():
    """Cleanup Julia workers on program exit"""
    try:
        # First clean memory on workers
        for worker in Distributed.workers():
            Distributed.remotecall_fetch(lambda: Main.eval("GC.gc()"), worker)
            
        # Then remove workers
        Distributed.rmprocs(Distributed.workers())
        Main.GC.gc()
        
        # Clear any remaining Python objects
        gc.collect()
    except:
        pass

def setup_julia_workers(n_workers, simulation_type):
    """Setup Julia workers and load necessary modules"""
    print(f"Adding {n_workers} workers...")
    
    # Calculate max workers based on available memory
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    max_workers = min(n_workers, int(total_memory / 2))  # Assume each worker needs ~2GB
    print(f"Adjusting workers to {max_workers} based on available memory")
    
    Distributed.addprocs(max_workers)

    print("Activating environment on workers...")
    Main.eval('''
        using Distributed;
        @everywhere begin
            using Pkg;
            Pkg.activate(".");
            ENV["GKSwstype"] = "100";  # Disable plotting display
            
            # Pre-allocate some common array sizes to reduce memory fragmentation
            const PREALLOCATED_ARRAYS = Dict{Tuple{Int,Type},Vector}()
            
            function get_preallocated_array(size::Int, T::Type)
                key = (size, T)
                if !haskey(PREALLOCATED_ARRAYS, key)
                    PREALLOCATED_ARRAYS[key] = Vector{T}(undef, size)
                end
                return PREALLOCATED_ARRAYS[key]
            end
        end
    ''')

    if simulation_type == 0:  # OIM
        print("Loading OIM module...")
        module_path = os.path.join("..", "oim-simulator", "code", "EP", "oim_simulations.jl")
        abs_module_path = os.path.abspath(module_path)
        Main.eval(f'''
            include("{abs_module_path}");
            @everywhere include("{abs_module_path}");
            using .OIMSimulations;
            @everywhere using .OIMSimulations;
        ''')
    elif simulation_type == 2:  # Scellier
        print("Loading Scellier module...")
        module_path = os.path.join("..", "oim-simulator", "code", "EP", "scellier_simulations.jl")
        abs_module_path = os.path.abspath(module_path)
        Main.eval(f'''
            include("{abs_module_path}");
            @everywhere include("{abs_module_path}");
            using .ScellierSimulations;
            @everywhere using .ScellierSimulations;
        ''')

def setup_julia(n_workers=40, simulation_type=0):
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

    # Setup workers and load modules
    setup_julia_workers(n_workers, simulation_type)
    
    print("Julia setup complete!")
    return Main.ScellierSimulations if simulation_type == 2 else Main.OIMSimulations



























# Parse arguments from command line
parser = argparse.ArgumentParser(description='Binary Equilibrium Propagation with OIM')

parser.add_argument(
    '--comparison', 
    type=int,
    default=0,
    help='Comparison with hardcoded alternative annealing (default=0)')
parser.add_argument(
    '--verbose',
    type=int,
    default=0,
    help='Print verbose output (default=1)')
parser.add_argument(
    '--load_model',
    type=int,
    default=0,
    help='If we load the parameters from a previously trained model to continue the training (default=0, else = 1)')

parser.add_argument(
    '--oim_duration',
    type=float,
    default=20.0,
    help='Duration of the OIM simulation (default=20.0)')
parser.add_argument(
    '--oim_dt',
    type=float,
    default=0.1,
    help='Time step of the OIM simulation (default=0.1)')
parser.add_argument(
    '--oim_noise',
    type=int,
    default=0,
    help='Noise in the OIM simulation (default=0)')
parser.add_argument(
    '--ode_solver_method',
    type=str,
    default='Euler',
    choices=['Euler', 'RK45'],
    help='ODE solver method to use (default=Euler)')
parser.add_argument(
    '--rounding',
    type=int,
    default=1,
    help='OIM Rounding to Ising spins (1) or keeping as soft cos(\phi) continuous variables (0) after anneal (default=1)')
parser.add_argument(
    '--N_data_train',
    type=int,
    default=10000,
    help='Number of data points for training (default=10000)') 
parser.add_argument(
    '--N_data_test',
    type=int,
    default=1000,
    help='Number of data points for testing (default=1000)')

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Number of epochs (default=50)')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 120, 40],
    help='List of layer sizes (default: [784, 120, 40])')
parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
    help='Size of mini-batches we use (for training only) (default=4)')
parser.add_argument(
    '--beta',
    type=float,
    default=5,
    help='Beta - hyperparameter of EP (default=5)')
parser.add_argument(
    '--h_clip',
    type=float,
    default=1.0,
    help='Max limit for the amplitude of the local h (default=1)')
parser.add_argument(
    '--J_clip',
    type=float,
    default=1.0,
    help='Max limit for the amplitude of the local J (default=1)')

parser.add_argument(
    '--Ks_max_nudge',
    type=float,
    default=0.25,
    help='Maximum value of the Ks for the nudge phase (default=0.25)')
parser.add_argument(
    '--frac_anneal_nudge',
    type=float,
    default=0.25,
    help='fraction of system non-annealed (default=0.25)')

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
    help='Gain for initialization of the weights - input-hidden (default=0.5)')
parser.add_argument(
    '--gain_weight1',
    type=float,
    default=0.25,
    help='Gain for initialization of the weights  - hidden-output (default=0.25)')

args = parser.parse_args()

# Print arguments
print(args)

with torch.no_grad():
    ## Files saving: create a folder for the simulation and save simulation's parameters
    BASE_PATH = createPath(args)
    dataframe = initDataframe(BASE_PATH)
    print(BASE_PATH)

    ## Generate DATA
    if args.dataset == "digits":
        train_loader, test_loader = generate_digits(args)
    elif args.dataset == "mnist":
        train_loader, test_loader, dataset = generate_mnist(args)

    # Print a single example of the data
    for i, (data, target) in enumerate(train_loader):
        print(data[0].shape)
        plt.imshow(data[0].numpy().reshape(28, 28), cmap='gray')
        plt.show()
        break

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

        # Train the network
        loss, error = train(net, args, train_loader)
        train_loss_tab[epoch] = loss
        train_error_tab[epoch] = error

        # Test the network
        loss, error = test(net, args, test_loader)
        test_loss_tab[epoch] = loss
        test_error_tab[epoch] = error

        # Store error and loss at each epoch
        dataframe = updateDataframe(BASE_PATH, dataframe, 
                                  np.array(train_error_tab)[epoch]/len(train_loader.dataset)*100,
                                  np.array(test_error_tab)[epoch]/len(test_loader.dataset)*100,
                                  train_loss_tab[epoch], test_loss_tab[epoch])

        save_model_numpy(BASE_PATH, net)

        print(f"Epoch {epoch}, max weights: {np.max(np.abs(net.weights_0))}, {np.max(np.abs(net.weights_1))}")
        print(f"Epoch {epoch}, max biases: {np.max(np.abs(net.bias_0))}, {np.max(np.abs(net.bias_1))}")

        # Force garbage collection
        gc.collect()


