print("Starting imports...")  # Add at very top of file
import os
import gc
import numpy as np
print("Basic imports done")
print("Importing torch...")
import torch
print("Torch imported")
import argparse
print("Importing wandb...")
import wandb
print("Wandb imported")
from tqdm import tqdm as progressbars

print("Importing local modules...")
print("Importing train_test...")
from train_test import train, test
print("Importing dataset_generation_tools...")
from tools.dataset_generation_tools import generate_mnist
print("Importing project_tools...")
from tools.project_tools import createPath, initDataframe, updateDataframe, save_model_numpy, load_model_numpy, saveHyperparameters
print("Importing Network...")
from Network import Network
print("All imports completed")

def print_network_stats(net, args):
    """Print network statistics in a clean format."""
    print("\nNetwork Statistics:")
    print(f"Weights (max abs) - Input->Hidden: {np.max(np.abs(net.weights_0)):.4f}, Hidden->Output: {np.max(np.abs(net.weights_1)):.4f} (clip: {args.J_clip})")
    print(f"Biases (max abs) - Hidden: {np.max(np.abs(net.bias_0)):.4f}, Output: {np.max(np.abs(net.bias_1)):.4f} (clip: {args.h_clip})")
    print(f"Syncs (max abs) - Hidden: {np.max(np.abs(net.sync_0)):.4f}, Output: {np.max(np.abs(net.sync_1)):.4f} (clip: {args.sync_clip})")


def main():
    print("\nStarting main.py...")
    
    # Parse arguments from command line
    print("Setting up argument parser...")
    parser = argparse.ArgumentParser(description='Continuous Equilibrium Propagation with OIM')

    # Add wandb arguments
    parser.add_argument('--wandb_project', type=str, default='oim-eq-prop', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='alexgower-team', help='WandB entity/username')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')
    parser.add_argument('--wandb_mode', type=str, default='online', help='WandB mode (online/offline/disabled)')

    # Simulation parameters
    parser.add_argument('--simulation_type', type=int, default=0, help='Type of simulation, 0=OIM, 1=SA (default=0)')
    parser.add_argument('--rounding', type=int, default=0, help='Rounding type (default=0)') # TODO theres no rounding in this current codebase
    parser.add_argument('--exact_grads', type=int, default=0, help='Exact gradients calculated using RBP instead of EP (default=0)')
    parser.add_argument('--debug', type=int, default=0, help='Debug mode (default=0)')

    # OIM dynamics parameters
    parser.add_argument('--oim_duration', type=float, default=40.0, help='Duration of the OIM simulation (default=20.0)')
    parser.add_argument('--oim_dt', type=float, default=0.2, help='Time step of the OIM simulation (default=0.1)')
    parser.add_argument('--oim_noise', type=int, default=0, help='Noise in the OIM simulation (default=0)')
    parser.add_argument('--oim_random_initialization', type=int, default=0, help='Random initialization of the OIM simulation as opposed to fixed pi/2 initialisation (default=1)')
    parser.add_argument('--nudge_reinitialisation', type=int, default=0, help='Nudge phase uses reinitialised phases (not free steady state phases) in the OIM simulation (default=1)')
    parser.add_argument('--n_procs', type=int, default=None, help='Number of processors for parallel processing (default=None)')

    # Network parameters
    parser.add_argument('--layersList', nargs='+', type=int, default=[784, 500, 10], help='List of layer sizes (default: [784, 120, 40])')
    parser.add_argument('--batch_size', type=int, default=40, help='Size of mini-batches (default=4)')
    parser.add_argument('--max_chunk_size', type=int, default=40, help='Size of chunks for parallel processing (default=20)')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta - hyperparameter of EP (default=1.0)') # TODO think
    
    # Clipping parameters
    parser.add_argument('--h_clip', type=float, default=1.0, help='Max limit for local h (default=1)') # TODO think    
    parser.add_argument('--J_clip', type=float, default=1.0, help='Max limit for local J (default=1)') # TODO think
    parser.add_argument('--sync_clip', type=float, default=1.0, help='Max limit for sync terms (default=1)') # TODO think



    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default=50)')
    parser.add_argument('--lrW0', type=float, default=0.01, help='Learning rate for weights - input-hidden (default=0.01)') # TODO think
    parser.add_argument('--lrW1', type=float, default=0.01, help='Learning rate for weights - hidden-output (default=0.01)') # TODO think
    parser.add_argument('--lrB0', type=float, default=0.001, help='Learning rate for biases - hidden (default=0.001)')
    parser.add_argument('--lrB1', type=float, default=0.001, help='Learning rate for biases - output (default=0.001)')
    parser.add_argument('--lrSYNC0', type=float, default=0.001, help='Learning rate for syncs - input-hidden (default=0.001)')
    parser.add_argument('--lrSYNC1', type=float, default=0.001, help='Learning rate for syncs - hidden-output (default=0.001)')
    parser.add_argument('--gain_weight0', type=float, default=0.5, help='Gain for weights - input-hidden (default=0.5)')  # TODO think
    parser.add_argument('--gain_weight1', type=float, default=0.25, help='Gain for weights - hidden-output (default=0.25)')  # TODO think
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use, mnist or digits (default=mnist)')
    parser.add_argument('--N_data_train', type=int, default=60000, help='Number of data points for training (default=10000)') 
    parser.add_argument('--N_data_test', type=int, default=10000, help='Number of data points for testing (default=1000)')
    parser.add_argument('--data_augmentation', type=int, default=0, help='Set data augmentation or not for the problems - (default=False)')
    parser.add_argument('--mnist_positive_negative_remapping', type=int, default=0, help='Remap MNIST data from [0,1] to [-1,1] range (default=1)')

    # Other parameters
    parser.add_argument('--load_model', type=int, default=0, help='Load previously trained model (default=0)')

    args = parser.parse_args()
    print("\nArguments parsed successfully")

    # Initialize wandb if not disabled
    if args.wandb_mode != 'disabled':
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
            mode=args.wandb_mode
        )

    print("Arguments:", args)

    with torch.no_grad():
        # Setup paths and tracking
        print("\nSetting up paths and tracking...")
        base_path = createPath(args)
        dataframe = initDataframe(base_path)
        print(f"Saving results to: {base_path}")

        # Load MNIST dataset
        # TODO try digits dataset too at some point
        train_loader, test_loader, dataset = generate_mnist(args)

        # Initialize or load network
        if args.load_model == 0:
            net = Network(args)
            saveHyperparameters(base_path, args)
        else:
            net = load_model_numpy(base_path)

        # Pre-allocate arrays for tracking
        train_loss = np.zeros(args.epochs, dtype=np.float32)
        train_error = np.zeros(args.epochs, dtype=np.float32)
        test_loss = np.zeros(args.epochs, dtype=np.float32)
        test_error = np.zeros(args.epochs, dtype=np.float32)
        print("\nStarting training loop...")

        # Training loop
        for epoch in progressbars(range(args.epochs), desc="Training Progress"):
            print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
            
            
            # Train
            loss, error = train(net, args, train_loader)
            train_loss[epoch] = loss
            train_error[epoch] = error

            # Test
            loss, error = test(net, args, test_loader)
            test_loss[epoch] = loss
            test_error[epoch] = error

            # Log metrics to wandb if enabled
            if args.wandb_mode != 'disabled':
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss[epoch],
                    'train/error': train_error[epoch]/len(train_loader.dataset)*100,
                    'test/loss': test_loss[epoch],
                    'test/error': test_error[epoch]/len(test_loader.dataset)*100,
                    'network/weights_0_max': np.max(np.abs(net.weights_0)),
                    'network/weights_1_max': np.max(np.abs(net.weights_1)),
                    'network/bias_0_max': np.max(np.abs(net.bias_0)),
                    'network/bias_1_max': np.max(np.abs(net.bias_1)),
                    'network/sync_0_max': np.max(np.abs(net.sync_0)),
                    'network/sync_1_max': np.max(np.abs(net.sync_1))
                })

            # Update tracking
            dataframe = updateDataframe(
                base_path, dataframe,
                train_error[epoch]/len(train_loader.dataset)*100,
                test_error[epoch]/len(test_loader.dataset)*100,
                train_loss[epoch], test_loss[epoch]
            )
            
            # Save model state
            save_model_numpy(base_path, net)

            # Print network statistics
            print_network_stats(net, args)

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close wandb run if it was used
        if args.wandb_mode != 'disabled':
            wandb.finish()

if __name__ == '__main__':
    main()


