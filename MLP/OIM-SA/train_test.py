from tqdm import tqdm as progressbars

import numpy as np 

import torch
from scipy import*
from copy import*

import dimod
import gc
# from solvers.oim_dynamics import OIMDynamics, batch_solve_oim  # Using Julia solver instead
from solvers.julia_oim_bridge import batch_solve_oim





########## BINARY QUADRATIC MODEL (BQM) FUNCTIONS ##########

def createIsingProblem(net, args, input, beta=0.0, target=None):
    """Create Ising problem parameters for OIM dynamics.
    
    Args:
        net: Network object containing weights and biases
        args: Arguments containing layer sizes and clipping values
        input: Input data
        beta: Nudging strength (default=0.0 for free phase)
        target: Target data (only needed for nudged phase)
        
    Returns:
        h: External field vector
        J: Coupling matrix
        K_s: SHIL sync vector
    """
    with torch.no_grad():
        # Pre-compute sizes
        num_hidden = args.layersList[1]
        num_output = args.layersList[2]
        num_spins = num_hidden + num_output

        # Initialize arrays with correct sizes
        J = np.zeros((num_spins, num_spins), dtype=np.float32)
        h = np.zeros(num_spins, dtype=np.float32)
        K_s = np.zeros(num_spins, dtype=np.float32)

        ### BIASES AND SYNCS

        # Hidden layer biases and syncs
        # Use bias trick to encode the input layer
        # BUT THIS ONLY WORKS FOR E_{ij} = J_ij * s_i * s_j product form
        # Therefore if input s_i = cos(\phi_i) then implicitly we are assuming J_{ij} cos(\phi_i) cos(\phi_j) 
        # synapses between input and first hidden layer and so we must account for this in learning rule
        # Note :num_hidden indexes hidden layer neurons
        bias_input = np.matmul(input, net.weights_0) # Note weights_0 is the weights from input to hidden layer
        h[:num_hidden] = (net.bias_0 + bias_input).clip(-args.h_clip, args.h_clip)
        K_s[:num_hidden] = net.sync_0.clip(-args.sync_clip, args.sync_clip)


        # Output layer biases
        # Note num_hidden: just indexes output layer neurons
        if target is not None:
            h[num_hidden:] = (net.bias_1 + beta * target).clip(-args.h_clip, args.h_clip) # NOTE WE USE OIM CONVENTIONS SO WE SHOULD USE +h NOT -h IN OIM DYNAMICS FUNCTION CALLS
            K_s[num_hidden:] = (net.sync_1 - beta/4).clip(-args.sync_clip, args.sync_clip) # Note we also need to nudge the syncs to get the MSE loss functional form correct
        else:
            # Free phase: no beta terms
            h[num_hidden:] = net.bias_1.clip(-args.h_clip, args.h_clip)
            K_s[num_hidden:] = net.sync_1.clip(-args.sync_clip, args.sync_clip)


        ### WEIGHTS
        # Weights between hidden and output layers
        weights = net.weights_1.clip(-args.J_clip, args.J_clip) # Note we use weights_1 for hidden-output weights (and weights_0 for input-hidden weights are used above already)

        # Fill in the J matrix with symmetric weights
        J[:num_hidden, num_hidden:] = weights
        J[num_hidden:, :num_hidden] = weights.T


        return h, J, K_s

def create_batch_problems(net, args, data, target=None, beta=0.0):
    """Helper function to create batch of Ising problems.
    
    Args:
        net: Network object
        args: Arguments object
        data: Batch of input data
        target: Batch of target data (optional)
        beta: Nudging strength (default=0.0)
        
    Returns:
        h_batch, J_batch, K_s_batch: Lists of Ising parameters
    """
    batch_size = data.size()[0]
    J_batch = []
    h_batch = []
    K_s_batch = []
    
    for k in range(batch_size):
        h, J, K_s = createIsingProblem(
            net, args, data[k].numpy(),
            beta=beta,
            target=target[k].numpy() if target is not None else None
        )
        h_batch.append(h)
        J_batch.append(J)
        K_s_batch.append(K_s)
    
    return h_batch, J_batch, K_s_batch

def solve_batch_oim(J_batch, h_batch, K_s_batch, args, u0_batch=None, batch_size=None, phase="Free", target=None):
    """Helper function to solve batch of OIM problems."""
    # Process in smaller chunks to reduce memory pressure
    MAX_CHUNK_SIZE = args.max_chunk_size  # Process at most 20 samples at a time
    
    if batch_size is None:
        batch_size = len(J_batch)
    
    results = []
    for start_idx in range(0, batch_size, MAX_CHUNK_SIZE):
        end_idx = min(start_idx + MAX_CHUNK_SIZE, batch_size)
        
        # Get target index for this chunk
        chunk_target_idx = None
        if target is not None and len(target) > start_idx:
            # Get target index from the first item's target (which is a one-hot encoding)
            # NOTE: This assumes we're using one output neuron per class (i.e., args.layersList[2] == 10 for MNIST)
            # If using multiple output neurons per class, this would need to be modified
            chunk_target_idx = torch.where(target[start_idx] > 0)[0][0]
        
        # Process chunk
        chunk_results = batch_solve_oim(
            J_batch[start_idx:end_idx],
            None if h_batch is None else h_batch[start_idx:end_idx],
            None if K_s_batch is None else K_s_batch[start_idx:end_idx],
            None if u0_batch is None else u0_batch[start_idx:end_idx],
            duration=args.oim_duration,
            dt=args.oim_dt,
            n_procs=args.n_procs,
            plot=(args.debug==1 and start_idx==0),  # Only plot first chunk if debug is on
            phase=phase,
            output_start_idx=args.layersList[1],
            random_initialization=args.oim_random_initialization==1,
            target_idx=chunk_target_idx
        )
        
        results.extend(chunk_results)
        gc.collect()  # Clean up after each chunk
        
    return results












########## TRAINING AND TESTING FUNCTIONS ##########

def train(net, args, train_loader):
    '''Train the network for 1 epoch using batched OIM dynamics'''
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Training")
        for DATA, TARGET in progressbars(train_loader, position=0, leave=True):
            # Force garbage collection at start of each batch
            gc.collect()
            
            batch_size = DATA.size()[0]
            
            # Split into layers
            hidden_slice = slice(None, args.layersList[1])  # Slice for hidden layer
            output_slice = slice(args.layersList[1], None)  # Slice for output layer

            ### FREE PHASE
            # Create and solve free phase problems
            h_batch, J_batch, K_s_batch = create_batch_problems(net, args, DATA)
            free_samples = solve_batch_oim(J_batch, h_batch, K_s_batch, args, batch_size=batch_size, phase="Free", target=TARGET)
            
            # Clear memory immediately after use
            del h_batch, J_batch, K_s_batch
            gc.collect()

            ### POSITIVE AND NEGATIVE BETA NUDGE PHASES
            # Get initial spins if not reinitialising
            initial_spins = None if args.nudge_reinitialisation else free_samples

            # Create and solve positive beta problems
            h_batch, J_batch, K_s_batch = create_batch_problems(net, args, DATA, TARGET, beta=args.beta)
            positive_beta_samples = solve_batch_oim(
                J_batch, h_batch, K_s_batch, args,
                u0_batch=initial_spins,
                batch_size=batch_size,
                phase="Positive",
                target=TARGET
            )
            
            # Clear memory immediately
            del h_batch, J_batch, K_s_batch
            gc.collect()
            
            # Create and solve negative beta problems
            h_batch, J_batch, K_s_batch = create_batch_problems(net, args, DATA, TARGET, beta=-args.beta)
            negative_beta_samples = solve_batch_oim(
                J_batch, h_batch, K_s_batch, args,
                u0_batch=initial_spins,
                batch_size=batch_size,
                phase="Negative",
                target=TARGET
            )
            
            # Clear memory
            del h_batch, J_batch, K_s_batch
            if not args.nudge_reinitialisation:
                del initial_spins
            gc.collect()
            
            # Process results
            store_free = np.stack(free_samples)
            store_pos = np.stack(positive_beta_samples)
            store_neg = np.stack(negative_beta_samples)
            
            # Clear original arrays after stacking
            del free_samples, positive_beta_samples, negative_beta_samples
            gc.collect()
            
            seq = [store_free[:,hidden_slice], store_free[:,output_slice]]
            s_pos = [store_pos[:,hidden_slice], store_pos[:,output_slice]]
            s_neg = [store_neg[:,hidden_slice], store_neg[:,output_slice]]

            # Debug printing
            if args.debug:
                # Print input hidden and output states for random example
                k = np.random.randint(0, batch_size)
                print(f"Example {k} of {batch_size}")

                print(f"Input: {DATA[k].numpy()}")


                print("\n Example Hidden state values:")
                print("Free phase hidden:", np.cos(seq[0][k]))
                print("Positive phase hidden:", np.cos(s_pos[0][k])) 
                print("Negative phase hidden:", np.cos(s_neg[0][k]))
                print("Difference:", np.cos(s_pos[0][k]) - np.cos(s_neg[0][k]))

                print("\n Maximum and minimum values of hidden layer weights, biases and syncs:")
                print("Hidden weights:", np.max(net.weights_0), np.min(net.weights_0))
                print("Hidden biases:", np.max(net.bias_0), np.min(net.bias_0))
                print("Hidden syncs:", np.max(net.sync_0), np.min(net.sync_0))

                # Calculate percentage of hidden neurons that round to ±0.99
                hidden_vals = np.cos(seq[0][k])
                rounded = np.round(hidden_vals, decimals=2)
                num_extreme = np.sum(np.abs(rounded) >= 0.99)
                pct_extreme = (num_extreme / len(hidden_vals)) * 100
                print(f"\nPercentage of hidden neurons at ±0.99: {pct_extreme:.1f}%")

                print("\n Example Output state values:")
                print("Free phase output:", np.cos(seq[1][k]))
                print("Positive phase output:", np.cos(s_pos[1][k]))
                print("Negative phase output:", np.cos(s_neg[1][k]))
                print("Difference:", np.cos(s_pos[1][k]) - np.cos(s_neg[1][k]))
                print(f"\n Target: {TARGET[k].numpy()}")

                print("\n Maximum and minimum values of output layer weights, biases and syncs:")
                print("Output weights:", np.max(net.weights_1), np.min(net.weights_1))
                print("Output biases:", np.max(net.bias_1), np.min(net.bias_1))
                print("Output syncs:", np.max(net.sync_1), np.min(net.sync_1))

                # Calculate percentage of output neurons that round to ±0.99
                output_vals = np.cos(seq[1][k])
                rounded = np.round(output_vals, decimals=2)
                num_extreme = np.sum(np.abs(rounded) >= 0.99)
                pct_extreme = (num_extreme / len(output_vals)) * 100
                print(f"\nPercentage of output neurons at ±0.99: {pct_extreme:.1f}%")

            # Clear stacked arrays after slicing
            del store_free, store_pos, store_neg
            gc.collect()

            # Compute loss and update
            loss, pred = net.computeLossAcc(seq, TARGET, args)
            total_pred += pred
            total_loss += loss

            net.updateParams(DATA, s_pos, s_neg, args)

            # Final cleanup
            del seq, s_pos, s_neg
            del loss, pred, DATA, TARGET
            gc.collect()

    print(f"Total Loss: {total_loss:.4f} (Normalised: {total_loss/len(train_loader.dataset):.4f})")
    print(f"Total Pred: {total_pred} (Accuracy: {100*total_pred/len(train_loader.dataset):.2f}%)")
    return total_loss, total_pred




def test(net, args, test_loader):
    '''Test the network using batched OIM dynamics'''
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Testing")
        for data, target in progressbars(test_loader, position=0, leave=True):
            batch_size = data.size()[0]
            
            # Create and solve problems
            h_batch, J_batch, K_s_batch = create_batch_problems(net, args, data)
            results = solve_batch_oim(J_batch, h_batch, K_s_batch, args, batch_size=batch_size)
            
            # Process results
            store_results = np.stack(results)
            hidden_slice = slice(None, args.layersList[1])
            output_slice = slice(args.layersList[1], None)
            seq = [store_results[:,hidden_slice], store_results[:,output_slice]]
            
            # Compute metrics
            loss, pred = net.computeLossAcc(seq, target, args)
            total_pred += pred
            total_loss += loss

            # Cleanup
            del h_batch, J_batch, K_s_batch, results, store_results, seq
            del loss, pred, data, target
            gc.collect()

    print(f"Total Loss: {total_loss:.4f} (Normalised: {total_loss/len(test_loader.dataset):.4f})")
    print(f"Total Pred: {total_pred} (Accuracy: {100*total_pred/len(test_loader.dataset):.2f}%)")
    return total_loss, total_pred