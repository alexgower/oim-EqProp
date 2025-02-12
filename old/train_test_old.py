from tqdm import tqdm as progressbars

import numpy as np 

import torch
from scipy import*
from copy import*

import dimod






########## BINARY QUADRATIC MODEL (BQM) FUNCTIONS ##########

def createIsingProblem(net, args, input, beta=0.0, target=None, simulation_type=0):
    """Create Ising problem in format for simulation dynamics to accept"""
    with torch.no_grad():
        ### BIASES
        # Use bias trick to encode the input layer
        # BUT THIS ONLY WORKS FOR E_{ij} = J_ij * s_i * s_j product form
        # For E_{input, hidden} = (\sum_j) (\sum_i [input]_i * J_{ij}) * [hidden]_j = (\sum_j) ([input] \cdot J_{ij}) * [hidden]_j

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

        # Number of spins/units
        num_hidden = args.layersList[1]
        num_output = args.layersList[2]
        num_spins = num_hidden + num_output

        # Handle different simulation types
        if simulation_type == 0:  # OIM annealing
            # Initialize J as a zero matrix
            J = np.zeros((num_spins, num_spins))
            # Fill in the J matrix with symmetric weights
            J[:num_hidden, num_hidden:] = weights
            J[num_hidden:, :num_hidden] = weights.T
            return h, J
            # TODO NEED TO ADD BETA TERM TO K_S IF WANT CONTINUOUS OIM TO BE MINIMISING MEAN SQUARED ERROR FOR cos(\phi) OUTPUT

        elif simulation_type == 1:  # Simulated annealing
            # Generate h and J as dictionaries
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

        elif simulation_type == 2:  # Scellier
            # Initialize J as a zero matrix
            J = np.zeros((num_spins, num_spins))
            # Fill in the J matrix with symmetric weights
            J[:num_hidden, num_hidden:] = weights
            J[num_hidden:, :num_hidden] = weights.T
            # For Scellier we also need to pass beta, target and n_hidden
            y_target = target if (target is not None and beta > 0) else None
            return h, J

        else:
            raise ValueError("Invalid simulation type")












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
                    h, J = createIsingProblem(net, args, data, beta=0.0, target=None, simulation_type=2)
    
                    # Use Scellier solver for free phase
                    best_seq_sample, _ = sampler.scellier_problem_solver(
                        J, args.oim_duration, args.oim_dt,
                        h=h,
                        noise=args.oim_noise==1,
                        runs=args.n_iter_free,
                        n_hidden=args.layersList[1],
                        beta=0.0,
                        y_target=None
                    )
                    best_seq_sample_to_store = best_seq_sample
                    del best_seq_sample  # Cleanup
                else: 
                    print("Invalid simulation type")
                    raise ValueError("Invalid simulation type")
                    

                ### NUDGE PHASE

                if np.array_equal(best_seq_sample_to_store.reshape(1,-1)[:,args.layersList[1]:][0], target):
                    best_s_sample_to_store = best_seq_sample_to_store
                else:
                    if args.simulation_type == 0: # OIM annealing
                        h, J = createIsingProblem(net, args, data, beta=args.beta, target=target, simulation_type=args.simulation_type)


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
                        h, J = createIsingProblem(net, args, data, beta=args.beta, target=target, simulation_type=2)
                        
                        best_s_sample, _ = sampler.scellier_problem_solver(
                            J, args.oim_duration, args.oim_dt,
                            h=h,
                            noise=args.oim_noise==1,
                            initial_spins=best_seq_sample_to_store,
                            runs=args.n_iter_nudge,
                            n_hidden=args.layersList[1],
                            beta=args.beta,
                            y_target=target
                        )
                        best_s_sample_to_store = best_s_sample
                        del best_s_sample  # Cleanup
                    
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

            # Simulated sampling
            if args.simulation_type == 0: # OIM annealing
                h, J = createIsingProblem(net, args, data, simulation_type=args.simulation_type)

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
                h, J = createIsingProblem(net, args, data, beta=0.0, target=None, simulation_type=2)
    
                actual_seq, _ = sampler.scellier_problem_solver(
                    J, args.oim_duration, args.oim_dt,
                    h=h,
                    noise=args.oim_noise==1,
                    runs=args.n_iter_free,
                    n_hidden=args.layersList[1],
                    beta=0.0,
                    y_target=None
                )
                
                # Reshaping
                actual_seq = actual_seq.reshape(1, actual_seq.shape[0])

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

def train_oim_julia_batch_parallel(net, args, train_loader, Simulations):
    '''
    function to train the network for 1 epoch using Julia's batch parallel solver,
    with memory optimization
    '''
    import gc  # Add garbage collector import
    total_pred, total_loss = 0, 0

    with torch.no_grad():
        print("Training")
        
        for idx, (DATA, TARGET) in enumerate(progressbars(train_loader, position=0, leave=True)):
            # Force garbage collection at start of batch
            gc.collect()
            batch_size = DATA.size()[0]
            
            ### FREE PHASE
            if args.simulation_type == 0:  # OIM Dynamcis (note SA just uses train function)
                # Pre-allocate batch data arrays
                J_batch = []
                h_batch = []
                for k in range(batch_size):
                    h, J = createIsingProblem(net, args, DATA[k].numpy(), simulation_type=0)
                    J_batch.append(-2*J)
                    h_batch.append(-h)
                    del h, J  # Cleanup intermediates
                
                # Select dynamics functions
                if args.oim_dynamics == 0:
                    dynamics = Simulations.wang_oim_dynamics
                    stochastic_dynamics = Simulations.wang_oim_stochastic_dynamics
                elif args.oim_dynamics == 1:
                    dynamics = Simulations.simple_oim_dynamics
                    stochastic_dynamics = Simulations.simple_oim_stochastic_dynamics
                    
                # Batch solve free phase
                best_seq_configs, best_seq_energies = Simulations.oim_batch_parallel_problem_solver(
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

            elif args.simulation_type == 2:  # Scellier case
                # Pre-allocate batch data arrays 
                J_batch = []
                h_batch = []
                y_target_batch = []

                # Create problem parameters for each item in batch
                for k in range(batch_size):
                    h, J = createIsingProblem(net, args, DATA[k].numpy(), beta=0.0, target=None, simulation_type=2)
                    J_batch.append(J)  # No -2 multiplication needed
                    h_batch.append(h)
                    y_target_batch.append(None)  # No target for free phase
                    
                # Batch solve free phase
                best_seq_configs, best_seq_energies = Simulations.scellier_batch_parallel_problem_solver(
                    J_batch, h_batch,
                    duration=args.oim_duration,
                    timestep=args.oim_dt,
                    noise=args.oim_noise==1,
                    runs=args.n_iter_free,
                    n_hidden=args.layersList[1],
                    beta=0.0,
                    y_target_batch=y_target_batch
                )

                
            # Convert to list and cleanup
            best_seq_samples = list(best_seq_configs)

            if args.verbose == 1:
                print(f"Free Phase: {best_seq_energies}")

            del best_seq_configs, best_seq_energies, J_batch, h_batch

            ### NUDGE PHASE
            # Pre-allocate arrays for nudge phase
            need_nudge = []
            best_s_samples = [None] * batch_size  # Pre-allocate full result array
            J_batch = []
            h_batch = []
            initial_spins = []
            if args.simulation_type == 2:
                y_target_batch = []
            
            # Determine which samples need nudging
            for k in range(batch_size):
                data, target = DATA[k].numpy(), TARGET[k].numpy()
                
                if np.array_equal(best_seq_samples[k].reshape(1,-1)[:,args.layersList[1]:][0], target):
                    best_s_samples[k] = best_seq_samples[k]
                else:
                    need_nudge.append(k)

                    if args.simulation_type == 0:
                        h, J = createIsingProblem(net, args, data, beta=args.beta, target=target, simulation_type=0)
                        J_batch.append(-2*J)
                        h_batch.append(-h)
                        initial_spins.append(best_seq_samples[k])
                    

                    elif args.simulation_type == 2:
                        h, J = createIsingProblem(net, args, data, beta=args.beta, target=target, simulation_type=2)
                        J_batch.append(J)
                        h_batch.append(h)
                        initial_spins.append(best_seq_samples[k])
                        y_target_batch.append(target)

                    del h, J  # Cleanup

            # Only run nudge phase if needed
            if need_nudge:

                if args.simulation_type == 0: # OIM Dynamics (note SA just uses train function)

                    best_s_configs, best_s_energies = Simulations.oim_batch_parallel_problem_solver(
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

                elif args.simulation_type == 2: # Scellier dynamics
                    # Batch solve nudge phase for Scellier
                    best_s_configs, best_s_energies = Simulations.scellier_batch_parallel_problem_solver(
                        J_batch, h_batch,
                        initial_spins_batch=initial_spins,
                        duration=args.oim_duration,
                        timestep=args.oim_dt,
                        noise=args.oim_noise==1,
                        runs=args.n_iter_nudge,
                        n_hidden=args.layersList[1],
                        beta=args.beta,
                        y_target_batch=y_target_batch
                    )
                
                # Insert nudged results back in correct positions
                for idx, batch_idx in enumerate(need_nudge):
                    best_s_samples[batch_idx] = best_s_configs[idx]


                if args.verbose == 1:
                    print(f"Nudge Phase: {best_s_energies}")
                
                # Cleanup
                del best_s_configs, best_s_energies
            
            del J_batch, h_batch, initial_spins, need_nudge
            if args.simulation_type == 2:
                del y_target_batch
            
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

            # Clean up all remaining variables
            del seq, s, loss, pred, DATA, TARGET
            
            # Force garbage collection
            gc.collect()

            # Check if workers need reset
            if should_reset_workers():
                reset_julia_workers(args.procs)
                gc.collect()

    print(f"Total Loss: {total_loss} (Normalised: {total_loss/len(train_loader.dataset)})")
    print(f"Total Pred: {total_pred} (Normalised: {total_pred/len(train_loader.dataset)})")
    return total_loss, total_pred