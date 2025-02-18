import os
import numpy as np
import gc

print("\nInitializing Julia bridge...")

# Set Julia threads before importing Julia and specify the Julia project environment.
os.environ["JULIA_NUM_THREADS"] = str(max(1, os.cpu_count() - 10))  # Leave more CPUs free for Python/system
os.environ["JULIA_PROJECT"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
print("Activated Julia project environment:", os.environ["JULIA_PROJECT"])

# Now import Julia (which will use the activated project)
print("Importing Julia (this may take a few seconds)...")
from julia import Julia, Main
jl = Julia(compiled_modules=False)
print("Julia imported")

# Install required packages if not already installed
print("Checking Julia packages...")
Main.eval('''
    import Pkg
    if !("Plots" in keys(Pkg.project().dependencies))
        Pkg.add("Plots")
    end
''')

# Ensure DifferentialEquations is installed
Main.eval('''
    import Pkg
    if !("DifferentialEquations" in keys(Pkg.project().dependencies))
        println("Installing DifferentialEquations...")
        Pkg.add("DifferentialEquations")
    end
''')
print("Package check completed")

# Get absolute path to the oim_dynamics.jl file
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oim_dynamics.jl")

# Include the module directly
print("Loading OIMDynamics module...")
Main.eval(f'''
    include("{module_path}");
    using .OIMDynamics;
''')
print("Successfully loaded OIMDynamics module from:", module_path)

def batch_solve_oim(J_batch, h_batch=None, K_s_batch=None, u0_batch=None,
                   duration=20.0, dt=0.01, n_procs=None, plot=False, phase="Free", 
                   output_start_idx=120, random_initialization=True, target_idx=None, ode_solver="Euler"):
    """
    Python interface to Julia's OIM solver.
    All inputs will be converted to float64 for Julia compatibility.
    Uses Julia's multi-threading for parallel processing.
    
    Parameters:
    -----------
    J_batch : list of np.ndarray
        List of coupling matrices
    h_batch : list of np.ndarray, optional
        List of external fields
    K_s_batch : list of np.ndarray, optional
        List of SHIL sync vectors
    u0_batch : list of np.ndarray, optional
        List of initial states. If None, uses seeded random initialization between 0 and 2π
        with seed=42 for reproducibility (previously used fixed π/2 initialization)
    duration : float
        Total simulation time
    dt : float
        Time step
    n_procs : int, optional
        Number of threads to use. If None, uses all available threads.
    plot : bool, optional
        Whether to plot the dynamics of the first problem in the batch
    phase : str, optional
        Which phase we're in ("Free", "Positive", or "Negative") for plot title
    output_start_idx : int, optional
        Index where output layer neurons start (default=120 for 120 hidden neurons)
    random_initialization : bool, optional
        Whether to use random initialization (True) or fixed π/2 initialization (False)
    target_idx : int, optional
        Target class index to highlight in the plot (0-based)
    ode_solver : str, optional
        Type of ODE solver to use: "Euler" or "RK" (default: "Euler")
    
    Returns:
    --------
    tuple
        (final_states, convergence_metrics) where:
        - final_states: list of np.ndarray containing final states for each problem
        - convergence_metrics: list of dicts containing convergence metrics for each problem
    """
    try:
        # Pre-allocate lists for converted arrays
        batch_size = len(J_batch)
        J_batch_jl = [None] * batch_size
        h_batch_jl = None if h_batch is None else [None] * batch_size
        K_s_batch_jl = None if K_s_batch is None else [None] * batch_size
        u0_batch_jl = None if u0_batch is None else [None] * batch_size

        # Convert arrays one at a time with explicit garbage collection
        for i in range(batch_size):
            J_batch_jl[i] = np.asarray(J_batch[i], dtype=np.float64, order='C')
            if h_batch is not None:
                h_batch_jl[i] = np.asarray(h_batch[i], dtype=np.float64, order='C')
            if K_s_batch is not None:
                K_s_batch_jl[i] = np.asarray(K_s_batch[i], dtype=np.float64, order='C')
            if u0_batch is not None:
                u0_batch_jl[i] = np.asarray(u0_batch[i], dtype=np.float64, order='C')
            
            if i % 10 == 0:  # Collect garbage periodically
                gc.collect()
        
        # Convert target_idx to Julia Int64 if it's not None
        target_idx_jl = int(target_idx) if target_idx is not None else None
        
        # Call Julia function through Main
        results, metrics = Main.OIMDynamics.solve_batch_oim(
            J_batch_jl,
            h_batch_jl,
            K_s_batch_jl,
            u0_batch=u0_batch_jl,
            duration=duration,
            dt=dt,
            n_procs=n_procs,
            make_plot=plot,
            phase=phase,
            output_start_idx=output_start_idx,
            random_initialization=random_initialization,
            target_idx=target_idx_jl,
            ode_solver=ode_solver
        )
        
        # Convert results and metrics back one at a time
        converted_results = [None] * len(results)
        converted_metrics = [None] * len(metrics)
        for i, (r, m) in enumerate(zip(results, metrics)):
            converted_results[i] = np.array(r, dtype=np.float64, order='C')
            converted_metrics[i] = {
                'max_velocity': float(m['max_velocity']),
                'mean_velocity': float(m['mean_velocity']),
                'rms_velocity': float(m['rms_velocity'])
            }
            if i % 10 == 0:  # Collect garbage periodically
                gc.collect()
                
        return converted_results, converted_metrics
        
    except Exception as e:
        print(f"Error in batch_solve_oim: {str(e)}")
        raise 