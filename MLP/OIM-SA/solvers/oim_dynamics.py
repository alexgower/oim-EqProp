import os

# Set OpenMP environment variable to avoid fork issues
# These must be set before importing numpy or other OpenMP-using libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
from scipy.integrate import solve_ivp
from numba import jit, prange
import warnings
from typing import Optional, Union, List, Tuple
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import time

class OIMDynamics:
    """
    OIM dynamics solver with fixed coupling parameters.
    Supports both CPU (Numba) and GPU (PyTorch) acceleration.
    """
    def __init__(self, 
                 J: Union[np.ndarray, torch.Tensor], 
                 h: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 K_s: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 use_gpu: bool = False):
        """
        Initialize OIM dynamics solver with fixed parameters.
        
        Args:
            J: Coupling matrix (fixed in time)
            h: External field vector (fixed in time)
            K_s: SHIL sync vector (fixed in time)
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Move problem parameters to GPU if requested
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.J = torch.as_tensor(J, device=self.device, dtype=torch.float32)
            self.h = torch.as_tensor(h, device=self.device, dtype=torch.float32) if h is not None else None
            self.K_s = torch.as_tensor(K_s, device=self.device, dtype=torch.float32) if K_s is not None else None
        else:
            self.J = np.asarray(J, dtype=np.float32)
            self.h = np.asarray(h, dtype=np.float32) if h is not None else None
            self.K_s = np.asarray(K_s, dtype=np.float32) if K_s is not None else None
            
        self.n_spins = len(J)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _dynamics_numba(u: np.ndarray, J: np.ndarray, h: Optional[np.ndarray], 
                       K_s: Optional[np.ndarray]) -> np.ndarray:
        """Numba-accelerated OIM dynamics with fixed parameters."""
        n = len(u)
        du = np.zeros_like(u)
        
        # Oscillator coupling - parallelized over spins
        for i in prange(n):
            for j in range(n):
                if J[i,j] != 0:  # Skip zero couplings
                    phase_diff = u[i] - u[j]
                    du[i] -= J[i,j] * np.sin(phase_diff)
        
        # External field
        if h is not None:
            du -= h * np.sin(u)
            
        # SHIL sync
        if K_s is not None:
            du -= K_s * np.sin(2 * u)
        
        return du

    def _dynamics_gpu(self, u: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated OIM dynamics using PyTorch."""
        # Reshape for broadcasting
        u_i = u.view(-1, 1) # column vector
        u_j = u.view(1, -1) # row vector

        # Compute all phase differences at once (i.e. u_i - u_j is a matrix)
        phase_diffs = u_i - u_j
        
        # Oscillator coupling using matrix operations
        du = -torch.sum(self.J * torch.sin(phase_diffs), dim=1)
        
        # External field
        if self.h is not None:
            du -= self.h * torch.sin(u)
            
        # SHIL sync
        if self.K_s is not None:
            du -= self.K_s * torch.sin(2 * u)
        
        return du

    def dynamics(self, t: float, u: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Main dynamics function that routes to appropriate implementation."""
        if self.use_gpu:
            u_tensor = torch.as_tensor(u, device=self.device, dtype=torch.float32)
            return self._dynamics_gpu(u_tensor).cpu().numpy()
        else:
            u_array = np.asarray(u, dtype=np.float32)
            return self._dynamics_numba(u_array, self.J, self.h, self.K_s)

    def solve(self, 
             u0: Union[np.ndarray, torch.Tensor], 
             duration: float, 
             dt: float,
             method: str = 'RK45',
             noise: bool = False,
             noise_strength: float = 0.0,
             plot_dynamics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Solve OIM dynamics.
        
        Args:
            u0: Initial state
            duration: Total simulation time
            dt: Time step
            method: Integration method ('RK45' or 'Euler')
            noise: Whether to add noise
            noise_strength: Strength of noise term
            plot_dynamics: Whether to plot phase evolution
            
        Returns:
            If plot_dynamics=False: Final state
            If plot_dynamics=True: Tuple of (final state, time points, phase evolution)
        """
        if method == 'Euler':
            return self._solve_euler(u0, duration, dt, noise, noise_strength, plot_dynamics)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_ivp(
                    self.dynamics,
                    (0, duration),
                    np.asarray(u0),
                    method=method,
                    t_eval=np.arange(0, duration, dt),
                    rtol=1e-6,
                    atol=1e-6
                )
                
            if plot_dynamics:
                self._plot_dynamics(sol.t, sol.y)
                return sol.y[:,-1], sol.t, sol.y
            
            return sol.y[:,-1]

    def _solve_euler(self, 
                    u0: Union[np.ndarray, torch.Tensor], 
                    duration: float, 
                    dt: float,
                    noise: bool = False,
                    noise_strength: float = 0.0,
                    plot_dynamics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Simple Euler integration with optional noise and plotting."""
        steps = int(duration / dt)
        t = np.arange(0, duration, dt)
        u = np.asarray(u0, dtype=np.float32)
        
        if plot_dynamics:
            u_history = np.zeros((len(u), len(t)))
            u_history[:,0] = u
        
        for i, t_i in enumerate(t[:-1]): # t[:-1] means we don't include the last time point
            du = self.dynamics(t_i, u)
            if noise:
                du += noise_strength * np.random.randn(self.n_spins).astype(np.float32)
            u += dt * du
            
            if plot_dynamics:
                u_history[:,i+1] = u # we calculate u at the next time point and store it
        
        if plot_dynamics:
            self._plot_dynamics(t, u_history)
            return u, t, u_history
            
        return u

    def _plot_dynamics(self, t: np.ndarray, phases: np.ndarray):
        """Plot phase evolution over time for all neurons and save to file."""
        plt.figure(figsize=(12, 6))
        
        # Plot all phases with same style
        for i in range(phases.shape[0]):
            plt.plot(t, phases[i], '-', alpha=0.7)
        
        plt.title('Phase Evolution Over Time')
        plt.xlabel('Time')
        plt.ylabel('Phase')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('oim_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()

def solve_single(args):
    """Helper function to solve a single OIM problem."""
    J, h, K_s, u0, params = args
    solver = OIMDynamics(J, h, K_s, use_gpu=params['use_gpu'])
    if u0 is None:
        u0 = np.full(len(J), np.pi/2, dtype=np.float32) # Note if u0 is None then we use the free phase initial conditions ie. u0 = [pi/2, pi/2, ...]
        
    final_state = solver.solve(
        u0, 
        params['duration'], 
        params['dt'], 
        params['method'], 
        params['noise'], 
        params['noise_strength'],
        plot_dynamics=params['plot_dynamics']
    )
    
    if params['plot_dynamics']:
        final_state = final_state[0]
        
    return final_state

def batch_solve_oim(J_batch: List[np.ndarray],
                   h_batch: Optional[List[np.ndarray]] = None,
                   K_s_batch: Optional[List[np.ndarray]] = None,
                   u0_batch: Optional[List[np.ndarray]] = None,
                   duration: float = 20.0,
                   dt: float = 0.1,
                   use_gpu: bool = False,
                   method: str = 'RK45',
                   noise: bool = False,
                   noise_strength: float = 0.0,
                   batch_size: int = 20,
                   plot_dynamics: bool = False,
                   n_procs: Optional[int] = None) -> List[np.ndarray]:
    """
    Solve OIM dynamics for a batch of problems in parallel.
    Processes in smaller chunks to manage memory.
    
    Args:
        J_batch: List of coupling matrices
        h_batch: List of external fields (optional)
        K_s_batch: List of SHIL sync vectors (optional)
        u0_batch: List of initial states (optional)
        duration: Total simulation time
        dt: Time step
        use_gpu: Whether to use GPU acceleration
        method: Integration method
        noise: Whether to add noise
        noise_strength: Strength of noise term
        batch_size: Size of chunks to process
        plot_dynamics: Whether to plot phase evolution for each problem
        n_procs: Number of processors to use for parallel processing. If None, uses (cpu_count - 1)
        
    Returns:
        List of final states
    """
    n_problems = len(J_batch)
    results = []
    
    # Determine number of processors to use
    if n_procs is None:
        n_procs = max(1, mp.cpu_count() - 1)
    
    # Shared parameters dictionary
    params = {
        'duration': duration,
        'dt': dt,
        'use_gpu': use_gpu,
        'method': method,
        'noise': noise,
        'noise_strength': noise_strength,
        'plot_dynamics': plot_dynamics
    }
    
    # Process in batches
    for i in range(0, n_problems, batch_size):
        batch_end = min(i + batch_size, n_problems)
        batch_J = J_batch[i:batch_end]
        batch_h = h_batch[i:batch_end] if h_batch is not None else [None] * (batch_end - i)
        batch_K_s = K_s_batch[i:batch_end] if K_s_batch is not None else [None] * (batch_end - i)
        batch_u0 = u0_batch[i:batch_end] if u0_batch is not None else [None] * (batch_end - i)
        
        # Create argument tuples for parallel processing
        args_list = [(J, h, K_s, u0, params) for J, h, K_s, u0 in zip(batch_J, batch_h, batch_K_s, batch_u0)]
        
        # Process batch in parallel
        with mp.Pool(n_procs) as pool:
            batch_results = pool.map(solve_single, args_list)
            results.extend(batch_results)
        
        # Force cleanup
        if use_gpu:
            torch.cuda.empty_cache()
    
    return results

def test_oim_dynamics():
    """Run a test of OIM dynamics with a simple example problem."""
    # Create a small test problem
    n_spins = 5
    
    # Create a simple coupling matrix with nearest-neighbor interactions
    J = np.zeros((n_spins, n_spins))
    for i in range(n_spins-1):
        coupling = np.random.choice([-1.0, 1.0])  # Random ±1 coupling
        J[i,i+1] = J[i+1,i] = coupling  # Symmetric coupling between neighbors
        
    # Create a simple external field
    h = np.ones(n_spins) * 0.0
    
    # Create SHIL sync vector
    K_s = np.ones(n_spins) * 0.0
    
    # Create solver
    solver = OIMDynamics(J, h, K_s)
    
    # Initial condition
    u0 = np.random.uniform(0, 2*np.pi, n_spins)
    
    # Solve with plotting
    final_state = solver.solve(
        u0=u0,
        duration=20.0,
        dt=0.1,
        method='Euler',
        noise=False,
        noise_strength=0.0,
        plot_dynamics=False
    )
    
    print("Initial phases:", u0)
    print("Final phases:", final_state)

def test_batch_oim_dynamics():
    """Test parallel batch processing with multiple example problems."""
    print("\nTesting batch OIM dynamics processing...")
    
    # Create multiple test problems
    n_problems = 12  # Number of test problems
    n_spins = 5     # Size of each problem
    n_procs = 4     # Number of processors to use
    
    print(f"Using {n_procs} processors for parallel processing")
    
    # Initialize batch lists
    J_batch = []
    h_batch = []
    K_s_batch = []
    u0_batch = []
    
    # Create test problems
    for _ in range(n_problems):
        # Create coupling matrix with random nearest-neighbor interactions
        J = np.zeros((n_spins, n_spins))
        for i in range(n_spins-1):
            coupling = np.random.choice([-1.0, 1.0])
            J[i,i+1] = J[i+1,i] = coupling
        J_batch.append(J)
        
        # Create random external fields
        h = np.random.uniform(-0.5, 0.5, n_spins)
        h_batch.append(h)
        
        # Create random SHIL sync vectors
        K_s = np.random.uniform(0, 0.2, n_spins)
        K_s_batch.append(K_s)
        
        # Create random initial conditions
        u0 = np.random.uniform(0, 2*np.pi, n_spins)
        u0_batch.append(u0)
    
    # Test different batch sizes
    batch_sizes = [6, 12]
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch_size = {batch_size}")
        
        # Time the parallel execution
        start_time = time.time()
        results = batch_solve_oim(
            J_batch=J_batch,
            h_batch=h_batch,
            K_s_batch=K_s_batch,
            u0_batch=u0_batch,
            duration=10.0,
            dt=0.1,
            method='Euler',
            noise=False,
            noise_strength=0.0,
            batch_size=batch_size,
            n_procs=n_procs
        )
        end_time = time.time()
        
        print(f"Number of problems solved: {len(results)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Average time per problem: {(end_time - start_time) / n_problems:.2f} seconds")
        
        # Print sample results
        print("\nSample results (first 2 problems):")
        for i in range(min(2, len(results))):
            print(f"Problem {i+1} final phases:", results[i])

def test_deterministic_behavior():
    """Test that identical inputs produce identical outputs across multiple runs."""
    print("\nTesting deterministic behavior...")
    
    # Create a fixed test problem
    n_spins = 5
    
    # Fixed coupling matrix
    J = np.zeros((n_spins, n_spins))
    for i in range(n_spins-1):
        J[i,i+1] = J[i+1,i] = 1.0  # All ferromagnetic couplings
    
    # Fixed external field and SHIL sync
    h = np.ones(n_spins) * 0.5
    K_s = np.ones(n_spins) * 0.1
    
    # Fixed initial condition
    u0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    # Parameters for multiple trials
    n_trials = 5
    results = []
    
    print("Running multiple trials with identical inputs...")
    print(f"Initial phases: {u0}")
    
    # Run multiple trials
    for trial in range(n_trials):
        solver = OIMDynamics(J, h, K_s)
        final_state = solver.solve(
            u0=u0,
            duration=10.0,
            dt=0.1,
            method='Euler',
            noise=False,
            noise_strength=0.0,
            plot_dynamics=False
        )
        results.append(final_state)
        print(f"Trial {trial + 1} final phases: {final_state}")
    
    # Check if all results are identical
    reference = results[0]
    all_identical = all(np.allclose(result, reference, rtol=1e-5) for result in results[1:])
    
    if all_identical:
        print("\nTest PASSED: All trials produced identical results")
    else:
        print("\nTest FAILED: Trials produced different results")
        # Print the maximum difference between any two results
        max_diff = max(np.max(np.abs(result - reference)) for result in results[1:])
        print(f"Maximum difference between trials: {max_diff}")

    # Now test with parallel batch processing
    print("\nTesting deterministic behavior with parallel processing...")
    
    # Create batch with identical problems
    n_problems = 4
    J_batch = [J.copy() for _ in range(n_problems)]
    h_batch = [h.copy() for _ in range(n_problems)]
    K_s_batch = [K_s.copy() for _ in range(n_problems)]
    u0_batch = [u0.copy() for _ in range(n_problems)]
    
    # Run batch processing multiple times
    batch_results = []
    for trial in range(3):
        results = batch_solve_oim(
            J_batch=J_batch,
            h_batch=h_batch,
            K_s_batch=K_s_batch,
            u0_batch=u0_batch,
            duration=10.0,
            dt=0.1,
            method='Euler',
            noise=False,
            noise_strength=0.0,
            batch_size=2,
            n_procs=2
        )
        batch_results.append(results)
        print(f"\nBatch trial {trial + 1} results:")
        for i, result in enumerate(results):
            print(f"Problem {i + 1}: {result}")
    
    # Check if all batch results are identical
    batch_reference = batch_results[0]
    batch_identical = all(
        all(np.allclose(result, ref, rtol=1e-5) 
            for result, ref in zip(trial_results, batch_reference))
        for trial_results in batch_results[1:]
    )
    
    if batch_identical:
        print("\nBatch test PASSED: All parallel trials produced identical results")
    else:
        print("\nBatch test FAILED: Parallel trials produced different results")
        # Print the maximum difference between any two batch results
        max_batch_diff = max(
            max(np.max(np.abs(result - ref)) 
                for result, ref in zip(trial_results, batch_reference))
            for trial_results in batch_results[1:]
        )
        print(f"Maximum difference between parallel trials: {max_batch_diff}")

if __name__ == "__main__":
    import multiprocessing as mp

    # Switch to the 'spawn' start method at the very beginning:
    mp.set_start_method('spawn', force=True)

    test_oim_dynamics()
    test_batch_oim_dynamics()
    test_deterministic_behavior()