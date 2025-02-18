module OIMDynamics

using LinearAlgebra
using Base.Threads
using Random
using Plots  # For visualization
using DifferentialEquations
using Statistics  # Add this for mean function

# Print detailed thread information at module load
println("\nJulia OIM solver initialized:")
println("- Total threads available: $(Threads.nthreads())")
println("- Thread IDs: $(collect(1:Threads.nthreads()))")
println("- Physical cores: $(div(Sys.CPU_THREADS, 2))")
println("- Total CPU threads: $(Sys.CPU_THREADS)\n")

export solve_batch_oim, solve_single_oim, test_oim_dynamics

"""
Compute OIM dynamics for a single state vector.
Optimized for maximum performance with SIMD, fastmath, and in-place operations.
"""
@fastmath function compute_dynamics!(du::Vector{Float64}, u::Vector{Float64}, 
                         J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, 
                         K_s::Union{Vector{Float64},Nothing})
    n = length(u)
    fill!(du, 0.0)
    
    # Compute oscillator coupling terms with optimized loops
    @inbounds for i in 1:n
        du_sum = 0.0
        @simd for j in 1:n
            if J[i,j] != 0  # Skip zero couplings
                du_sum -= J[i,j] * sin(u[i] - u[j])
            end
        end
        du[i] = du_sum
    end
    
    # External field (if provided)
    if !isnothing(h)
        @inbounds @simd for i in 1:n
            du[i] -= h[i] * sin(u[i])
        end
    end
    
    # SHIL sync (if provided)
    if !isnothing(K_s)
        @inbounds @simd for i in 1:n
            du[i] -= K_s[i] * sin(2 * u[i])
        end
    end
    
    return du
end

"""
Euler integration with fixed time step and history tracking.
"""
function euler_integration_with_history(u::Vector{Float64}, J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, K_s::Union{Vector{Float64},Nothing}, dt::Float64, duration::Float64)
    n_steps = Int(round(duration/dt)) + 1
    t_history = collect(range(0, duration, length=n_steps))
    u_history = zeros(length(u), n_steps)
    u_history[:, 1] = u
    t = 0.0
    step = 1
    du = similar(u)
    while t < duration
        compute_dynamics!(du, u, J, h, K_s)
        @. u += dt * du
        t += dt
        step += 1
        if step <= n_steps
            u_history[:, step] = u
        end
    end
    return t_history, u_history
end

"""
RK integration using DifferentialEquations.jl with adaptive time steps and history tracking.
"""
function rk_integration_with_history(u::Vector{Float64}, J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, K_s::Union{Vector{Float64},Nothing}, dt::Float64, duration::Float64)
    # Package parameters into a named tuple
    p = (J=J, h=h, K_s=K_s)
    
    # Pre-allocate the derivative vector
    du = similar(u)
    
    function f_de(du::Vector{Float64}, u::Vector{Float64}, p, t::Float64)
        compute_dynamics!(du, u, p.J, p.h, p.K_s)
    end
    
    tspan = (0.0, duration)
    prob = ODEProblem(f_de, u, tspan, p)
    
    # Option 1: BS3 with looser tolerances
    # sol = solve(prob, BS3(), dt=dt, abstol=4e-2, reltol=2e-2, saveat=dt, adaptive=true, dtmax=dt*10)
    
    # Option 2: BS3 with tighter tolerances
    sol = solve(prob, BS3(), dt=dt, abstol=1e-2, reltol=1e-2, saveat=dt, adaptive=true, dtmax=dt*5)
    
    # Option 3: Tsit5 with tighter tolerances
    # sol = solve(prob, Tsit5(), dt=dt, abstol=1e-2, reltol=1e-2, saveat=dt, adaptive=true, dtmax=dt*5)
    
    t_history = sol.t
    u_history = Array(sol)
    return t_history, u_history
end

"""
Fast Euler integration without saving history.
"""
function euler_integration(u::Vector{Float64}, J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, K_s::Union{Vector{Float64},Nothing}, dt::Float64, duration::Float64)
    t = 0.0
    du = similar(u)
    while t < duration
        compute_dynamics!(du, u, J, h, K_s)
        @. u += dt * du
        t += dt
    end
    return u
end

"""
Fast RK integration using DifferentialEquations.jl without saving history.
"""
function rk_integration(u::Vector{Float64}, J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, K_s::Union{Vector{Float64},Nothing}, dt::Float64, duration::Float64)
    p = (J=J, h=h, K_s=K_s)
    
    # Pre-allocate the derivative vector
    du = similar(u)
    
    function f_de(du::Vector{Float64}, u::Vector{Float64}, p, t::Float64)
        compute_dynamics!(du, u, p.J, p.h, p.K_s)
    end
    
    tspan = (0.0, duration)
    prob = ODEProblem(f_de, u, tspan, p)
    
    # Using BS3 with tighter tolerances and increased dtmax multiplier for better adaptivity
    sol = solve(prob, BS3(), dt=dt, abstol=1e-2, reltol=1e-2, save_everystep=false, save_start=false, adaptive=true, dtmax=dt*50)
    
    return sol.u[end]
end

"""
Solve OIM dynamics and track the full evolution.
"""
function solve_single_oim_with_history(J::Matrix{Float64}, 
                                     h::Union{Vector{Float64},Nothing},
                                     K_s::Union{Vector{Float64},Nothing};
                                     u0::Union{Vector{Float64},Nothing}=nothing,
                                     duration::Float64=20.0,
                                     dt::Float64=0.1,
                                     random_initialization::Bool=true,
                                     ode_solver::String="Euler")
    n = size(J, 1)

    # Initialize state
    if isnothing(u0)
        if random_initialization
            # Set fixed seed for reproducibility
            Random.seed!(42)
            u = rand(n) * 2π
        else
            u = fill(π/2, n)
        end
    else
        u = copy(u0)
    end
    
    if ode_solver == "Euler"
        t_history, u_history = euler_integration_with_history(u, J, h, K_s, dt, duration)
    elseif ode_solver == "RK"
        t_history, u_history = rk_integration_with_history(u, J, h, K_s, dt, duration)
    else
        error("Unknown ode_solver option: $ode_solver")
    end
    
    return u, t_history, u_history
end

"""
Solve OIM dynamics for a single problem using Euler integration.
"""
function solve_single_oim(J::Matrix{Float64}, 
                        h::Union{Vector{Float64},Nothing}=nothing,
                        K_s::Union{Vector{Float64},Nothing}=nothing;
                        u0::Union{Vector{Float64},Nothing}=nothing,
                        duration::Float64=20.0,
                        dt::Float64=0.1,
                        make_plot::Bool=false,
                        phase::String="Free",
                        output_start_idx::Int=120,
                        random_initialization::Bool=true,
                        target_idx=nothing,
                        ode_solver::String="Euler")
    
    if !make_plot
        # Fast path without plotting - use fast integration without saving history
        n = size(J, 1)
        
        # Initialize state
        if isnothing(u0)
            if random_initialization
                # Set fixed seed for reproducibility
                Random.seed!(42)
                u = rand(n) * 2π  # Random initial conditions between 0 and 2π
            else
                u = fill(π/2, n)  # Fixed initial conditions
            end
        else
            u = copy(u0)
        end
        if ode_solver == "Euler"
            u_final = euler_integration(u, J, h, K_s, dt, duration)
        elseif ode_solver == "RK"
            u_final = rk_integration(u, J, h, K_s, dt, duration)
        else
            error("Unknown ode_solver option: $ode_solver")
        end
        return u_final
    else
        # Path with plotting enabled - use solve_single_oim_with_history
        u, t_history, u_history = solve_single_oim_with_history(
            J, h, K_s,
            u0=u0,
            duration=duration,
            dt=dt,
            random_initialization=random_initialization,
            ode_solver=ode_solver
        )
        
        # Create plot using the existing plot_dynamics function
        plot_dynamics(t_history, u_history, 
                     phase=phase, 
                     output_start_idx=output_start_idx,
                     target_idx=target_idx)
        return u
    end
end

"""
Calculate phase velocities for the current state.
"""
function calculate_phase_velocities(u::Vector{Float64}, J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, K_s::Union{Vector{Float64},Nothing})
    du = similar(u)
    compute_dynamics!(du, u, J, h, K_s)
    return du
end

"""
Calculate various metrics to assess convergence of the dynamics.
"""
function get_convergence_metrics(u::Vector{Float64}, J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, K_s::Union{Vector{Float64},Nothing})
    velocities = calculate_phase_velocities(u, J, h, K_s)
    return Dict(
        "max_velocity" => maximum(abs.(velocities)),
        "mean_velocity" => mean(abs.(velocities)),
        "rms_velocity" => sqrt(mean(velocities.^2))
    )
end

"""
Solve OIM dynamics for a batch of problems in parallel.
Now returns both final states and convergence metrics.
"""
function solve_batch_oim(J_batch::Vector{Matrix{Float64}},
                        h_batch::Union{Vector{Vector{Float64}},Nothing}=nothing,
                        K_s_batch::Union{Vector{Vector{Float64}},Nothing}=nothing;
                        u0_batch::Union{Vector{Vector{Float64}},Nothing}=nothing,
                        duration::Float64=20.0,
                        dt::Float64=0.1,
                        n_procs::Union{Int,Nothing}=nothing,
                        make_plot::Bool=false,
                        phase::String="Free",
                        output_start_idx::Int=120,
                        random_initialization::Bool=true,
                        target_idx::Union{Int,Nothing}=nothing,
                        ode_solver::String="Euler")
    
    n_problems = length(J_batch)
    results = Vector{Vector{Float64}}(undef, n_problems)
    metrics = Vector{Dict{String,Float64}}(undef, n_problems)
    
    # Determine number of threads to use
    max_threads = Threads.nthreads()
    n_threads = isnothing(n_procs) ? max_threads : min(n_procs, max_threads)
    
    # Process problems in chunks based on number of threads
    chunk_size = ceil(Int, n_problems / n_threads)
    
    # Process all problems in parallel using threads
    @threads for chunk in 1:n_threads
        start_idx = (chunk - 1) * chunk_size + 1
        end_idx = min(chunk * chunk_size, n_problems)
        
        for i in start_idx:end_idx
            h_i = isnothing(h_batch) ? nothing : h_batch[i]
            K_s_i = isnothing(K_s_batch) ? nothing : K_s_batch[i]
            u0_i = isnothing(u0_batch) ? nothing : u0_batch[i]
            
            # Solve single problem
            results[i] = solve_single_oim(
                J_batch[i],
                h_i,
                K_s_i,
                u0=u0_i,
                duration=duration,
                dt=dt,
                make_plot=make_plot && i == 1, # Only plot the first problem in debug mode
                phase=phase,
                output_start_idx=output_start_idx,
                random_initialization=random_initialization,
                target_idx=target_idx,
                ode_solver=ode_solver
            )
            
            # Calculate convergence metrics
            metrics[i] = get_convergence_metrics(results[i], J_batch[i], h_i, K_s_i)
        end
    end
    
    return results, metrics
end

"""
Plot phase evolution over time and save to file.
"""
function plot_dynamics(t::Vector{Float64}, phases::Matrix{Float64}; 
                      title::String="Phase Evolution",
                      phase::String="Free",
                      savepath::String="oim_dynamics_plot.png",
                      output_start_idx::Int=120,  # Default for 120 hidden, 10 output
                      target_idx=nothing)  # Accept any type and convert inside
    n_phases = size(phases, 1)
    p = plot()  # Create empty plot
    
    # Convert target_idx to Int if it's not nothing
    target_idx_int = nothing
    if !isnothing(target_idx)
        target_idx_int = convert(Int, target_idx)
    end
    
    # Plot hidden layer phases with default colors
    plot!(p, t, phases[1:output_start_idx,:]', 
         label=nothing,
         alpha=0.7,
         linewidth=1)  # Thinner lines for hidden neurons
    
    # Plot output layer phases with distinct styles
    if output_start_idx < n_phases
        # Define a set of vibrant colors for output neurons
        output_colors = [:black, :red, :blue, :green, :purple, :orange, :cyan, :magenta, :brown, :gold]
        line_styles = [:dash]
        
        for (i, output_idx) in enumerate(output_start_idx+1:n_phases)
            color_idx = mod1(i, length(output_colors))
            style_idx = mod1(i, length(line_styles))
            
            # If this is the target neuron, give it a special color and label
            if !isnothing(target_idx_int) && (i-1) == target_idx_int
                plot!(p, t, phases[output_idx:output_idx,:]',
                     label="Target (Class $target_idx_int)",
                     alpha=1.0,
                     linewidth=3,
                     color=:lime,  # Neon green for target
                     linestyle=:solid)
            else
                plot!(p, t, phases[output_idx:output_idx,:]',
                     label=nothing,
                     alpha=1.0,
                     linewidth=3,
                     color=output_colors[color_idx],
                     linestyle=line_styles[style_idx])
            end
        end
    end
    
    # Set plot attributes
    plot!(p, title="$phase Phase Evolution",
         xlabel="Time",
         ylabel="Phase",
         legend=:right)  # Show legend inside on the right
    
    # Modify filename to include phase
    base, ext = splitext(savepath)
    savepath = "$(base)_$(lowercase(phase))$(ext)"
    savefig(p, savepath)
    return p
end

"""
Test the OIM dynamics solver on a simple example.
"""
function test_oim_dynamics(; n_spins::Int=10, plot_result::Bool=true)
    println("\nRunning OIM dynamics test:")
    println("- Number of spins: $n_spins")
    
    # Create a simple coupling matrix 
    J = zeros(n_spins, n_spins)
    for i in 1:n_spins-1
        J[i,i+1] = J[i+1,i] = -1.0
    end
    println("- Created nearest-neighbor coupling matrix")
    
    # Create simple external field and SHIL sync
    h = fill(0.0, n_spins)
    K_s = fill(10.0, n_spins)
    println("- Created uniform external field and SHIL sync vectors")
    
    # Random initial conditions
    u0 = rand(n_spins) * 2π
    println("- Generated random initial conditions")
    
    # Solve with history tracking
    println("- Solving dynamics...")
    t, history = solve_single_oim_with_history(
        J, h, K_s,
        u0=u0,
        duration=20.0,
        dt=0.1
    )
    
    # Print results
    println("- Initial phases: ", round.(u0, digits=3))
    println("- Final phases: ", round.(history[:,end], digits=3))
    
    # Plot if requested
    if plot_result
        println("- Plotting results...")
        plot_dynamics(t, history, title="Test Case: $n_spins-Spin Chain")
    end
    
    return t, history
end

end # module 