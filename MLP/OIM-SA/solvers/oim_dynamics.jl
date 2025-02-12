module OIMDynamics

using LinearAlgebra
using Base.Threads
using Random
using Plots  # For visualization

# Print detailed thread information at module load
println("\nJulia OIM solver initialized:")
println("- Total threads available: $(Threads.nthreads())")
println("- Thread IDs: $(collect(1:Threads.nthreads()))")
println("- Physical cores: $(div(Sys.CPU_THREADS, 2))")
println("- Total CPU threads: $(Sys.CPU_THREADS)\n")

export solve_batch_oim, solve_single_oim, test_oim_dynamics

"""
Compute OIM dynamics for a single state vector.
Optimized for Euler integration without noise.
"""
function compute_dynamics!(du::Vector{Float64}, u::Vector{Float64}, 
                         J::Matrix{Float64}, h::Union{Vector{Float64},Nothing}, 
                         K_s::Union{Vector{Float64},Nothing})
    n = length(u)
    fill!(du, 0.0)
    
    # Compute oscillator coupling terms directly
    @simd for i in 1:n
        du_sum = 0.0
        @simd for j in 1:n
            du_sum -= J[i,j] * sin(u[i] - u[j])
        end
        du[i] = du_sum
    end
    
    # External field (if provided)
    if !isnothing(h)
        @. du -= h * sin(u)
    end
    
    # SHIL sync (if provided)
    if !isnothing(K_s)
        @. du -= K_s * sin(2 * u)
    end
    
    return du
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
                                     random_initialization::Bool=true)
    
    n = size(J, 1)
    n_steps = Int(duration/dt) + 1
    
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
    
    # Pre-allocate arrays
    du = similar(u)
    t_history = collect(range(0, duration, length=n_steps))
    u_history = zeros(n, n_steps)
    u_history[:, 1] = u
    
    # Euler integration with history tracking
    t = 0.0
    step = 1
    while t < duration
        compute_dynamics!(du, u, J, h, K_s)
        @. u += dt * du
        t += dt
        
        if step < n_steps
            step += 1
            u_history[:, step] = u
        end
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
                        target_idx=nothing)
    
    if !make_plot
        # Fast path without plotting - original implementation
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
        
        du = similar(u)
        
        # Euler integration
        t = 0.0
        while t < duration
            compute_dynamics!(du, u, J, h, K_s)
            @. u += dt * du
            t += dt
        end
        return u
    else
        # Path with plotting enabled - use solve_single_oim_with_history
        u, t_history, u_history = solve_single_oim_with_history(
            J, h, K_s,
            u0=u0,
            duration=duration,
            dt=dt,
            random_initialization=random_initialization
        )
        
        # Create plot using the existing plot_dynamics function
        plot_dynamics(t_history, u_history, 
                     phase=phase, 
                     output_start_idx=output_start_idx,
                     target_idx=target_idx)  # Pass target_idx to plotting function
        return u
    end
end

"""
Solve a batch of OIM problems in parallel using threads.
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
                        target_idx=nothing)
    
    n_problems = length(J_batch)
    results = Vector{Vector{Float64}}(undef, n_problems)
    
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
                target_idx=target_idx
            )
        end
    end
    
    return results
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