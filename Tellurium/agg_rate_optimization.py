import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import K_rates_extrapolate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from K_rates_extrapolate import calculate_k_rates
from scipy.optimize import minimize

# Target gain factor (Geerts value)
TARGET_GAIN = 7.0

# Create a list to store loss values and parameter history
loss_history = []
param_history = []

# Define simulation selections
simulation_selections = ['time', '[AB42_Monomer]', 
                        '[AB42_Oligomer02]', '[AB42_Oligomer03]', '[AB42_Oligomer04]',
                        '[AB42_Oligomer05]', '[AB42_Oligomer06]', '[AB42_Oligomer07]',
                        '[AB42_Oligomer08]', '[AB42_Oligomer09]', '[AB42_Oligomer10]',
                        '[AB42_Oligomer11]', '[AB42_Oligomer12]', '[AB42_Oligomer13]',
                        '[AB42_Oligomer14]', '[AB42_Oligomer15]', '[AB42_Oligomer16]',
                        '[AB42_Fibril17]', '[AB42_Fibril18]', '[AB42_Fibril19]',
                        '[AB42_Fibril20]', '[AB42_Fibril21]', '[AB42_Fibril22]',
                        '[AB42_Fibril23]', '[AB42_Fibril24]',
                        '[AB42_Plaque_unbound]']

# Load the SBML model
with open("combined_master_model.xml", "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

def setup_integrator():
    """Setup robust integrator settings"""
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-15
    rr.integrator.relative_tolerance = 1e-15
    rr.integrator.setValue('stiff', True)

def calculate_gain_factors_from_concentrations(rates, final_concentrations, oligomer_sizes):
    """Calculate concentration-dependent gain factors given rates and concentrations"""
    concentration_dependent_gains = []
    
    for size in oligomer_sizes:
        # Get rate constants
        if size < 17:
            kf = rates[f'k_O{size-1}_O{size}_fortytwo']
            kb = rates[f'k_O{size}_O{size-1}_fortytwo']
            reactant_species = f'AB42_Oligomer{size-1:02d}' if size > 2 else 'AB42_Monomer'
            product_species = f'AB42_Oligomer{size:02d}'
        elif size == 17:
            kf = rates[f'k_O{size-1}_F{size}_fortytwo']
            kb = rates[f'k_F{size}_O{size-1}_fortytwo']
            reactant_species = f'AB42_Oligomer{size-1:02d}'
            product_species = f'AB42_Fibril{size}'
        else:
            kf = rates[f'k_F{size-1}_F{size}_fortytwo']
            kb = rates[f'k_F{size}_F{size-1}_fortytwo']
            reactant_species = f'AB42_Fibril{size-1}'
            product_species = f'AB42_Fibril{size}'
        
        # Get concentrations
        monomer_conc = final_concentrations['AB42_Monomer']
        reactant_conc = final_concentrations[reactant_species]
        product_conc = final_concentrations[product_species]
        
        # Calculate reaction velocities
        forward_velocity = kf * reactant_conc * monomer_conc
        backward_velocity = kb * product_conc
        
        # Concentration-dependent gain factor
        if backward_velocity > 0:
            gain = forward_velocity / backward_velocity
        else:
            gain = float('inf')
        
        concentration_dependent_gains.append(gain)
    
    return concentration_dependent_gains

def objective_function(params, oligomer_sizes_subset=None, fast_mode=False, regularization_weight=0.1):
    """
    Objective function to minimize: sum of squared differences from target gain
    
    params: [kb0_fortytwo, kb1_fortytwo] in units of 1/h, 1/h
    oligomer_sizes_subset: which oligomer sizes to include in optimization (default: all)
    fast_mode: if True, use shorter simulation time for faster optimization
    regularization_weight: weight for penalty encouraging lower parameter values (0 = no penalty)
    """
    kb0_fortytwo, kb1_fortytwo = params
    
    try:
        # Convert to Garai units (1/s)
        Garai_kb0 = kb0_fortytwo / 3600
        Garai_kb1 = kb1_fortytwo / 3600
        
        # Calculate all rates for these parameters
        rates = calculate_k_rates(original_kb0_fortytwo=Garai_kb0, 
                                 original_kb1_fortytwo=Garai_kb1)
        
        # Run simulation
        rr.reset()
        setup_integrator()
        
        # Set parameters
        rr.k_O2_M_fortytwo = kb0_fortytwo
        rr.k_O3_O2_fortytwo = kb1_fortytwo
        
        # Keep clearance parameters at default values (don't optimize)
        # rr.CL_AB42_IDE and rr.AB42_IDE_Kcat_exp will use model defaults
        
        # Update related AB42 rates
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    setattr(rr, key, value)
                except Exception as e:
                    pass  # Skip if parameter doesn't exist
        
        # Run simulation to equilibrium (shorter time in fast mode)
        t_sim = (10*365*24) if fast_mode else (50*365*24)  # 10 or 50 years
        n_points = 50 if fast_mode else 100  # Fewer time points in fast mode
        sim_result = rr.simulate(0, t_sim, n_points, selections=simulation_selections)
        
        # Check for simulation errors
        if np.any(np.isnan(sim_result)) or np.any(np.isinf(sim_result)):
            return 1e6  # Large penalty for failed simulations
        
        # Extract final concentrations
        final_concentrations = {}
        final_concentrations['AB42_Monomer'] = sim_result[-1, 1]
        
        col_idx = 2
        for size in range(2, 25):
            if size < 17:
                species_name = f'AB42_Oligomer{size:02d}'
            else:
                species_name = f'AB42_Fibril{size}'
            concentration = sim_result[-1, col_idx]
            
            if concentration < 0:
                concentration = 0
            
            final_concentrations[species_name] = concentration
            col_idx += 1
        
        # Calculate gain factors
        oligomer_sizes = list(range(4, 25)) if oligomer_sizes_subset is None else oligomer_sizes_subset
        gains = calculate_gain_factors_from_concentrations(rates, final_concentrations, oligomer_sizes)
        
        # Calculate error from target
        gains = np.array(gains)
        gains = gains[np.isfinite(gains)]  # Remove any infinite values
        
        if len(gains) == 0:
            return 1e6  # Large penalty if no valid gains
        
        # Sum of squared errors from target
        error = np.sum((gains - TARGET_GAIN)**2)
        
        # Add regularization penalty to encourage lower parameter values
        if regularization_weight > 0:
            # Normalize kb0 by its maximum value (we want this to be lower)
            normalized_kb0 = kb0_fortytwo / 45.72  # Original max value
            
            # For kb1, penalize values that are too high (away from the promising low region)
            kb1_penalty = max(0, (kb1_fortytwo - 0.005) / 0.005)  # Penalty increases above 0.005
            
            # L2 regularization - only for kb0 and kb1
            regularization_penalty = regularization_weight * (normalized_kb0**2 + kb1_penalty**2)
            error += regularization_penalty
        
        # Add penalty for parameter values outside bounds
        if kb0_fortytwo < 0.0 or kb0_fortytwo > 45.72:
            error += 1e3
        if kb1_fortytwo < 0.0 or kb1_fortytwo > 1.08:
            error += 1e3
        
        # Store the error and parameters in history
        loss_history.append(error)
        param_history.append([kb0_fortytwo, kb1_fortytwo])  # Only store kb0 and kb1
        
        # Print progress
        mean_gain = np.mean(gains) if len(gains) > 0 else float('inf')
        print(f"Testing params: kb0={kb0_fortytwo:.3f}, kb1={kb1_fortytwo:.5f}, "
              f"mean_gain={mean_gain:.3f}, error={error:.2e}")
            
        return error
        
    except Exception as e:
        print(f"Simulation failed with params {params}: {e}")
        return 1e6  # Large penalty for any errors

def optimize_parameters(fast_mode=False, regularization_weight=0.1):
    """Perform parameter optimization to target gain factor of 7"""
    
    if fast_mode:
        print("Starting FAST parameter optimization to target gain factor of 7...")
        print("Fast mode: shorter simulations, fewer iterations")
    else:
        print("Starting parameter optimization to target gain factor of 7...")
        print("This may take several minutes...")
    
    print(f"Regularization weight: {regularization_weight} (higher values prefer lower parameters)")
    
    # Use the winning strategy: L-BFGS-B starting from [10.0, 0.001]
    # starts from very low kb1 region where gain factors are optimal
    print("Using optimized strategy based on sensitivity analysis...")
    print("Starting from low kb1 region where gain factors are optimal")
    print("Note: AB42_IDE_Kcat_exp will remain at model default (not optimized)")
    
    # Starting point - only kb0 and kb1
    starting_point = [10.0, 0.001]  # kb0=10.0, kb1=0.001
    
    # Define bounds - only for kb0 and kb1
    bounds = [
        (0.001, 45.72),      # kb0: keep full range
        (1e-5, 1.08),        # kb1: full range
    ]
    
    print(f"Starting point: kb0={starting_point[0]:.1f}, kb1={starting_point[1]:.4f}")
    print(f"Parameter bounds:")
    print(f"  kb0: {bounds[0][0]:.3f} to {bounds[0][1]:.2f} h⁻¹")
    print(f"  kb1: {bounds[1][0]:.2e} to {bounds[1][1]:.3f} h⁻¹")
    
    # Create objective function with fast_mode and regularization
    def obj_func_wrapper(params):
        return objective_function(params, fast_mode=fast_mode, regularization_weight=regularization_weight)
    
    # Clear history for fresh start
    global loss_history, param_history
    loss_history = []
    param_history = []
    
    # Run the winning optimization strategy
    print("\nStarting L-BFGS-B optimization...")
    
    result = minimize(
        obj_func_wrapper, 
        starting_point,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 200 if not fast_mode else 100,
            'disp': True
        }
    )
    
    # Extract the optimized parameters
    optimized_kb0 = result.x[0]
    optimized_kb1 = result.x[1]
    
    print(f"\nOptimization complete!")
    print(f"Optimized kb0 = {optimized_kb0:.6f} h⁻¹")
    print(f"Optimized kb1 = {optimized_kb1:.6e} h⁻¹")
    print(f"AB42_IDE_Kcat_exp = 50 (model default, not optimized)")
    print(f"Final error = {result.fun:.4e}")
    print(f"Success: {result.success}")
    
    return result.x, result.fun

def analyze_optimized_parameters(optimal_params):
    """Analyze the optimized parameters and plot results"""
    kb0_opt, kb1_opt = optimal_params
    
    print(f"\n" + "="*60)
    print(f"ANALYZING OPTIMIZED PARAMETERS")
    print(f"="*60)
    
    # Calculate rates with optimized parameters
    Garai_kb0 = kb0_opt / 3600
    Garai_kb1 = kb1_opt / 3600
    rates = calculate_k_rates(original_kb0_fortytwo=Garai_kb0, 
                             original_kb1_fortytwo=Garai_kb1)
    
    # Run simulation with optimized parameters
    rr.reset()
    setup_integrator()
    
    rr.k_O2_M_fortytwo = kb0_opt
    rr.k_O3_O2_fortytwo = kb1_opt
    
    for key, value in rates.items():
        if '_fortytwo' in key and 'Plaque' not in key:
            try:
                setattr(rr, key, value)
            except Exception as e:
                pass
    
    # Run simulation
    t_sim = 50*365*24
    sim_result = rr.simulate(0, t_sim, 100, selections=simulation_selections)
    
    # Extract concentrations
    final_concentrations = {}
    final_concentrations['AB42_Monomer'] = sim_result[-1, 1]
    
    col_idx = 2
    for size in range(2, 25):
        if size < 17:
            species_name = f'AB42_Oligomer{size:02d}'
        else:
            species_name = f'AB42_Fibril{size}'
        concentration = sim_result[-1, col_idx]
        final_concentrations[species_name] = max(0, concentration)
        col_idx += 1
    
    # Calculate gain factors
    oligomer_sizes = list(range(4, 25))
    optimized_gains = calculate_gain_factors_from_concentrations(rates, final_concentrations, oligomer_sizes)
    
    # Calculate default gains for comparison
    default_rates = calculate_k_rates()  # Use default values
    default_gains = []
    for size in oligomer_sizes:
        if size < 17:
            kf = default_rates[f'k_O{size-1}_O{size}_fortytwo']
            kb = default_rates[f'k_O{size}_O{size-1}_fortytwo']
        elif size == 17:
            kf = default_rates[f'k_O{size-1}_F{size}_fortytwo']
            kb = default_rates[f'k_F{size}_O{size-1}_fortytwo']
        else:
            kf = default_rates[f'k_F{size-1}_F{size}_fortytwo']
            kb = default_rates[f'k_F{size}_F{size-1}_fortytwo']
        default_gains.append(kf / kb)  # Theoretical gain
    
    # Print analysis
    print(f"Mean gain factor (optimized): {np.mean(optimized_gains):.2f}")
    print(f"Std deviation (optimized): {np.std(optimized_gains):.2f}")
    print(f"Min/Max gains (optimized): {np.min(optimized_gains):.2f} / {np.max(optimized_gains):.2f}")
    print(f"RMS error from target (7.0): {np.sqrt(np.mean((np.array(optimized_gains) - 7.0)**2)):.3f}")
    
    # Create comprehensive results plot (like Linear_Fit.py)
    fig = plt.figure(figsize=(14, 12))
    
    # Plot 1: Gain factors comparison
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(oligomer_sizes, optimized_gains, 'bo-', linewidth=3, markersize=8, 
             label=f'Optimized (kb0={kb0_opt:.2f}, kb1={kb1_opt:.4f})')
    plt.plot(oligomer_sizes, default_gains, 'ro-', linewidth=3, markersize=8, alpha=0.7,
             label='Default (kb0=45.72, kb1=1.08)')
    plt.axhline(y=7, color='green', linestyle='--', linewidth=3, alpha=0.8, label='Target (7.0)')
    plt.axvline(x=17, color='grey', linestyle='-', alpha=0.6, linewidth=2)
    plt.yscale('log')
    plt.xlabel('Oligomer Size', fontsize=12, fontweight='bold')
    plt.ylabel('Gain Factor', fontsize=12, fontweight='bold')
    plt.title('Optimized vs Default Gain Factors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss history during optimization
    ax2 = plt.subplot(2, 2, 2)
    iterations = np.arange(1, len(loss_history) + 1)
    plt.semilogy(iterations, loss_history, 'r-', linewidth=2)
    plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (Sum of Squared Errors)', fontsize=12, fontweight='bold')
    plt.title('Optimization Loss History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameter evolution during optimization
    ax3 = plt.subplot(2, 2, 3)
    param_array = np.array(param_history)
    if len(param_array) > 0:
        iterations = np.arange(1, len(param_array) + 1)
        plt.plot(iterations, param_array[:, 0], 'b-', linewidth=2, label='kb0', alpha=0.7)
        plt.plot(iterations, param_array[:, 1] * 400, 'r-', linewidth=2, label='kb1 × 400', alpha=0.7)  # Scale for visibility
        plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Parameter Values (scaled)', fontsize=12, fontweight='bold')
        plt.title('Parameter Evolution During Optimization', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Error from target by oligomer size
    ax4 = plt.subplot(2, 2, 4)
    errors = np.abs(np.array(optimized_gains) - 7.0)
    plt.plot(oligomer_sizes, errors, 'ko-', linewidth=3, markersize=6)
    plt.axvline(x=17, color='grey', linestyle='-', alpha=0.6, linewidth=2, label='O→F transition')
    plt.yscale('log')
    plt.xlabel('Oligomer Size', fontsize=12, fontweight='bold')
    plt.ylabel('|Gain - 7.0|', fontsize=12, fontweight='bold')
    plt.title('Absolute Error from Target', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with optimized parameters (like Linear_Fit.py)
    final_error = np.sqrt(np.mean((np.array(optimized_gains) - 7.0)**2))
    param_text = (
        f"Optimized Parameters:\n"
        f"kb0 = {kb0_opt:.4f} h⁻¹ ({100*kb0_opt/45.72:.1f}% of original)\n"
        f"kb1 = {kb1_opt:.4e} h⁻¹ ({100*kb1_opt/1.08:.1f}% of original)\n"
        f"Mean gain = {np.mean(optimized_gains):.3f}\n"
        f"RMS error = {final_error:.3f}\n"
        f"Function evaluations = {len(loss_history)}"
    )
    plt.figtext(0.5, 0.02, param_text, ha="center", fontsize=11, 
                bbox={"facecolor":"lightgray", "alpha":0.8, "pad":8})
    
    plt.suptitle('Parameter Optimization Results: Targeting Gain Factor = 7', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])  # Make room for parameter text
    plt.savefig('simulation_plots/agg_rate_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimized_gains, sim_result

if __name__ == "__main__":
    import sys
    
    # Check for command line argument for fast mode
    fast_mode = len(sys.argv) > 1 and sys.argv[1].lower() in ['fast', 'quick', '-f', '--fast']
    
    if not fast_mode:
        print("Choose optimization mode:")
        print("1. Full optimization (50 years, more iterations)")
        print("2. Fast optimization (10 years, fewer iterations)")
        choice = input("Enter 1 or 2 (default: 1): ").strip()
        fast_mode = choice == '2'
    
    # Simplified regularization selection
    print(f"\nMode: {'Fast' if fast_mode else 'Full'}")
    print("Choose regularization strength:")
    print("1. Light (0.1) - Slight preference for lower parameters") 
    print("2. Medium (0.5) - Moderate preference for lower parameters")
    print("3. Strong (1.0) - Strong preference for lower parameters")
    
    reg_choice = input("Enter 1-3 (default: 2): ").strip()
    regularization_weights = {'1': 0.1, '2': 0.5, '3': 1.0}
    regularization_weight = regularization_weights.get(reg_choice, 0.5)
    
    # Run optimization
    optimal_params, final_error = optimize_parameters(fast_mode=fast_mode, 
                                                     regularization_weight=regularization_weight)
    
    # Analyze results
    optimized_gains, simulation_result = analyze_optimized_parameters(optimal_params)
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE")
    print(f"="*60)
    print(f"Mode: {'Fast' if fast_mode else 'Full'}")
    print(f"Regularization weight: {regularization_weight}")
    print(f"Results saved to: simulation_plots/agg_rate_optimization_results.png")
    print(f"Final optimized parameters:")
    print(f"  kb0 = {optimal_params[0]:.4f} h⁻¹ ({100*optimal_params[0]/45.72:.1f}% of original)")
    print(f"  kb1 = {optimal_params[1]:.6f} h⁻¹ ({100*optimal_params[1]/1.08:.1f}% of original)")
    print(f"Target gain factor: {TARGET_GAIN}")
    print(f"Achieved mean gain: {np.mean(optimized_gains):.3f}")
    print(f"RMS error: {np.sqrt(final_error):.4f}") 