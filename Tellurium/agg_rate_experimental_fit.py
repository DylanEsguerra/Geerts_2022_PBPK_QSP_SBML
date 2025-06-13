import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from K_rates_extrapolate import calculate_k_rates

# Load the experimental data (same as exp_fit.py)
data = pd.read_csv('ISF_Monomers_pg_ml.csv')
exp_years = np.array(data['Year'])
exp_concentrations = np.array(data['ISF_Monomers_pg_ml'])

# Conversion factor from nM to pg/ml for AB42 (same as exp_fit.py)
# nM * MW / 1000 = pg/ml (MW of AB42 ≈ 4514.0 Da)
conversion_factor = 4514.0 

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

def objective_function(log_params, fast_mode=False):
    """
    Simple objective function: sum of squared differences from experimental data
    
    log_params: [log(kb0_fortytwo), log(kb1_fortytwo), log(AB42_IDE_Kcat_exp)]
    All parameters are log-transformed to ensure positivity
    """
    # Transform back from log space to ensure positivity
    kb0_fortytwo = np.exp(log_params[0])
    kb1_fortytwo = np.exp(log_params[1]) 
    AB42_IDE_Kcat_exp = np.exp(log_params[2])
    
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
        rr.CL_AB42_IDE = AB42_IDE_Kcat_exp
        rr.AB42_IDE_Kcat_exp = AB42_IDE_Kcat_exp
        
        # Update related AB42 rates
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    setattr(rr, key, value)
                except Exception as e:
                    pass  # Skip if parameter doesn't exist
        
        # Run simulation to cover experimental time range
        t_sim = 100*365*24  # 100 years in hours
        n_points = 50 if fast_mode else 200
        sim_result = rr.simulate(0, t_sim, n_points, selections=['time', '[AB42_Monomer]'])
        
        # Check for simulation errors
        if np.any(np.isnan(sim_result)) or np.any(np.isinf(sim_result)):
            return 1e10  # Large penalty for failed simulations
        
        # Convert time to years and concentration to pg/ml
        sim_years = sim_result[:, 0] / 365 / 24
        sim_concentrations = sim_result[:, 1] * conversion_factor
        
        # Interpolate simulation results to match experimental time points
        f = interp1d(sim_years, sim_concentrations, bounds_error=False, fill_value="extrapolate")
        sim_concentrations_interp = f(exp_years)
        
        # Simple objective: sum of squared residuals
        residuals = exp_concentrations - sim_concentrations_interp
        error = np.sum(residuals**2)
        
        # Store the error and parameters in history
        loss_history.append(error)
        param_history.append([kb0_fortytwo, kb1_fortytwo, AB42_IDE_Kcat_exp])
        
        # Print progress
        final_monomer_pgml = sim_concentrations_interp[-1] if len(sim_concentrations_interp) > 0 else 0
        exp_final_pgml = exp_concentrations[-1]
        ratio = final_monomer_pgml / exp_final_pgml if exp_final_pgml > 0 else float('inf')
        print(f"kb0={kb0_fortytwo:.2f}, kb1={kb1_fortytwo:.6f}, kcat={AB42_IDE_Kcat_exp:.0f}, "
              f"sim/exp ratio={ratio:.2f}, error={error:.2e}")
            
        return error
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e10  # Large penalty for any errors

def optimize_parameters(fast_mode=False):
    """Simple parameter optimization using log-transformed parameters with gain factor insights"""
    
    print("Starting parameter optimization informed by gain factor analysis...")
    print("Using log-transformed parameters with realistic bounds")
    print(f"Experimental data: {exp_concentrations[0]:.0f} to {exp_concentrations[-1]:.0f} pg/ml")
    print(f"Time range: {exp_years[0]:.1f} to {exp_years[-1]:.1f} years")
    
    # Insights from gain factor optimization:
    # - kb0 = 9.99 h⁻¹ achieved good gain factors (target was 7.0, achieved 6.51)
    # - kb1 = 0.00172 h⁻¹ was optimal for gain factors
    # These provide excellent starting points for our optimization
    print("\nUsing insights from gain factor optimization:")
    print("- kb0 ~ 10 h⁻¹ achieved good gain factors")
    print("- kb1 ~ 0.0017 h⁻¹ was optimal for gain factors")
    print("- Will explore kcat values while keeping kb0 and kb1 near these optimal values")
    
    # Define realistic parameter bounds (in real space)
    # Based on optimization results: need much lower kb0/kb1 and much higher kcat
    # kb0: allow 1e-6 to 1000 h⁻¹ (much lower minimum)
    # kb1: allow 1e-9 to 10 h⁻¹ (much lower minimum) 
    # kcat: allow 1 to 500000 (much higher maximum for clearance)
    kb0_bounds = (1e-6, 1000.0)
    kb1_bounds = (1e-9, 10.0)
    kcat_bounds = (1.0, 500000.0)
    
    # Convert to log space bounds
    log_bounds = [
        (np.log(kb0_bounds[0]), np.log(kb0_bounds[1])),
        (np.log(kb1_bounds[0]), np.log(kb1_bounds[1])),
        (np.log(kcat_bounds[0]), np.log(kcat_bounds[1]))
    ]
    
    print(f"\nParameter bounds (real space):")
    print(f"  kb0: {kb0_bounds[0]:.3f} to {kb0_bounds[1]:.0f} h⁻¹")
    print(f"  kb1: {kb1_bounds[0]:.0e} to {kb1_bounds[1]:.1f} h⁻¹")
    print(f"  kcat: {kcat_bounds[0]:.0f} to {kcat_bounds[1]:.0f}")
    
    # Create objective function wrapper
    def obj_func_wrapper(log_params):
        return objective_function(log_params, fast_mode=fast_mode)
    
    # Clear history
    global loss_history, param_history
    loss_history = []
    param_history = []
    
    # Starting points informed by gain factor optimization
    # Center around the gain factor optimal values but explore kcat
    gain_optimal_kb0 = 9.99
    gain_optimal_kb1 = 0.00172
    
    starting_points_log = [
        # Start near gain factor optimum with different kcat values
        [np.log(gain_optimal_kb0), np.log(gain_optimal_kb1), np.log(5)],   # Low clearance
        [np.log(gain_optimal_kb0), np.log(gain_optimal_kb1), np.log(50)],   # Medium clearance  
        [np.log(gain_optimal_kb0), np.log(gain_optimal_kb1), np.log(500)],  # High clearance
        [np.log(gain_optimal_kb0), np.log(gain_optimal_kb1), np.log(5000)],  # Very high clearance
        
        # Slight variations around the gain factor optimum
        [np.log(gain_optimal_kb0 * 0.5), np.log(gain_optimal_kb1), np.log(50)],   # Lower kb0
        [np.log(gain_optimal_kb0 * 2.0), np.log(gain_optimal_kb1), np.log(50)],   # Higher kb0
        [np.log(gain_optimal_kb0), np.log(gain_optimal_kb1 * 0.5), np.log(50)],   # Lower kb1
        [np.log(gain_optimal_kb0), np.log(gain_optimal_kb1 * 2.0), np.log(50)],   # Higher kb1
    ]
    
    best_result = None
    best_params_log = None
    best_error = float('inf')
    
    print(f"\nTrying {len(starting_points_log)} starting points with bounded optimization...")
    
    for i, start_log in enumerate(starting_points_log):
        # Convert back to real space for display
        start_real = [np.exp(x) for x in start_log]
        print(f"\n=== ATTEMPT {i+1}/{len(starting_points_log)} ===")
        print(f"Starting: kb0={start_real[0]:.2f}, kb1={start_real[1]:.6f}, kcat={start_real[2]:.0f}")
        
        try:
            result = minimize(
                obj_func_wrapper,
                start_log,
                method='L-BFGS-B',  # Use bounded optimization
                bounds=log_bounds,  # Apply the bounds
                options={
                    'maxiter': 50 if fast_mode else 150,
                    'disp': False,
                    'gtol': 1e-7
                }
            )
            
            # Convert optimized parameters back to real space
            opt_real = [np.exp(x) for x in result.x]
            print(f"Result: kb0={opt_real[0]:.2f}, kb1={opt_real[1]:.6f}, kcat={opt_real[2]:.0f}")
            print(f"Error: {result.fun:.2e}, Success: {result.success}")
            
            if result.success and result.fun < best_error:
                best_result = result
                best_params_log = result.x.copy()
                best_error = result.fun
                print("*** NEW BEST RESULT ***")
                
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("All optimization attempts failed!")
    
    # Convert best parameters back to real space
    best_params_real = [np.exp(x) for x in best_params_log]
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE")
    print(f"="*60)
    print(f"Best parameters:")
    print(f"  kb0 = {best_params_real[0]:.4f} h⁻¹")
    print(f"  kb1 = {best_params_real[1]:.6e} h⁻¹") 
    print(f"  AB42_IDE_Kcat_exp = {best_params_real[2]:.1f}")
    print(f"Final error = {best_error:.4e}")
    
    # Check if parameters are at bounds
    kb0_at_bound = abs(best_params_real[0] - kb0_bounds[0]) < 0.01 or abs(best_params_real[0] - kb0_bounds[1]) < 0.01
    kb1_at_bound = abs(best_params_real[1] - kb1_bounds[0]) < 1e-8 or abs(best_params_real[1] - kb1_bounds[1]) < 0.01
    kcat_at_bound = abs(best_params_real[2] - kcat_bounds[0]) < 0.1 or abs(best_params_real[2] - kcat_bounds[1]) < 0.1
    
    if kb0_at_bound or kb1_at_bound or kcat_at_bound:
        print("\nWARNING: One or more parameters are at their bounds!")
        print("Consider widening bounds if this is limiting the optimization.")
    
    # Compare with gain factor optimal values
    print(f"\nComparison with gain factor optimal values:")
    print(f"  kb0: {best_params_real[0]:.2f} vs gain optimal {gain_optimal_kb0:.2f} (ratio: {best_params_real[0]/gain_optimal_kb0:.2f})")
    print(f"  kb1: {best_params_real[1]:.6f} vs gain optimal {gain_optimal_kb1:.6f} (ratio: {best_params_real[1]/gain_optimal_kb1:.2f})")
    
    # Run one final evaluation to populate history
    final_error = objective_function(best_params_log, fast_mode=fast_mode)
    
    return best_params_real, best_error

def analyze_optimized_parameters(optimal_params):
    """Analyze the optimized parameters and plot results"""
    kb0_opt, kb1_opt, kcat_opt = optimal_params
    
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
    rr.CL_AB42_IDE = kcat_opt
    rr.AB42_IDE_Kcat_exp = kcat_opt
    
    for key, value in rates.items():
        if '_fortytwo' in key and 'Plaque' not in key:
            try:
                setattr(rr, key, value)
            except Exception as e:
                pass
    
    # Run simulation (100 years like exp_fit.py)
    t_sim = 100*365*24
    sim_result = rr.simulate(0, t_sim, 300, selections=['time', '[AB42_Monomer]'])
    
    # Convert simulation results
    sim_years = sim_result[:, 0] / 365 / 24
    sim_concentrations = sim_result[:, 1] * conversion_factor
    
    # Interpolate to experimental time points for error calculation
    f = interp1d(sim_years, sim_concentrations, bounds_error=False, fill_value="extrapolate")
    sim_concentrations_interp = f(exp_years)
    
    # Calculate final error metrics
    residuals = exp_concentrations - sim_concentrations_interp
    final_error = np.sum(residuals**2)
    rms_error = np.sqrt(np.mean(residuals**2))
    
    # Print analysis
    print(f"Final sum of squared residuals: {final_error:.2e}")
    print(f"RMS error: {rms_error:.2f} pg/ml")
    print(f"Mean experimental concentration: {np.mean(exp_concentrations):.1f} pg/ml")
    print(f"Mean simulated concentration: {np.mean(sim_concentrations_interp):.1f} pg/ml")
    
    # Create comprehensive results plot (like Linear_Fit.py and exp_fit.py)
    fig = plt.figure(figsize=(14, 12))
    
    # Plot 1: Experimental data vs model fit
    ax1 = plt.subplot(2, 2, 1)
    plt.scatter(exp_years, exp_concentrations, color='red', label='Experimental data', s=50, zorder=3)
    plt.plot(sim_years, sim_concentrations, 'b-', linewidth=3, label='Optimized model', alpha=0.8)
    plt.scatter(exp_years, sim_concentrations_interp, color='blue', marker='x', s=40, 
                label='Model at exp. points', zorder=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('AB42 Monomer (pg/ml)', fontsize=12, fontweight='bold')
    plt.title('Experimental Data vs Optimized Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss history during optimization
    ax2 = plt.subplot(2, 2, 2)
    iterations = np.arange(1, len(loss_history) + 1)
    plt.semilogy(iterations, loss_history, 'r-', linewidth=2)
    plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (Sum of Squared Residuals)', fontsize=12, fontweight='bold')
    plt.title('Optimization Loss History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameter evolution during optimization
    ax3 = plt.subplot(2, 2, 3)
    param_array = np.array(param_history)
    if len(param_array) > 0:
        iterations = np.arange(1, len(param_array) + 1)
        plt.plot(iterations, param_array[:, 0], 'b-', linewidth=2, label='kb0', alpha=0.7)
        plt.plot(iterations, param_array[:, 1] * 400, 'r-', linewidth=2, label='kb1 × 400', alpha=0.7)
        plt.plot(iterations, param_array[:, 2] / 100, 'g-', linewidth=2, label='kcat / 100', alpha=0.7)
        plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Parameter Values (scaled)', fontsize=12, fontweight='bold')
        plt.title('Parameter Evolution During Optimization', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Residuals analysis
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(exp_years, residuals, 'ko-', linewidth=2, markersize=6)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (exp - sim) pg/ml', fontsize=12, fontweight='bold')
    plt.title('Model Residuals vs Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add text box with optimized parameters (like Linear_Fit.py)
    param_text = (
        f"Optimized Parameters:\n"
        f"kb0 = {kb0_opt:.4f} h⁻¹ ({100*kb0_opt/45.72:.1f}% of original)\n"
        f"kb1 = {kb1_opt:.4e} h⁻¹ ({100*kb1_opt/1.08:.1f}% of original)\n"
        f"AB42_IDE_Kcat_exp = {kcat_opt:.0f} ({100*kcat_opt/50:.0f}% of original)\n"
        f"RMS error = {rms_error:.2f} pg/ml\n"
        f"Sum squared residuals = {final_error:.2e}\n"
        f"Function evaluations = {len(loss_history)}\n"
        f"Sim/Exp final ratio = {sim_concentrations_interp[-1]/exp_concentrations[-1]:.2f}"
    )
    plt.figtext(0.5, 0.02, param_text, ha="center", fontsize=11, 
                bbox={"facecolor":"lightgray", "alpha":0.8, "pad":8})
    
    plt.suptitle('Parameter Optimization Results: Fitting Experimental AB42 Monomer Data', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])  # Make room for parameter text
    plt.savefig('simulation_plots/agg_rate_experimental_fit_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sim_concentrations_interp, sim_result

if __name__ == "__main__":
    import sys
    
    # Check for command line argument for fast mode
    fast_mode = len(sys.argv) > 1 and sys.argv[1].lower() in ['fast', 'quick', '-f', '--fast']
    
    if not fast_mode:
        print("Choose optimization mode:")
        print("1. Full optimization (more iterations)")
        print("2. Fast optimization (fewer iterations)")
        choice = input("Enter 1 or 2 (default: 1): ").strip()
        fast_mode = choice == '2'
        
    print(f"\nMode: {'Fast' if fast_mode else 'Full'}")
    
    # Run optimization
    optimal_params, final_error = optimize_parameters(fast_mode=fast_mode)
    
    # Analyze results
    sim_concentrations, simulation_result = analyze_optimized_parameters(optimal_params)
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE")
    print(f"="*60)
    print(f"Mode: {'Fast' if fast_mode else 'Full'}")
    print(f"Results saved to: simulation_plots/agg_rate_experimental_fit_results.png")
    print(f"Final optimized parameters:")
    print(f"  kb0 = {optimal_params[0]:.4f} h⁻¹ ({100*optimal_params[0]/45.72:.1f}% of original)")
    print(f"  kb1 = {optimal_params[1]:.6f} h⁻¹ ({100*optimal_params[1]/1.08:.1f}% of original)")
    print(f"  AB42_IDE_Kcat_exp = {optimal_params[2]:.1f} ({100*optimal_params[2]/50:.0f}% of original)")
    print(f"Final sum of squared residuals: {final_error:.4e}")
    print(f"RMS error: {np.sqrt(final_error/len(exp_concentrations)):.4f} pg/ml")
    
    # Calculate and display the final simulation vs experimental ratio
    final_sim_conc = sim_concentrations[-1] if len(sim_concentrations) > 0 else 0
    final_exp_conc = exp_concentrations[-1]
    ratio = final_sim_conc / final_exp_conc if final_exp_conc > 0 else float('inf')
    print(f"Final concentration ratio (sim/exp): {ratio:.2f}")
    print(f"Target ratio should be close to 1.0 for good fit") 