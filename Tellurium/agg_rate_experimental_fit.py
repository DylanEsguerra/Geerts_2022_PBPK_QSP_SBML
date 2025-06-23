import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import sys
import os

# Add parent directory to path to import K_rates_extrapolate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from K_rates_extrapolate import calculate_k_rates, convert_backward_rate, convert_forward_rate

# Define parent directory for file paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the experimental data for plasma ratio
data = pd.read_csv('Plasma_Ratio_Geerts.csv')
exp_years = np.array(data['Year'])
exp_ratios = np.array(data['Plasma_Ratio'])

# Create a list to store loss values and parameter history
loss_history = []
param_history = []

# Define simulation selections for Aβ42/Aβ40 ratio in plasma
simulation_selections = ['time', '[AB42Mu_Brain_Plasma]', '[AB40Mu_Brain_Plasma]']

# Load the SBML model
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model_garai.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

def setup_integrator():
    """Setup robust integrator settings"""
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-10
    rr.integrator.relative_tolerance = 1e-10
    rr.integrator.setValue('stiff', True)

def objective_function(log_params):
    """
    Objective function: sum of squared differences from experimental Aβ42/Aβ40 ratio
    
    log_params: [log(kb0_forty), log(kb1_forty), log(kb0_fortytwo), log(kb1_fortytwo), log(kf0_forty), log(kf1_forty), log(kf0_fortytwo), log(kf1_fortytwo), log(baseline_ab40_rate), log(baseline_ab42_rate)]
    All parameters are log-transformed to ensure positivity
    """
    # Transform back from log space
    kb0_forty_h = np.exp(log_params[0])
    kb1_forty_h = np.exp(log_params[1])
    kb0_fortytwo_h = np.exp(log_params[2])
    kb1_fortytwo_h = np.exp(log_params[3])
    kf0_forty_h = np.exp(log_params[4])
    kf1_forty_h = np.exp(log_params[5])
    kf0_fortytwo_h = np.exp(log_params[6])
    kf1_fortytwo_h = np.exp(log_params[7])
    baseline_ab40_rate = np.exp(log_params[8])
    baseline_ab42_rate = np.exp(log_params[9])

    try:
        # Convert backward rates from h⁻¹ to s⁻¹ for calculate_k_rates
        kb0_forty_s = kb0_forty_h / 3600
        kb1_forty_s = kb1_forty_h / 3600
        kb0_fortytwo_s = kb0_fortytwo_h / 3600
        kb1_fortytwo_s = kb1_fortytwo_h / 3600

        # Convert forward rates from nM⁻¹h⁻¹ to M⁻¹s⁻¹
        kf0_forty_M_s = kf0_forty_h / 3.6e-6
        kf1_forty_M_s = kf1_forty_h / 3.6e-6
        kf0_fortytwo_M_s = kf0_fortytwo_h / 3.6e-6
        kf1_fortytwo_M_s = kf1_fortytwo_h / 3.6e-6

        # Calculate all rates for these parameters
        rates = calculate_k_rates(
            original_kb0_forty=kb0_forty_s,
            original_kb1_forty=kb1_forty_s,
            original_kb0_fortytwo=kb0_fortytwo_s,
            original_kb1_fortytwo=kb1_fortytwo_s,
            original_kf0_forty=kf0_forty_M_s,
            original_kf1_forty=kf1_forty_M_s,
            original_kf0_fortytwo=kf0_fortytwo_M_s,
            original_kf1_fortytwo=kf1_fortytwo_M_s,
            baseline_ab40_plaque_rate=baseline_ab40_rate,
            baseline_ab42_plaque_rate=baseline_ab42_rate
        )
        
        # Run simulation
        rr.reset()
        setup_integrator()
        
        # Set parameters in h⁻¹ or nM⁻¹h⁻¹
        rr.k_O2_M_forty = kb0_forty_h
        rr.k_O3_O2_forty = kb1_forty_h
        rr.k_O2_M_fortytwo = kb0_fortytwo_h
        rr.k_O3_O2_fortytwo = kb1_fortytwo_h
        rr.k_M_O2_forty = kf0_forty_h
        rr.k_O2_O3_forty = kf1_forty_h
        rr.k_M_O2_fortytwo = kf0_fortytwo_h
        rr.k_O2_O3_fortytwo = kf1_fortytwo_h
        
        # Update related rates for both AB40 and AB42
        for key, value in rates.items():
            try:
                setattr(rr, key, value)
            except Exception as e:
                pass  # Skip if parameter doesn't exist
        
        # Run simulation to cover experimental time range
        t_sim = 100*365*24  # 100 years in hours
        n_points = 200  # Always use high resolution
        sim_result = rr.simulate(0, t_sim, n_points, selections=simulation_selections)
        
        # Check for simulation errors
        if np.any(np.isnan(sim_result)) or np.any(np.isinf(sim_result)):
            return 1e10  # Large penalty for failed simulations
        
        # Convert time to years and calculate the ratio
        sim_years = sim_result[:, 0] / 365 / 24
        sim_ratio = sim_result[:, 1] / sim_result[:, 2] # AB42/AB40
        
        # Interpolate simulation results to match experimental time points
        f = interp1d(sim_years, sim_ratio, bounds_error=False, fill_value="extrapolate")
        sim_ratio_interp = f(exp_years)
        
        # Simple objective: sum of squared residuals
        residuals = exp_ratios - sim_ratio_interp
        error = np.sum(residuals**2)
        
        # Store the error and parameters in history
        loss_history.append(error)
        param_history.append([
            kb0_forty_h, kb1_forty_h, kb0_fortytwo_h, kb1_fortytwo_h,
            kf0_forty_h, kf1_forty_h, kf0_fortytwo_h, kf1_fortytwo_h,
            baseline_ab40_rate, baseline_ab42_rate
        ])
        
        # Print progress
        final_ratio = sim_ratio_interp[-1] if len(sim_ratio_interp) > 0 else 0
        exp_final_ratio = exp_ratios[-1]
        ratio_of_ratios = final_ratio / exp_final_ratio if exp_final_ratio > 0 else float('inf')
        print(f"kb0_40={kb0_forty_h:.2e}, kb1_40={kb1_forty_h:.2e}, kb0_42={kb0_fortytwo_h:.2e}, kb1_42={kb1_fortytwo_h:.2e}, "
              f"kf0_40={kf0_forty_h:.2e}, kf1_40={kf1_forty_h:.2e}, kf0_42={kf0_fortytwo_h:.2e}, kf1_42={kf1_fortytwo_h:.2e}, "
              f"base_40={baseline_ab40_rate:.2e}, base_42={baseline_ab42_rate:.2e}, "
              f"sim/exp ratio={ratio_of_ratios:.2f}, error={error:.2e}")
            
        return error
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e10  # Large penalty for any errors

def optimize_parameters():
    """Simple parameter optimization with early stopping and parameter bounds"""
    
    print("Starting parameter optimization for Aβ42/Aβ40 plasma ratio...")
    print("Using current PK_Geerts values as initial conditions")
    print(f"Experimental data: {exp_ratios[0]:.3f} to {exp_ratios[-1]:.3f} ratio")
    print(f"Time range: {exp_years[0]:.1f} to {exp_years[-1]:.1f} years")
    
    # Current values from PK_Geerts.csv and K_rates_extrapolate.py, converted to h⁻¹ or nM⁻¹h⁻¹
    current_kb0_forty_h = convert_backward_rate(2.7e-3)
    current_kb1_forty_h = convert_backward_rate(0.00001 / 3600)
    current_kb0_fortytwo_h = convert_backward_rate(12.7e-3)
    current_kb1_fortytwo_h = convert_backward_rate(0.00001 / 3600)
    current_kf0_forty_h = convert_forward_rate(0.5 * 10**2)
    current_kf1_forty_h = convert_forward_rate(20.0)
    current_kf0_fortytwo_h = convert_forward_rate(9.9 * 10**2)
    current_kf1_fortytwo_h = convert_forward_rate(38.0)
    # Baseline plaque formation rates (these are the minimum values)
    current_baseline_ab40_rate = 0.000005  # 5e-06 L/(nM·h)
    current_baseline_ab42_rate = 0.00005   # 5e-05 L/(nM·h)

    print(f"\nCurrent initial values (h⁻¹ or nM⁻¹h⁻¹):")
    print(f"  kb0_40 = {current_kb0_forty_h:.4e}")
    print(f"  kb1_40 = {current_kb1_forty_h:.4e}")
    print(f"  kb0_42 = {current_kb0_fortytwo_h:.4e}")
    print(f"  kb1_42 = {current_kb1_fortytwo_h:.4e}")
    print(f"  kf0_40 = {current_kf0_forty_h:.4e}")
    print(f"  kf1_40 = {current_kf1_forty_h:.4e}")
    print(f"  kf0_42 = {current_kf0_fortytwo_h:.4e}")
    print(f"  kf1_42 = {current_kf1_fortytwo_h:.4e}")
    print(f"  base_40 = {current_baseline_ab40_rate:.4e}")
    print(f"  base_42 = {current_baseline_ab42_rate:.4e}")

    # Define reasonable parameter bounds (in log space)
    # Note: baseline rates have minimum at current values since they are baseline
    # Backward rates have minimum of 1e-05 h⁻¹
    bounds = [
        (np.log(max(1e-05, current_kb0_forty_h * 0.001)), np.log(current_kb0_forty_h * 100)),
        (np.log(max(1e-05, current_kb1_forty_h * 0.001)), np.log(current_kb1_forty_h * 100)),
        (np.log(max(1e-05, current_kb0_fortytwo_h * 0.001)), np.log(current_kb0_fortytwo_h * 100)),
        (np.log(max(1e-05, current_kb1_fortytwo_h * 0.001)), np.log(current_kb1_fortytwo_h * 100)),
        (np.log(current_kf0_forty_h * 0.001), np.log(current_kf0_forty_h * 100)),
        (np.log(current_kf1_forty_h * 0.001), np.log(current_kf1_forty_h * 100)),
        (np.log(current_kf0_fortytwo_h * 0.001), np.log(current_kf0_fortytwo_h * 100)),
        (np.log(current_kf1_fortytwo_h * 0.001), np.log(current_kf1_fortytwo_h * 100)),
        (np.log(current_baseline_ab40_rate), np.log(current_baseline_ab40_rate * 100)),  # Min at current (baseline)
        (np.log(current_baseline_ab42_rate), np.log(current_baseline_ab42_rate * 100)),  # Min at current (baseline)
    ]
    
    print(f"\nParameter bounds (h⁻¹ or nM⁻¹h⁻¹):")
    print(f"  kb0_40: {np.exp(bounds[0][0]):.1e} to {np.exp(bounds[0][1]):.1e}")
    print(f"  kb1_40: {np.exp(bounds[1][0]):.1e} to {np.exp(bounds[1][1]):.1e}")
    print(f"  kb0_42: {np.exp(bounds[2][0]):.1e} to {np.exp(bounds[2][1]):.1e}")
    print(f"  kb1_42: {np.exp(bounds[3][0]):.1e} to {np.exp(bounds[3][1]):.1e}")
    print(f"  kf0_40: {np.exp(bounds[4][0]):.1e} to {np.exp(bounds[4][1]):.1e}")
    print(f"  kf1_40: {np.exp(bounds[5][0]):.1e} to {np.exp(bounds[5][1]):.1e}")
    print(f"  kf0_42: {np.exp(bounds[6][0]):.1e} to {np.exp(bounds[6][1]):.1e}")
    print(f"  kf1_42: {np.exp(bounds[7][0]):.1e} to {np.exp(bounds[7][1]):.1e}")
    print(f"  base_40: {np.exp(bounds[8][0]):.1e} to {np.exp(bounds[8][1]):.1e}")
    print(f"  base_42: {np.exp(bounds[9][0]):.1e} to {np.exp(bounds[9][1]):.1e}")
    
    # Early stopping variables
    best_error = float('inf')
    best_params = None
    no_improvement_count = 0
    patience = 20  # Stop if no improvement for 20 iterations
    
    def obj_func_wrapper_with_early_stopping(log_params):
        nonlocal best_error, best_params, no_improvement_count
        
        error = objective_function(log_params)
        
        # Check for improvement
        if error < best_error * 0.999:  # Require at least 0.1% improvement
            best_error = error
            best_params = log_params.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # Early stopping check
        if no_improvement_count >= patience:
            print(f"\nEarly stopping triggered after {no_improvement_count} iterations without improvement")
            print(f"Best error so far: {best_error:.2e}")
            # Return a large value to signal stopping
            return best_error
            
        return error
    
    # Clear history
    global loss_history, param_history
    loss_history = []
    param_history = []
    
    # Starting point using current values
    starting_point_log = [
        np.log(current_kb0_forty_h),
        np.log(current_kb1_forty_h),
        np.log(current_kb0_fortytwo_h),
        np.log(current_kb1_fortytwo_h),
        np.log(current_kf0_forty_h),
        np.log(current_kf1_forty_h),
        np.log(current_kf0_fortytwo_h),
        np.log(current_kf1_fortytwo_h),
        np.log(current_baseline_ab40_rate),
        np.log(current_baseline_ab42_rate),
    ]
    
    print(f"\nStarting optimization with bounds and early stopping...")
    print(f"Early stopping patience: {patience} iterations")
    
    try:
        result = minimize(
            obj_func_wrapper_with_early_stopping,
            starting_point_log,
            method='L-BFGS-B',
            bounds=bounds,  # Apply parameter bounds
            options={
                'maxiter': 200,  # Reduced from 300 since we have early stopping
                'disp': True,
                'gtol': 1e-6,   # Slightly relaxed tolerance
                'ftol': 1e-9    # Relaxed function tolerance
            }
        )
        
        # Use best parameters if early stopping was triggered
        if best_params is not None and best_error < result.fun:
            print(f"Using early stopping result (better than final result)")
            final_params = best_params
            final_error = best_error
        else:
            final_params = result.x
            final_error = result.fun
        
        # Convert optimized parameters back to real space
        opt_real = [np.exp(x) for x in final_params]
        print(f"\nOptimization complete!")
        print(f"Result: kb0_40={opt_real[0]:.2e}, kb1_40={opt_real[1]:.2e}, kb0_42={opt_real[2]:.2e}, kb1_42={opt_real[3]:.2e}, "
              f"kf0_40={opt_real[4]:.2e}, kf1_40={opt_real[5]:.2e}, kf0_42={opt_real[6]:.2e}, kf1_42={opt_real[7]:.2e}, "
              f"base_40={opt_real[8]:.2e}, base_42={opt_real[9]:.2e}")
        print(f"Error: {final_error:.2e}, Success: {result.success}")
        
        if not result.success:
            print("Warning: Optimization may not have converged!")
            
        return opt_real, final_error
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        raise RuntimeError("Optimization failed!")

def analyze_optimized_parameters(optimal_params):
    """Analyze the optimized parameters and plot results"""
    kb0_forty_h, kb1_forty_h, kb0_fortytwo_h, kb1_fortytwo_h, \
    kf0_forty_h, kf1_forty_h, kf0_fortytwo_h, kf1_fortytwo_h, \
    baseline_ab40_rate, baseline_ab42_rate = optimal_params
    
    print(f"\n" + "="*60)
    print(f"ANALYZING OPTIMIZED PARAMETERS")
    print(f"="*60)
    
    # Calculate rates with optimized parameters
    rates = calculate_k_rates(
        original_kb0_forty=kb0_forty_h / 3600,
        original_kb1_forty=kb1_forty_h / 3600,
        original_kb0_fortytwo=kb0_fortytwo_h / 3600,
        original_kb1_fortytwo=kb1_fortytwo_h / 3600,
        original_kf0_forty=kf0_forty_h / 3.6e-6,
        original_kf1_forty=kf1_forty_h / 3.6e-6,
        original_kf0_fortytwo=kf0_fortytwo_h / 3.6e-6,
        original_kf1_fortytwo=kf1_fortytwo_h / 3.6e-6,
        baseline_ab40_plaque_rate=baseline_ab40_rate,
        baseline_ab42_plaque_rate=baseline_ab42_rate
    )
    
    # Run simulation with optimized parameters
    rr.reset()
    setup_integrator()
    
    rr.k_O2_M_forty = kb0_forty_h
    rr.k_O3_O2_forty = kb1_forty_h
    rr.k_O2_M_fortytwo = kb0_fortytwo_h
    rr.k_O3_O2_fortytwo = kb1_fortytwo_h
    rr.k_M_O2_forty = kf0_forty_h
    rr.k_O2_O3_forty = kf1_forty_h
    rr.k_M_O2_fortytwo = kf0_fortytwo_h
    rr.k_O2_O3_fortytwo = kf1_fortytwo_h
    
    for key, value in rates.items():
        try:
            setattr(rr, key, value)
        except Exception as e:
            pass
    
    # Run simulation (100 years)
    t_sim = 100*365*24
    sim_result = rr.simulate(0, t_sim, 300, selections=simulation_selections)
    
    # Convert simulation results
    sim_years = sim_result[:, 0] / 365 / 24
    sim_ratios = sim_result[:, 1] / sim_result[:, 2]
    
    # Interpolate to experimental time points for error calculation
    f = interp1d(sim_years, sim_ratios, bounds_error=False, fill_value="extrapolate")
    sim_ratios_interp = f(exp_years)
    
    # Calculate final error metrics
    residuals = exp_ratios - sim_ratios_interp
    final_error = np.sum(residuals**2)
    rms_error = np.sqrt(np.mean(residuals**2))
    
    # Print analysis
    print(f"Final sum of squared residuals: {final_error:.2e}")
    print(f"RMS error: {rms_error:.4f} (ratio units)")
    print(f"Mean experimental ratio: {np.mean(exp_ratios):.3f}")
    print(f"Mean simulated ratio: {np.mean(sim_ratios_interp):.3f}")
    
    # Create comprehensive results plot
    fig = plt.figure(figsize=(14, 12))
    
    # Plot 1: Experimental data vs model fit
    ax1 = plt.subplot(2, 2, 1)
    plt.scatter(exp_years, exp_ratios, color='red', label='Experimental data', s=80, zorder=3, alpha=0.8)
    plt.plot(sim_years, sim_ratios, 'b-', linewidth=3, label='Optimized model', alpha=0.8)
    plt.scatter(exp_years, sim_ratios_interp, color='blue', marker='x', s=60, 
                label='Model at exp. points', zorder=3, linewidth=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Aβ42/Aβ40 Plasma Ratio', fontsize=12, fontweight='bold')
    plt.title('Experimental Ratio vs Optimized Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
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
        plt.plot(iterations, param_array[:, 0], 'r-', linewidth=2, label='kb0_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 1], 'g-', linewidth=2, label='kb1_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 2], 'b-', linewidth=2, label='kb0_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 3], 'c-', linewidth=2, label='kb1_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 4], 'm-', linewidth=2, label='kf0_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 5], 'y-', linewidth=2, label='kf1_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 6], 'k-', linewidth=2, label='kf0_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 7], 'orange', linestyle='-', linewidth=2, label='kf1_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 8], 'brown', linestyle='-', linewidth=2, label='base_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 9], 'pink', linestyle='-', linewidth=2, label='base_42', alpha=0.7)
        plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Parameter Values (h⁻¹ or nM⁻¹h⁻¹)', fontsize=12, fontweight='bold')
        plt.title('Parameter Evolution During Optimization', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, ncol=2)  # Use smaller font and 2 columns for 10 parameters
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # Plot 4: Residuals analysis
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(exp_years, residuals, 'ko-', linewidth=2, markersize=6)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (exp - sim)', fontsize=12, fontweight='bold')
    plt.title('Model Residuals vs Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add text box with optimized parameters
    param_text = (
        f"Optimized Parameters (h⁻¹ or nM⁻¹h⁻¹):\n"
        f"kb0_40 = {kb0_forty_h:.2e}, kf0_40 = {kf0_forty_h:.2e}\n"
        f"kb1_40 = {kb1_forty_h:.2e}, kf1_40 = {kf1_forty_h:.2e}\n"
        f"kb0_42 = {kb0_fortytwo_h:.2e}, kf0_42 = {kf0_fortytwo_h:.2e}\n"
        f"kb1_42 = {kb1_fortytwo_h:.2e}, kf1_42 = {kf1_fortytwo_h:.2e}\n"
        f"base_40 = {baseline_ab40_rate:.2e}, base_42 = {baseline_ab42_rate:.2e}\n"
        f"RMS error = {rms_error:.4f}\n"
        f"Sum squared residuals = {final_error:.2e}\n"
        f"Function evaluations = {len(loss_history)}\n"
        f"Sim/Exp final ratio = {sim_ratios_interp[-1]/exp_ratios[-1]:.2f}"
    )
    plt.figtext(0.5, 0.02, param_text, ha="center", fontsize=11, 
                bbox={"facecolor":"lightgray", "alpha":0.8, "pad":8})
    
    plt.suptitle('Parameter Optimization Results: Fitting Plasma Aβ42/Aβ40 Ratio', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])  # Make room for parameter text
    plt.savefig('simulation_plots/agg_rate_experimental_fit_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sim_ratios_interp, sim_result

if __name__ == "__main__":
    # Run optimization
    optimal_params, final_error = optimize_parameters()
    
    # Analyze results
    sim_ratios, simulation_result = analyze_optimized_parameters(optimal_params)
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE")
    print(f"="*60)
    print(f"Results saved to: simulation_plots/agg_rate_experimental_fit_results.png")
    print(f"Final optimized parameters (h⁻¹ or nM⁻¹h⁻¹):")
    print(f"  kb0_40 = {optimal_params[0]:.4e}, kf0_40 = {optimal_params[4]:.4e}")
    print(f"  kb1_40 = {optimal_params[1]:.4e}, kf1_40 = {optimal_params[5]:.4e}")
    print(f"  kb0_42 = {optimal_params[2]:.4e}, kf0_42 = {optimal_params[6]:.4e}")
    print(f"  kb1_42 = {optimal_params[3]:.4e}, kf1_42 = {optimal_params[7]:.4e}")
    print(f"  base_40 = {optimal_params[8]:.4e}, base_42 = {optimal_params[9]:.4e}")
    print(f"Final sum of squared residuals: {final_error:.4e}")
    print(f"RMS error: {np.sqrt(final_error/len(exp_ratios)):.4f} (ratio units)")
    
    # Calculate and display the final simulation vs experimental ratio
    final_sim_ratio = sim_ratios[-1] if len(sim_ratios) > 0 else 0
    final_exp_ratio = exp_ratios[-1]
    ratio = final_sim_ratio / final_exp_ratio if final_exp_ratio > 0 else float('inf')
    print(f"Final concentration ratio (sim/exp): {ratio:.2f}")
    print(f"Target ratio should be close to 1.0 for good fit") 