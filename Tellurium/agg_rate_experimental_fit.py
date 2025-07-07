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

# Load the experimental data for ISF and CSF concentrations
isf_data = pd.read_csv('ISF_Monomers_pg_ml.csv')
csf_data = pd.read_csv('CSF_pg_ml.csv')

isf_years = np.array(isf_data['Year'])
isf_concentrations = np.array(isf_data['ISF_Monomers_pg_ml'])

csf_years = np.array(csf_data['Year'])
csf_concentrations = np.array(csf_data['CSF_pg_ml'])

# Create a list to store loss values and parameter history
loss_history = []
param_history = []

# Define simulation selections for ISF and CSF monomer concentrations
simulation_selections = ['time', '[AB42_Monomer]', '[AB42Mu_SAS]']

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

def transform_simulation_results(sim_result):
    """
    Transform simulation results to match the experimental data format.
    Convert to pg/mL units using the same scaling as in visualize_tellurium_simulation.py
    """
    # Volume scaling factors (same as in visualize_tellurium_simulation.py)
    volume_scale_factor_csf = 0.09875  # Division by volume for SAS concentration units
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    
    # Conversion factor from nM to pg/mL
    nM_to_pg = 4514.0
    
    # Extract time and concentrations
    sim_years = sim_result[:, 0] / 365 / 24  # Convert hours to years
    
    # Transform ISF AB42 monomer (index 1)
    ab42_isf_raw = sim_result[:, 1]
    ab42_isf_nM = ab42_isf_raw / volume_scale_factor_isf
    ab42_isf_pg_ml = ab42_isf_nM * nM_to_pg
    
    # Transform CSF SAS AB42 monomer (index 2)
    ab42_sas_raw = sim_result[:, 2]
    ab42_sas_nM = ab42_sas_raw / volume_scale_factor_csf
    ab42_sas_pg_ml = ab42_sas_nM * nM_to_pg
    
    return sim_years, ab42_isf_pg_ml, ab42_sas_pg_ml

def objective_function(log_params):
    """
    Objective function: sum of squared differences from experimental ISF and CSF concentrations
    
    log_params: [log(kb0_fortytwo), log(kb1_fortytwo), log(kf0_fortytwo), log(kf1_fortytwo), 
                asymp_fortytwo, forHill42, BackHill42]
    All rate parameters are log-transformed to ensure positivity
    Hill and Asymp parameters are kept in linear space
    """
    # Transform back from log space for rate parameters
    kb0_fortytwo_h = np.exp(log_params[0])
    kb1_fortytwo_h = np.exp(log_params[1])
    kf0_fortytwo_h = np.exp(log_params[2])
    kf1_fortytwo_h = np.exp(log_params[3])
    
    # Hill and Asymp parameters are in linear space
    asymp_fortytwo = log_params[4]  # Used for both forward and backward
    forHill42 = log_params[5]
    BackHill42 = log_params[6]

    # Use fixed values for AB40 parameters (from K_rates_extrapolate.py)
    kb0_forty_h = convert_backward_rate(2.7e-3)
    kb1_forty_h = convert_backward_rate(0.00001 / 3600)
    kf0_forty_h = convert_forward_rate(0.5 * 10**2)
    kf1_forty_h = convert_forward_rate(20.0)
    forAsymp40 = 0.3
    backAsymp40 = 0.3
    forHill40 = 2.0
    BackHill40 = 2.5

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
            forAsymp40=forAsymp40,
            forAsymp42=asymp_fortytwo,  # Use asymp_fortytwo for both forward and backward
            backAsymp40=backAsymp40,
            backAsymp42=asymp_fortytwo,  # Use asymp_fortytwo for both forward and backward
            forHill40=forHill40,
            forHill42=forHill42,
            BackHill40=BackHill40,
            BackHill42=BackHill42
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
        
        # Transform simulation results to match experimental data format
        sim_years, ab42_isf_pg_ml, ab42_sas_pg_ml = transform_simulation_results(sim_result)
        
        # Interpolate simulation results to match experimental time points
        f_isf = interp1d(sim_years, ab42_isf_pg_ml, bounds_error=False, fill_value="extrapolate")
        f_csf = interp1d(sim_years, ab42_sas_pg_ml, bounds_error=False, fill_value="extrapolate")
        
        sim_isf_interp = f_isf(isf_years)
        sim_csf_interp = f_csf(csf_years)
        
        # Calculate residuals for both ISF and CSF data
        isf_residuals = isf_concentrations - sim_isf_interp
        csf_residuals = csf_concentrations - sim_csf_interp
        
        # Combine errors (weighted equally for now)
        isf_error = np.sum(isf_residuals**2)
        csf_error = np.sum(csf_residuals**2)
        total_error = isf_error + csf_error
        
        # Store the error and parameters in history
        loss_history.append(total_error)
        param_history.append([
            kb0_fortytwo_h, kb1_fortytwo_h, kf0_fortytwo_h, kf1_fortytwo_h,
            asymp_fortytwo, forHill42, BackHill42
        ])
        
        # Print progress
        final_isf = sim_isf_interp[-1] if len(sim_isf_interp) > 0 else 0
        final_csf = sim_csf_interp[-1] if len(sim_csf_interp) > 0 else 0
        exp_final_isf = isf_concentrations[-1]
        exp_final_csf = csf_concentrations[-1]
        isf_ratio = final_isf / exp_final_isf if exp_final_isf > 0 else float('inf')
        csf_ratio = final_csf / exp_final_csf if exp_final_csf > 0 else float('inf')
        
        print(f"kb0_42={kb0_fortytwo_h:.2e}, kb1_42={kb1_fortytwo_h:.2e}, "
              f"kf0_42={kf0_fortytwo_h:.2e}, kf1_42={kf1_fortytwo_h:.2e}, "
              f"asymp_42={asymp_fortytwo:.3f}, forHill42={forHill42:.3f}, BackHill42={BackHill42:.3f}, "
              f"ISF sim/exp={isf_ratio:.2f}, CSF sim/exp={csf_ratio:.2f}, error={total_error:.2e}")
            
        return total_error
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e10  # Large penalty for any errors

def optimize_parameters():
    """Simple parameter optimization with early stopping and parameter bounds"""
    
    print("Starting parameter optimization for ISF and CSF Aβ42 monomer concentrations...")
    print("Using current PK_Geerts values as initial conditions")
    print(f"Experimental ISF data: {isf_concentrations[0]:.1f} to {isf_concentrations[-1]:.1f} pg/mL")
    print(f"Experimental CSF data: {csf_concentrations[0]:.1f} to {csf_concentrations[-1]:.1f} pg/mL")
    print(f"Time range: {isf_years[0]:.1f} to {isf_years[-1]:.1f} years")
    
    # Current values from PK_Geerts.csv and K_rates_extrapolate.py, converted to h⁻¹ or nM⁻¹h⁻¹
    current_kb0_fortytwo_h = convert_backward_rate(12.7e-3)
    current_kb1_fortytwo_h = convert_backward_rate(0.00001 / 3600)
    current_kf0_fortytwo_h = convert_forward_rate(9.9 * 10**2)
    current_kf1_fortytwo_h = convert_forward_rate(38.0)
    current_asymp_fortytwo = 2.0 # This is the forward and backward asymptotic value
    current_forHill42 = 3.0
    current_BackHill42 = 3.0

    print(f"\nCurrent initial values:")
    print(f"Rate parameters (h⁻¹ or nM⁻¹h⁻¹):")
    print(f"  kb0_42 = {current_kb0_fortytwo_h:.4e}")
    print(f"  kb1_42 = {current_kb1_fortytwo_h:.4e}")
    print(f"  kf0_42 = {current_kf0_fortytwo_h:.4e}")
    print(f"  kf1_42 = {current_kf1_fortytwo_h:.4e}")
    print(f"Hill and Asymp parameters:")
    print(f"  asymp_42 = {current_asymp_fortytwo:.3f}")
    print(f"  forHill42 = {current_forHill42:.3f}")
    print(f"  BackHill42 = {current_BackHill42:.3f}")

    # Define reasonable parameter bounds
    # Rate parameters: log-transformed with bounds
    rate_bounds = [
        (np.log(max(1e-05, current_kb0_fortytwo_h * 0.001)), np.log(current_kb0_fortytwo_h * 100)),
        (np.log(max(1e-05, current_kb1_fortytwo_h * 0.001)), np.log(current_kb1_fortytwo_h * 100)),
        (np.log(max(1e-05, current_kf0_fortytwo_h * 0.001)), np.log(current_kf0_fortytwo_h * 100)),
        (np.log(max(1e-05, current_kf1_fortytwo_h * 0.001)), np.log(current_kf1_fortytwo_h * 100)),
    ]
    
    # Hill and Asymp parameters: linear space with reasonable bounds
    hill_asymp_bounds = [
        (0.1, 5.0),    # asymp_fortytwo
        (0.5, 8.0),    # forHill42
        (0.5, 5.0),    # BackHill42
    ]
    
    # Combine all bounds
    bounds = rate_bounds + hill_asymp_bounds
    
    print(f"\nParameter bounds:")
    print(f"Rate parameters (h⁻¹ or nM⁻¹h⁻¹):")
    print(f"  kb0_42: {np.exp(bounds[0][0]):.1e} to {np.exp(bounds[0][1]):.1e}")
    print(f"  kb1_42: {np.exp(bounds[1][0]):.1e} to {np.exp(bounds[1][1]):.1e}")
    print(f"  kf0_42: {np.exp(bounds[2][0]):.1e} to {np.exp(bounds[2][1]):.1e}")
    print(f"  kf1_42: {np.exp(bounds[3][0]):.1e} to {np.exp(bounds[3][1]):.1e}")
    print(f"Hill and Asymp parameters:")
    print(f"  asymp_42: {bounds[4][0]:.1f} to {bounds[4][1]:.1f}")
    print(f"  forHill42: {bounds[5][0]:.1f} to {bounds[5][1]:.1f}")
    print(f"  BackHill42: {bounds[6][0]:.1f} to {bounds[6][1]:.1f}")
    
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
        np.log(current_kb0_fortytwo_h),
        np.log(current_kb1_fortytwo_h),
        np.log(current_kf0_fortytwo_h),
        np.log(current_kf1_fortytwo_h),
        current_asymp_fortytwo,
        current_forHill42,
        current_BackHill42,
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
        opt_real = []
        for i in range(4):  # First 4 are log-transformed rate parameters
            opt_real.append(np.exp(final_params[i]))
        for i in range(4, 7):  # Last 3 are linear Hill/Asymp parameters
            opt_real.append(final_params[i])
        
        print(f"\nOptimization complete!")
        print(f"Result: kb0_42={opt_real[0]:.2e}, kb1_42={opt_real[1]:.2e}, ")
        print(f"        kf0_42={opt_real[2]:.2e}, kf1_42={opt_real[3]:.2e}, ")
        print(f"        asymp_42={opt_real[4]:.3f}, forHill42={opt_real[5]:.3f}, BackHill42={opt_real[6]:.3f}")
        print(f"Error: {final_error:.2e}, Success: {result.success}")
        
        if not result.success:
            print("Warning: Optimization may not have converged!")
            
        return opt_real, final_error
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        raise RuntimeError("Optimization failed!")

def analyze_optimized_parameters(optimal_params):
    """Analyze the optimized parameters and plot results"""
    kb0_fortytwo_h, kb1_fortytwo_h, kf0_fortytwo_h, kf1_fortytwo_h, \
    asymp_fortytwo, forHill42, BackHill42 = optimal_params
    
    print(f"\n" + "="*60)
    print(f"ANALYZING OPTIMIZED PARAMETERS")
    print(f"="*60)
    
    # Calculate rates with optimized parameters
    rates = calculate_k_rates(
        original_kb0_forty=convert_backward_rate(2.7e-3) / 3600,
        original_kb1_forty=convert_backward_rate(0.00001 / 3600) / 3600,
        original_kb0_fortytwo=kb0_fortytwo_h / 3600,
        original_kb1_fortytwo=kb1_fortytwo_h / 3600,
        original_kf0_forty=convert_forward_rate(0.5 * 10**2) / 3.6e-6,
        original_kf1_forty=convert_forward_rate(20.0) / 3.6e-6,
        original_kf0_fortytwo=kf0_fortytwo_h / 3.6e-6,
        original_kf1_fortytwo=kf1_fortytwo_h / 3.6e-6,
        forAsymp40=0.3,
        forAsymp42=asymp_fortytwo,
        backAsymp40=0.3,
        backAsymp42=asymp_fortytwo,
        forHill40=2.0,
        forHill42=forHill42,
        BackHill40=2.5,
        BackHill42=BackHill42
    )
    
    # Run simulation with optimized parameters
    rr.reset()
    setup_integrator()
    
    rr.k_O2_M_forty = convert_backward_rate(2.7e-3)
    rr.k_O3_O2_forty = convert_backward_rate(0.00001 / 3600)
    rr.k_O2_M_fortytwo = kb0_fortytwo_h
    rr.k_O3_O2_fortytwo = kb1_fortytwo_h
    rr.k_M_O2_forty = convert_forward_rate(0.5 * 10**2)
    rr.k_O2_O3_forty = convert_forward_rate(20.0)
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
    
    # Transform simulation results
    sim_years, ab42_isf_pg_ml, ab42_sas_pg_ml = transform_simulation_results(sim_result)
    
    # Interpolate to experimental time points for error calculation
    f_isf = interp1d(sim_years, ab42_isf_pg_ml, bounds_error=False, fill_value="extrapolate")
    f_csf = interp1d(sim_years, ab42_sas_pg_ml, bounds_error=False, fill_value="extrapolate")

    sim_isf_interp = f_isf(isf_years)
    sim_csf_interp = f_csf(csf_years)
    
    # Calculate final error metrics
    isf_residuals = isf_concentrations - sim_isf_interp
    csf_residuals = csf_concentrations - sim_csf_interp
    final_error = np.sum(isf_residuals**2) + np.sum(csf_residuals**2)
    rms_error = np.sqrt(np.mean(isf_residuals**2)) # RMS error for ISF
    
    # Print analysis
    print(f"Final sum of squared residuals: {final_error:.2e}")
    print(f"RMS error (ISF): {rms_error:.4f} (pg/mL)")
    print(f"Mean experimental ISF: {np.mean(isf_concentrations):.3f}")
    print(f"Mean simulated ISF: {np.mean(sim_isf_interp):.3f}")
    print(f"RMS error (CSF): {np.sqrt(np.mean(csf_residuals**2)):.4f} (pg/mL)")
    print(f"Mean experimental CSF: {np.mean(csf_concentrations):.3f}")
    print(f"Mean simulated CSF: {np.mean(sim_csf_interp):.3f}")
    
    # Create comprehensive results plot
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Experimental ISF data vs model fit
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(isf_years, isf_concentrations, color='red', label='Experimental ISF', s=80, zorder=3, alpha=0.8)
    plt.plot(sim_years, ab42_isf_pg_ml, 'b-', linewidth=3, label='Optimized ISF', alpha=0.8)
    plt.scatter(isf_years, sim_isf_interp, color='blue', marker='x', s=60, 
                label='Model at exp. ISF points', zorder=3, linewidth=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Aβ42 Monomer (pg/mL)', fontsize=12, fontweight='bold')
    plt.title('Experimental ISF vs Optimized Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
    # Plot 2: Experimental CSF data vs model fit
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(csf_years, csf_concentrations, color='red', label='Experimental CSF', s=80, zorder=3, alpha=0.8)
    plt.plot(sim_years, ab42_sas_pg_ml, 'b-', linewidth=3, label='Optimized CSF', alpha=0.8)
    plt.scatter(csf_years, sim_csf_interp, color='blue', marker='x', s=60, 
                label='Model at exp. CSF points', zorder=3, linewidth=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Aβ42 Monomer (pg/mL)', fontsize=12, fontweight='bold')
    plt.title('Experimental CSF vs Optimized Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
    # Plot 3: Loss history during optimization
    ax3 = plt.subplot(2, 3, 3)
    iterations = np.arange(1, len(loss_history) + 1)
    plt.semilogy(iterations, loss_history, 'r-', linewidth=2)
    plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (Sum of Squared Residuals)', fontsize=12, fontweight='bold')
    plt.title('Optimization Loss History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Parameter evolution during optimization
    ax4 = plt.subplot(2, 3, 4)
    param_array = np.array(param_history)
    if len(param_array) > 0:
        iterations = np.arange(1, len(param_array) + 1)
        # Plot rate parameters (log scale)
        plt.plot(iterations, param_array[:, 0], 'r-', linewidth=2, label='kb0_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 1], 'g-', linewidth=2, label='kb1_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 2], 'b-', linewidth=2, label='kf0_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 3], 'c-', linewidth=2, label='kf1_42', alpha=0.7)
        plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Rate Parameter Values (h⁻¹ or nM⁻¹h⁻¹)', fontsize=12, fontweight='bold')
        plt.title('Rate Parameter Evolution During Optimization', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # Plot 5: Residuals analysis
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(isf_years, isf_residuals, 'ro-', linewidth=2, markersize=6, label='ISF Residuals')
    plt.plot(csf_years, csf_residuals, 'bo-', linewidth=2, markersize=6, label='CSF Residuals')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (exp - sim)', fontsize=12, fontweight='bold')
    plt.title('Model Residuals vs Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Hill and Asymp parameters evolution
    ax6 = plt.subplot(2, 3, 6)
    if len(param_array) > 0:
        plt.plot(iterations, param_array[:, 4], 'm-', linewidth=2, label='asymp_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 5], 'y-', linewidth=2, label='forHill42', alpha=0.7)
        plt.plot(iterations, param_array[:, 6], 'k-', linewidth=2, label='BackHill42', alpha=0.7)
        plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Parameter Values', fontsize=12, fontweight='bold')
        plt.title('Hill and Asymp Parameter Evolution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
    
    # Add text box with optimized parameters
    param_text = (
        f"Optimized Parameters:\n"
        f"Rate parameters (h⁻¹ or nM⁻¹h⁻¹):\n"
        f"kb0_42 = {kb0_fortytwo_h:.2e}, kf0_42 = {kf0_fortytwo_h:.2e}\n"
        f"kb1_42 = {kb1_fortytwo_h:.2e}, kf1_42 = {kf1_fortytwo_h:.2e}\n"
        f"Hill and Asymp parameters:\n"
        f"asymp_42 = {asymp_fortytwo:.3f}, forHill42 = {forHill42:.3f}\n"
        f"BackHill42 = {BackHill42:.3f}\n"
        f"RMS error (ISF) = {rms_error:.4f}\n"
        f"RMS error (CSF) = {np.sqrt(np.mean(csf_residuals**2)):.4f}\n"
        f"Sum squared residuals = {final_error:.2e}\n"
        f"Function evaluations = {len(loss_history)}\n"
        f"ISF final ratio = {sim_isf_interp[-1]/isf_concentrations[-1]:.2f}\n"
        f"CSF final ratio = {sim_csf_interp[-1]/csf_concentrations[-1]:.2f}"
    )
    plt.figtext(0.5, 0.02, param_text, ha="center", fontsize=11, 
                bbox={"facecolor":"lightgray", "alpha":0.8, "pad":8})
    
    plt.suptitle('Parameter Optimization Results: Fitting ISF and CSF Aβ42 Monomer Concentrations', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])  # Make room for parameter text
    plt.savefig('simulation_plots/agg_rate_experimental_fit_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sim_isf_interp, sim_csf_interp, sim_result

def calculate_percent_change_table(optimal_params):
    """Calculate and display percent changes in optimized parameters"""
    import pandas as pd
    
    kb0_fortytwo_h, kb1_fortytwo_h, kf0_fortytwo_h, kf1_fortytwo_h, \
    asymp_fortytwo, forHill42, BackHill42 = optimal_params
    
    # Initial parameter values (from optimization code)
    initial_kb0_fortytwo_h = convert_backward_rate(12.7e-3)
    initial_kb1_fortytwo_h = convert_backward_rate(0.00001 / 3600)
    initial_kf0_fortytwo_h = convert_forward_rate(9.9 * 10**2)
    initial_kf1_fortytwo_h = convert_forward_rate(38.0)
    initial_asymp_fortytwo = 2.0
    initial_forHill42 = 3.0
    initial_BackHill42 = 3.0

    # Calculate percent changes
    def calculate_percent_change(initial, final):
        return ((final - initial) / initial) * 100

    # Create the data for the table
    data = {
        'Parameter': [
            'kb0_42 (h⁻¹)',
            'kb1_42 (h⁻¹)', 
            'kf0_42 (nM⁻¹h⁻¹)',
            'kf1_42 (nM⁻¹h⁻¹)',
            'asymp_42',
            'forHill42',
            'BackHill42'
        ],
        'Initial Value': [
            f"{initial_kb0_fortytwo_h:.4e}",
            f"{initial_kb1_fortytwo_h:.4e}",
            f"{initial_kf0_fortytwo_h:.4e}",
            f"{initial_kf1_fortytwo_h:.4e}",
            f"{initial_asymp_fortytwo:.3f}",
            f"{initial_forHill42:.3f}",
            f"{initial_BackHill42:.3f}"
        ],
        'Final Value': [
            f"{kb0_fortytwo_h:.4e}",
            f"{kb1_fortytwo_h:.4e}",
            f"{kf0_fortytwo_h:.4e}",
            f"{kf1_fortytwo_h:.4e}",
            f"{asymp_fortytwo:.3f}",
            f"{forHill42:.3f}",
            f"{BackHill42:.3f}"
        ],
        'Percent Change (%)': [
            f"{calculate_percent_change(initial_kb0_fortytwo_h, kb0_fortytwo_h):.1f}",
            f"{calculate_percent_change(initial_kb1_fortytwo_h, kb1_fortytwo_h):.1f}",
            f"{calculate_percent_change(initial_kf0_fortytwo_h, kf0_fortytwo_h):.1f}",
            f"{calculate_percent_change(initial_kf1_fortytwo_h, kf1_fortytwo_h):.1f}",
            f"{calculate_percent_change(initial_asymp_fortytwo, asymp_fortytwo):.1f}",
            f"{calculate_percent_change(initial_forHill42, forHill42):.1f}",
            f"{calculate_percent_change(initial_BackHill42, BackHill42):.1f}"
        ]
    }

    # Create DataFrame and display
    df = pd.DataFrame(data)

    print(f"\n" + "="*80)
    print("PARAMETER OPTIMIZATION: PERCENT CHANGES")
    print("="*80)
    print(df.to_string(index=False))

    # Calculate raw percent changes for analysis
    percent_changes = [
        calculate_percent_change(initial_kb0_fortytwo_h, kb0_fortytwo_h),
        calculate_percent_change(initial_kb1_fortytwo_h, kb1_fortytwo_h),
        calculate_percent_change(initial_kf0_fortytwo_h, kf0_fortytwo_h),
        calculate_percent_change(initial_kf1_fortytwo_h, kf1_fortytwo_h),
        calculate_percent_change(initial_asymp_fortytwo, asymp_fortytwo),
        calculate_percent_change(initial_forHill42, forHill42),
        calculate_percent_change(initial_BackHill42, BackHill42)
    ]

    # Also create a summary
    print("\n\nSUMMARY:")
    print("="*40)

    largest_increase = max([pc for pc in percent_changes if pc > 0]) if any(pc > 0 for pc in percent_changes) else 0
    largest_decrease = min([pc for pc in percent_changes if pc < 0]) if any(pc < 0 for pc in percent_changes) else 0

    print(f"Largest increase: {largest_increase:.1f}%")
    print(f"Largest decrease: {largest_decrease:.1f}%")
    print(f"Average absolute change: {np.mean([abs(pc) for pc in percent_changes]):.1f}%")

    # Parameters that increased vs decreased
    increased = [data['Parameter'][i] for i, pc in enumerate(percent_changes) if pc > 0]
    decreased = [data['Parameter'][i] for i, pc in enumerate(percent_changes) if pc < 0]

    if increased:
        print(f"\nParameters that increased ({len(increased)}):")
        for param in increased:
            idx = data['Parameter'].index(param)
            print(f"  - {param}: {percent_changes[idx]:.1f}%")

    if decreased:
        print(f"\nParameters that decreased ({len(decreased)}):")
        for param in decreased:
            idx = data['Parameter'].index(param)
            print(f"  - {param}: {percent_changes[idx]:.1f}%")

if __name__ == "__main__":
    # Run optimization
    optimal_params, final_error = optimize_parameters()
    
    # Analyze results
    sim_isf_interp, sim_csf_interp, simulation_result = analyze_optimized_parameters(optimal_params)
    
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION COMPLETE")
    print(f"="*60)
    print(f"Results saved to: simulation_plots/agg_rate_experimental_fit_results.png")
    print(f"Final optimized parameters:")
    print(f"  kb0_42 = {optimal_params[0]:.4e}, kf0_42 = {optimal_params[2]:.4e}")
    print(f"  kb1_42 = {optimal_params[1]:.4e}, kf1_42 = {optimal_params[3]:.4e}")
    print(f"  asymp_42 = {optimal_params[4]:.3f}, forHill42 = {optimal_params[5]:.3f}")
    print(f"  BackHill42 = {optimal_params[6]:.3f}")
    print(f"Final sum of squared residuals: {final_error:.4e}")
    print(f"RMS error (ISF): {np.sqrt(final_error/len(isf_concentrations)):.4f} (pg/mL)")
    print(f"RMS error (CSF): {np.sqrt(final_error/len(csf_concentrations)):.4f} (pg/mL)")
    
    # Calculate and display the final simulation vs experimental ratios
    final_sim_isf = sim_isf_interp[-1] if len(sim_isf_interp) > 0 else 0
    final_sim_csf = sim_csf_interp[-1] if len(sim_csf_interp) > 0 else 0
    final_exp_isf = isf_concentrations[-1]
    final_exp_csf = csf_concentrations[-1]
    isf_ratio = final_sim_isf / final_exp_isf if final_exp_isf > 0 else float('inf')
    csf_ratio = final_sim_csf / final_exp_csf if final_exp_csf > 0 else float('inf')
    print(f"Final ISF ratio (sim/exp): {isf_ratio:.2f}")
    print(f"Final CSF ratio (sim/exp): {csf_ratio:.2f}")
    print(f"Target ratios should be close to 1.0 for good fit")
    
    # Calculate and display percent changes
    calculate_percent_change_table(optimal_params) 