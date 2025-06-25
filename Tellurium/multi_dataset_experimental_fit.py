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

# Load all three experimental datasets
print("Loading experimental datasets...")

# Dataset 1: Plasma ratio
plasma_data = pd.read_csv('Plasma_Ratio_Geerts.csv')
plasma_years = np.array(plasma_data['Year'])
plasma_ratios = np.array(plasma_data['Plasma_Ratio'])

# Dataset 2: CSF Aβ42 concentrations
csf_data = pd.read_csv('CSF_pg_ml.csv')
csf_years = np.array(csf_data['Year'])
csf_concentrations = np.array(csf_data['CSF_pg_ml'])

# Dataset 3: ISF Aβ42 monomer concentrations  
isf_data = pd.read_csv('ISF_Monomers_pg_ml.csv')
isf_years = np.array(isf_data['Year'])
isf_concentrations = np.array(isf_data['ISF_Monomers_pg_ml'])

print(f"Plasma dataset: {len(plasma_years)} points, years {plasma_years[0]:.1f}-{plasma_years[-1]:.1f}")
print(f"CSF dataset: {len(csf_years)} points, years {csf_years[0]:.1f}-{csf_years[-1]:.1f}")  
print(f"ISF dataset: {len(isf_years)} points, years {isf_years[0]:.1f}-{isf_years[-1]:.1f}")

# Create lists to store loss values and parameter history
loss_history = []
param_history = []
plasma_error_history = []
csf_error_history = []
isf_error_history = []

# Define simulation selections for all three datasets
simulation_selections = [
    'time', 
    '[AB42Mu_Brain_Plasma]', '[AB40Mu_Brain_Plasma]',  # For plasma ratio
    '[AB42Mu_SAS]',  # For CSF concentrations
    '[AB42_Monomer]'   # For ISF concentrations
]

# Load the SBML model
# Current model is combined_master_model_garai.xml which has kb1_fortytwo_h = 1.08
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model_garai.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

def setup_integrator():
    """Setup robust integrator settings"""
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8  
    rr.integrator.setValue('stiff', True)

def convert_nM_to_pg_ml(concentration_nM, molecular_weight=4514):
    """Convert concentration from nM to pg/ml for Aβ42"""
    # MW of Aβ42 ≈ 4514 g/mol
    # 1 nM = 1e-9 mol/L = 1e-9 * MW * 1e12 pg/1e-3 L = MW pg/ml
    return concentration_nM * molecular_weight

def objective_function(log_params, weights=[1.0, 1.0, 1.0]):
    """
    Multi-objective function: weighted sum of squared differences for all three datasets
    
    log_params: [log(kb0_forty), log(kb1_forty), log(kb0_fortytwo), log(kb1_fortytwo), 
                 log(baseline_ab40_rate), log(baseline_ab42_rate)]
    weights: [plasma_weight, csf_weight, isf_weight] for balancing different datasets
    """
    # Transform back from log space
    params = [np.exp(x) for x in log_params]
    kb0_forty_h, kb1_forty_h, kb0_fortytwo_h, kb1_fortytwo_h = params[:4]
    baseline_ab40_rate, baseline_ab42_rate = params[4:6]

    try:
        # Convert rates for calculate_k_rates function
        kb0_forty_s = kb0_forty_h / 3600
        kb1_forty_s = kb1_forty_h / 3600
        kb0_fortytwo_s = kb0_fortytwo_h / 3600
        kb1_fortytwo_s = kb1_fortytwo_h / 3600
        
        # Calculate all rates for these parameters
        rates = calculate_k_rates(
            original_kb0_forty=kb0_forty_s,
            original_kb1_forty=kb1_forty_s,
            original_kb0_fortytwo=kb0_fortytwo_s,
            original_kb1_fortytwo=kb1_fortytwo_s,
            baseline_ab40_plaque_rate=baseline_ab40_rate,
            baseline_ab42_plaque_rate=baseline_ab42_rate
        )
        
        # Run simulation with more robust approach
        rr.reset()
        setup_integrator()
        
        # Set parameters
        rr.k_O2_M_forty = kb0_forty_h
        rr.k_O3_O2_forty = kb1_forty_h
        rr.k_O2_M_fortytwo = kb0_fortytwo_h
        rr.k_O3_O2_fortytwo = kb1_fortytwo_h
        
        # Update related rates
        for key, value in rates.items():
            try:
                setattr(rr, key, value)
            except Exception as e:
                pass
        
        # Run simulation to cover all experimental time ranges
        t_sim = 100*365*24  # 100 years in hours
        n_points = 200
        sim_result = rr.simulate(0, t_sim, n_points, selections=simulation_selections)
        
        # Check for simulation errors
        if np.any(np.isnan(sim_result)) or np.any(np.isinf(sim_result)):
            return 1e8  # Smaller penalty
        
        # Convert time to years
        sim_years = sim_result[:, 0] / 365 / 24
        
        # Extract simulation results for each dataset
        sim_plasma_ab42_raw = sim_result[:, 1]  # AB42Mu_Brain_Plasma
        sim_plasma_ab40_raw = sim_result[:, 2]  # AB40Mu_Brain_Plasma
        sim_csf_ab42_raw = sim_result[:, 3]     # AB42Mu_SAS
        sim_isf_ab42_raw = sim_result[:, 4]     # AB42_Monomer
        
        # Apply volume scaling to get proper concentration units
        volume_scale_factor_isf = 0.2505  
        volume_scale_factor_csf = 0.09875 
        sim_plasma_ab42 = sim_plasma_ab42_raw 
        sim_plasma_ab40 = sim_plasma_ab40_raw   
        sim_csf_ab42 = sim_csf_ab42_raw / volume_scale_factor_csf
        sim_isf_ab42 = sim_isf_ab42_raw / volume_scale_factor_isf   
        
        # Calculate plasma ratio with safety check
        sim_plasma_ratio = np.divide(sim_plasma_ab42, sim_plasma_ab40, 
                                   out=np.zeros_like(sim_plasma_ab42), 
                                   where=sim_plasma_ab40!=0)
        
        # Convert CSF and ISF concentrations from nM to pg/ml
        sim_csf_ab42_pg_ml = convert_nM_to_pg_ml(sim_csf_ab42)
        sim_isf_ab42_pg_ml = convert_nM_to_pg_ml(sim_isf_ab42)
        
        # Interpolate simulation results to experimental time points
        try:
            f_plasma = interp1d(sim_years, sim_plasma_ratio, bounds_error=False, fill_value="extrapolate")
            f_csf = interp1d(sim_years, sim_csf_ab42_pg_ml, bounds_error=False, fill_value="extrapolate")
            f_isf = interp1d(sim_years, sim_isf_ab42_pg_ml, bounds_error=False, fill_value="extrapolate")
            
            sim_plasma_interp = f_plasma(plasma_years)
            sim_csf_interp = f_csf(csf_years)
            sim_isf_interp = f_isf(isf_years)
        except ValueError:
            return 1e8  # Interpolation failed
        
        # Calculate individual errors with better normalization
        # Use relative errors to handle different scales
        plasma_mean = np.mean(plasma_ratios)
        csf_mean = np.mean(csf_concentrations)
        isf_mean = np.mean(isf_concentrations)
        
        # Relative squared errors
        plasma_rel_residuals = (plasma_ratios - sim_plasma_interp) / plasma_mean
        csf_rel_residuals = (csf_concentrations - sim_csf_interp) / csf_mean
        isf_rel_residuals = (isf_concentrations - sim_isf_interp) / isf_mean
        
        plasma_error = np.sum(plasma_rel_residuals**2)
        csf_error = np.sum(csf_rel_residuals**2) 
        isf_error = np.sum(isf_rel_residuals**2)
        
        # Check for invalid errors
        if np.isnan(plasma_error) or np.isnan(csf_error) or np.isnan(isf_error):
            return 1e8
        if np.isinf(plasma_error) or np.isinf(csf_error) or np.isinf(isf_error):
            return 1e8
        
        # Weighted total error with more balanced scaling
        total_error = weights[0] * plasma_error + weights[1] * csf_error + weights[2] * isf_error
        
        # Store histories
        loss_history.append(total_error)
        plasma_error_history.append(plasma_error)
        csf_error_history.append(csf_error)
        isf_error_history.append(isf_error)
        param_history.append(params)
        
        # Print progress (less verbose)
        if len(loss_history) % 10 == 1:  # Print every 10th evaluation
            print(f"Eval {len(loss_history)}: Plasma: {plasma_error:.2e}, CSF: {csf_error:.2e}, ISF: {isf_error:.2e}, Total: {total_error:.2e}")
            
        return total_error
        
    except Exception as e:
        print(f"Objective function failed: {str(e)[:50]}...")
        return 1e8  # Smaller penalty

def test_parameter_sensitivity():
    """Test if changing parameters actually affects the objective function"""
    print("\n" + "="*60)
    print("TESTING PARAMETER SENSITIVITY")
    print("="*60)
    
    # Get initial parameter values
    kb0_forty_h = 2.7e-3
    kb1_forty_h = 0.00001 
    kb0_fortytwo_h = 12.7e-3
    kb1_fortytwo_h = 1.08 
    baseline_ab40_rate = 0.000005
    baseline_ab42_rate = 0.00005
    
    # Test log space parameters
    initial_log_params = [
        np.log(kb0_forty_h), np.log(kb1_forty_h),
        np.log(kb0_fortytwo_h), np.log(kb1_fortytwo_h),
        np.log(baseline_ab40_rate), np.log(baseline_ab42_rate),
    ]
    
    # Calculate baseline error
    baseline_error = objective_function(initial_log_params)
    print(f"Baseline error with initial parameters: {baseline_error:.4e}")
    
    # Test sensitivity by perturbing each parameter
    param_names = ['kb0_40', 'kb1_40', 'kb0_42', 'kb1_42', 'base_40', 'base_42']
    
    for i, param_name in enumerate(param_names):
        print(f"\nTesting sensitivity to {param_name}:")
        
        # Test increasing parameter by 10%
        test_params = initial_log_params.copy()
        test_params[i] = np.log(np.exp(initial_log_params[i]) * 1.1)  # 10% increase
        error_up = objective_function(test_params)
        
        # Test decreasing parameter by 10%
        test_params = initial_log_params.copy()
        test_params[i] = np.log(np.exp(initial_log_params[i]) * 0.9)  # 10% decrease
        error_down = objective_function(test_params)
        
        print(f"  +10%: {error_up:.4e} (change: {error_up - baseline_error:.4e})")
        print(f"  -10%: {error_down:.4e} (change: {error_down - baseline_error:.4e})")
        
        # Calculate numerical gradient
        gradient = (error_up - error_down) / (0.2 * np.exp(initial_log_params[i]))
        print(f"  Numerical gradient: {gradient:.4e}")

def optimize_parameters(weights=[1.0, 1.0, 1.0]):
    """Multi-objective parameter optimization with early stopping and parameter bounds"""
    
    print("Starting multi-dataset parameter optimization...")
    print(f"Dataset weights - Plasma: {weights[0]}, CSF: {weights[1]}, ISF: {weights[2]}")
    
    # Test parameter sensitivity first
    #test_parameter_sensitivity()
    
    # Initial parameter values (same as single-objective case)
    current_kb0_forty_h = 2.7e-3
    current_kb1_forty_h = 0.00001 
    current_kb0_fortytwo_h = 12.7e-3
    current_kb1_fortytwo_h = 1.08 
    current_baseline_ab40_rate = 0.000005
    current_baseline_ab42_rate = 0.00005

    print(f"\nInitial parameter values (h⁻¹):")
    print(f"  kb0_40 = {current_kb0_forty_h:.4e}, kb1_40 = {current_kb1_forty_h:.4e}")
    print(f"  kb0_42 = {current_kb0_fortytwo_h:.4e}, kb1_42 = {current_kb1_fortytwo_h:.4e}")
    print(f"  base_40 = {current_baseline_ab40_rate:.4e}, base_42 = {current_baseline_ab42_rate:.4e}")

    # Define more conservative parameter bounds to avoid simulation failures
    bounds = [
        (np.log(1e-05), np.log(current_kb0_forty_h * 10)),
        (np.log(1e-05), np.log(current_kb1_forty_h * 10)),
        (np.log(1e-05), np.log(current_kb0_fortytwo_h * 10)),
        (np.log(1e-05), np.log(current_kb1_fortytwo_h * 10)),
        (np.log(current_baseline_ab40_rate), np.log(current_baseline_ab40_rate * 10)),
        (np.log(current_baseline_ab42_rate), np.log(current_baseline_ab42_rate * 10)),
    ]
    
    print(f"\nParameter bounds (conservative to avoid simulation failures):")
    for i, (param_name, bound) in enumerate(zip(['kb0_40', 'kb1_40', 'kb0_42', 'kb1_42', 'base_40', 'base_42'], bounds)):
        print(f"  {param_name}: {np.exp(bound[0]):.2e} to {np.exp(bound[1]):.2e}")
    
    # Clear histories
    global loss_history, param_history, plasma_error_history, csf_error_history, isf_error_history
    loss_history = []
    param_history = []
    plasma_error_history = []
    csf_error_history = []
    isf_error_history = []
    
    # Starting point
    starting_point_log = [
        np.log(current_kb0_forty_h), np.log(current_kb1_forty_h),
        np.log(current_kb0_fortytwo_h), np.log(current_kb1_fortytwo_h),
        np.log(current_baseline_ab40_rate), np.log(current_baseline_ab42_rate),
    ]
    
    print(f"\nStarting multi-objective optimization...")
    print(f"Using more robust approach with conservative bounds")
    
    try:
        # First try with Nelder-Mead (more robust to simulation failures)
        print("Trying Nelder-Mead optimization (robust to noisy objectives)...")
        
        result_nm = minimize(
            lambda log_params: objective_function(log_params, weights),
            starting_point_log,
            method='Nelder-Mead',
            options={
                'maxiter': 100,
                'disp': True,
                'fatol': 1e-6,  # Function absolute tolerance
                'xatol': 1e-6   # Parameter absolute tolerance
            }
        )
        
        print(f"Nelder-Mead result: error={result_nm.fun:.4e}, success={result_nm.success}")
        
        # If Nelder-Mead worked, try to refine with L-BFGS-B
        if result_nm.success and result_nm.fun < 1e7:  # Only if we got a reasonable result
            print("Refining with L-BFGS-B...")
            
            # Check bounds for refined result
            refined_start = result_nm.x
            for i, (val, bound) in enumerate(zip(refined_start, bounds)):
                refined_start[i] = np.clip(val, bound[0], bound[1])
            
            result_lbfgs = minimize(
                lambda log_params: objective_function(log_params, weights),
                refined_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 50,  # Just refinement
                    'disp': True,
                    'gtol': 1e-6,
                    'ftol': 1e-9
                }
            )
            
            print(f"L-BFGS-B refinement: error={result_lbfgs.fun:.4e}, success={result_lbfgs.success}")
            
            # Use the better result
            if result_lbfgs.success and result_lbfgs.fun < result_nm.fun:
                print("Using L-BFGS-B refined result")
                final_result = result_lbfgs
            else:
                print("Using Nelder-Mead result")
                final_result = result_nm
        else:
            print("Using Nelder-Mead result (L-BFGS-B skipped)")
            final_result = result_nm
        
        final_params = final_result.x
        final_error = final_result.fun
        
        # Convert optimized parameters back to real space
        opt_real = [np.exp(x) for x in final_params]
        print(f"\nMulti-objective optimization complete!")
        print(f"Final parameters: kb0_40={opt_real[0]:.2e}, kb1_40={opt_real[1]:.2e}, kb0_42={opt_real[2]:.2e}, kb1_42={opt_real[3]:.2e}")
        print(f"                  base_40={opt_real[4]:.2e}, base_42={opt_real[5]:.2e}")
        print(f"Total weighted error: {final_error:.2e}, Success: {final_result.success}")
        print(f"Number of function evaluations: {final_result.nfev}")
        
        if hasattr(final_result, 'njev'):
            print(f"Number of gradient evaluations: {final_result.njev}")
        
        if not final_result.success:
            print("Warning: Optimization may not have converged!")
            if hasattr(final_result, 'message'):
                print(f"Message: {final_result.message}")
            
        return opt_real, final_error
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        raise RuntimeError("Optimization failed!")

def analyze_multi_dataset_fit(optimal_params, weights=[1.0, 1.0, 1.0]):
    """Analyze the optimized parameters and create comprehensive plots for all datasets"""
    kb0_forty_h, kb1_forty_h, kb0_fortytwo_h, kb1_fortytwo_h, \
    baseline_ab40_rate, baseline_ab42_rate = optimal_params
    
    print(f"\n" + "="*60)
    print(f"ANALYZING MULTI-DATASET OPTIMIZED PARAMETERS")
    print(f"="*60)
    
    # Calculate rates with optimized parameters
    rates = calculate_k_rates(
        original_kb0_forty=kb0_forty_h / 3600,
        original_kb1_forty=kb1_forty_h / 3600,
        original_kb0_fortytwo=kb0_fortytwo_h / 3600,
        original_kb1_fortytwo=kb1_fortytwo_h / 3600,
        baseline_ab40_plaque_rate=baseline_ab40_rate,
        baseline_ab42_plaque_rate=baseline_ab42_rate
    )
    
    # Run simulation with optimized parameters
    rr.reset()
    setup_integrator()
    
    # Set optimized parameters
    rr.k_O2_M_forty = kb0_forty_h
    rr.k_O3_O2_forty = kb1_forty_h
    rr.k_O2_M_fortytwo = kb0_fortytwo_h
    rr.k_O3_O2_fortytwo = kb1_fortytwo_h
    
    
    for key, value in rates.items():
        try:
            setattr(rr, key, value)
        except Exception as e:
            pass
    
    # Run simulation (100 years)
    t_sim = 100*365*24
    sim_result = rr.simulate(0, t_sim, 400, selections=simulation_selections)
    
    # Extract simulation results
    sim_years = sim_result[:, 0] / 365 / 24
    sim_plasma_ab42_raw = sim_result[:, 1]
    sim_plasma_ab40_raw = sim_result[:, 2]
    sim_csf_ab42_raw = sim_result[:, 3]
    sim_isf_ab42_raw = sim_result[:, 4]
    
    # Apply volume scaling to get proper concentration units (as done in plot_simulation_results.py)
    volume_scale_factor_csf = 0.09875 # Division by volume for SAS concentration units
    volume_scale_factor_isf = 0.2505  # Division by volume for ISF concentration units
    sim_plasma_ab42 = sim_plasma_ab42_raw 
    sim_plasma_ab40 = sim_plasma_ab40_raw 
    sim_csf_ab42_nM = sim_csf_ab42_raw / volume_scale_factor_csf
    sim_isf_ab42_nM = sim_isf_ab42_raw / volume_scale_factor_isf
    
    # Calculate derived quantities
    sim_plasma_ratio = sim_plasma_ab42 / sim_plasma_ab40
    sim_csf_ab42_pg_ml = convert_nM_to_pg_ml(sim_csf_ab42_nM)
    sim_isf_ab42_pg_ml = convert_nM_to_pg_ml(sim_isf_ab42_nM)
    
    # Interpolate to experimental time points for error calculation
    f_plasma = interp1d(sim_years, sim_plasma_ratio, bounds_error=False, fill_value="extrapolate")
    f_csf = interp1d(sim_years, sim_csf_ab42_pg_ml, bounds_error=False, fill_value="extrapolate")
    f_isf = interp1d(sim_years, sim_isf_ab42_pg_ml, bounds_error=False, fill_value="extrapolate")
    
    sim_plasma_interp = f_plasma(plasma_years)
    sim_csf_interp = f_csf(csf_years)
    sim_isf_interp = f_isf(isf_years)
    
    # Calculate individual errors and metrics
    plasma_residuals = plasma_ratios - sim_plasma_interp
    csf_residuals = csf_concentrations - sim_csf_interp
    isf_residuals = isf_concentrations - sim_isf_interp
    
    plasma_error = np.sum(plasma_residuals**2)
    csf_error = np.sum(csf_residuals**2)
    isf_error = np.sum(isf_residuals**2)
    
    plasma_rms = np.sqrt(np.mean(plasma_residuals**2))
    csf_rms = np.sqrt(np.mean(csf_residuals**2))
    isf_rms = np.sqrt(np.mean(isf_residuals**2))
    
    # Print analysis
    print(f"Individual dataset errors:")
    print(f"  Plasma ratio - SSE: {plasma_error:.2e}, RMS: {plasma_rms:.4f}")
    print(f"  CSF Aβ42 - SSE: {csf_error:.2e}, RMS: {csf_rms:.2f} pg/ml")
    print(f"  ISF Aβ42 - SSE: {isf_error:.2e}, RMS: {isf_rms:.2f} pg/ml")
    
    # Create comprehensive results plot
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Plasma ratio fit
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(plasma_years, plasma_ratios, color='red', label='Experimental', s=80, zorder=3, alpha=0.8)
    plt.plot(sim_years, sim_plasma_ratio, 'b-', linewidth=3, label='Optimized model', alpha=0.8)
    plt.scatter(plasma_years, sim_plasma_interp, color='blue', marker='x', s=60, 
                label='Model at exp. points', zorder=3, linewidth=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Aβ42/Aβ40 Plasma Ratio', fontsize=12, fontweight='bold')
    plt.title('Plasma Ratio Fit', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
    # Plot 2: CSF concentration fit
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(csf_years, csf_concentrations, color='red', label='Experimental', s=80, zorder=3, alpha=0.8)
    plt.plot(sim_years, sim_csf_ab42_pg_ml, 'g-', linewidth=3, label='Optimized model', alpha=0.8)
    plt.scatter(csf_years, sim_csf_interp, color='green', marker='x', s=60,
                label='Model at exp. points', zorder=3, linewidth=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('CSF Aβ42 (pg/ml)', fontsize=12, fontweight='bold')
    plt.title('CSF Aβ42 Concentration Fit', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
    # Plot 3: ISF concentration fit
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(isf_years, isf_concentrations, color='red', label='Experimental', s=80, zorder=3, alpha=0.8)
    plt.plot(sim_years, sim_isf_ab42_pg_ml, 'm-', linewidth=3, label='Optimized model', alpha=0.8)
    plt.scatter(isf_years, sim_isf_interp, color='magenta', marker='x', s=60,
                label='Model at exp. points', zorder=3, linewidth=3)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('ISF Aβ42 Monomers (pg/ml)', fontsize=12, fontweight='bold')
    plt.title('ISF Aβ42 Monomer Concentration Fit', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
    # Plot 4: Loss history during optimization
    ax4 = plt.subplot(2, 3, 4)
    iterations = np.arange(1, len(loss_history) + 1)
    plt.semilogy(iterations, loss_history, 'k-', linewidth=2, label='Total Loss')
    if len(plasma_error_history) > 0:
        plt.semilogy(iterations, plasma_error_history, 'b-', linewidth=2, alpha=0.7, label='Plasma')
        plt.semilogy(iterations, csf_error_history, 'g-', linewidth=2, alpha=0.7, label='CSF')
        plt.semilogy(iterations, isf_error_history, 'm-', linewidth=2, alpha=0.7, label='ISF')
    plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (Sum of Squared Errors)', fontsize=12, fontweight='bold')
    plt.title('Multi-Dataset Optimization History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Residuals for all datasets
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(plasma_years, plasma_residuals, 'bo-', linewidth=2, markersize=4, label='Plasma residuals')
    plt.plot(csf_years, csf_residuals/1000, 'go-', linewidth=2, markersize=4, label='CSF residuals (/1000)')
    plt.plot(isf_years, isf_residuals/10000, 'mo-', linewidth=2, markersize=4, label='ISF residuals (/10000)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.xlabel('Time (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Residuals (scaled)', fontsize=12, fontweight='bold')
    plt.title('Model Residuals vs Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(20, 100)
    
    # Plot 6: Parameter evolution during optimization (subset)
    ax6 = plt.subplot(2, 3, 6)
    if len(param_history) > 0:
        param_array = np.array(param_history)
        iterations = np.arange(1, len(param_array) + 1)
        plt.plot(iterations, param_array[:, 0], 'r-', linewidth=2, label='kb0_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 2], 'b-', linewidth=2, label='kb0_42', alpha=0.7)
        plt.plot(iterations, param_array[:, 4], 'g-', linewidth=2, label='base_40', alpha=0.7)
        plt.plot(iterations, param_array[:, 5], 'orange', linewidth=2, label='base_42', alpha=0.7)
        plt.xlabel('Function Evaluations', fontsize=12, fontweight='bold')
        plt.ylabel('Parameter Values', fontsize=12, fontweight='bold')
        plt.title('Key Parameter Evolution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.suptitle('Multi-Dataset Parameter Optimization Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig('simulation_plots/multi_dataset_experimental_fit_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'sim_years': sim_years,
        'plasma_ratio': sim_plasma_ratio, 'plasma_interp': sim_plasma_interp,
        'csf_pg_ml': sim_csf_ab42_pg_ml, 'csf_interp': sim_csf_interp,
        'isf_pg_ml': sim_isf_ab42_pg_ml, 'isf_interp': sim_isf_interp,
        'errors': {'plasma': plasma_error, 'csf': csf_error, 'isf': isf_error},
        'rms': {'plasma': plasma_rms, 'csf': csf_rms, 'isf': isf_rms}
    }

def calculate_percent_change_table(optimal_params):
    """Calculate and display percent changes in optimized parameters"""
    import pandas as pd
    
    kb0_forty_h, kb1_forty_h, kb0_fortytwo_h, kb1_fortytwo_h, \
    baseline_ab40_rate, baseline_ab42_rate = optimal_params
    
    # Initial parameter values
    initial_kb0_forty_h = 2.7e-3
    initial_kb1_forty_h = 0.00001 
    initial_kb0_fortytwo_h = 12.7e-3
    initial_kb1_fortytwo_h = 1.08 
    initial_baseline_ab40_rate = 0.000005
    initial_baseline_ab42_rate = 0.00005

    def calculate_percent_change(initial, final):
        return ((final - initial) / initial) * 100

    data = {
        'Parameter': [
            'kb0_40 (h⁻¹)', 'kb1_40 (h⁻¹)', 'kb0_42 (h⁻¹)', 'kb1_42 (h⁻¹)',
            'base_40 (L/(nM·h))', 'base_42 (L/(nM·h))'
        ],
        'Initial Value': [
            f"{initial_kb0_forty_h:.4e}", f"{initial_kb1_forty_h:.4e}",
            f"{initial_kb0_fortytwo_h:.4e}", f"{initial_kb1_fortytwo_h:.4e}",
            f"{initial_baseline_ab40_rate:.4e}", f"{initial_baseline_ab42_rate:.4e}"
        ],
        'Final Value': [
            f"{kb0_forty_h:.4e}", f"{kb1_forty_h:.4e}",
            f"{kb0_fortytwo_h:.4e}", f"{kb1_fortytwo_h:.4e}",
            f"{baseline_ab40_rate:.4e}", f"{baseline_ab42_rate:.4e}"
        ],
        'Percent Change (%)': [
            f"{calculate_percent_change(initial_kb0_forty_h, kb0_forty_h):.1f}",
            f"{calculate_percent_change(initial_kb1_forty_h, kb1_forty_h):.1f}",
            f"{calculate_percent_change(initial_kb0_fortytwo_h, kb0_fortytwo_h):.1f}",
            f"{calculate_percent_change(initial_kb1_fortytwo_h, kb1_fortytwo_h):.1f}",
            f"{calculate_percent_change(initial_baseline_ab40_rate, baseline_ab40_rate):.1f}",
            f"{calculate_percent_change(initial_baseline_ab42_rate, baseline_ab42_rate):.1f}"
        ]
    }

    df = pd.DataFrame(data)

    print(f"\n" + "="*80)
    print("MULTI-DATASET PARAMETER OPTIMIZATION: PERCENT CHANGES")
    print("="*80)
    print(df.to_string(index=False))

    # Summary statistics
    percent_changes = [
        calculate_percent_change(initial_kb0_forty_h, kb0_forty_h),
        calculate_percent_change(initial_kb1_forty_h, kb1_forty_h),
        calculate_percent_change(initial_kb0_fortytwo_h, kb0_fortytwo_h),
        calculate_percent_change(initial_kb1_fortytwo_h, kb1_fortytwo_h),
        calculate_percent_change(initial_baseline_ab40_rate, baseline_ab40_rate),
        calculate_percent_change(initial_baseline_ab42_rate, baseline_ab42_rate)
    ]

    print("\n\nSUMMARY:")
    print("="*40)
    largest_increase = max([pc for pc in percent_changes if pc > 0]) if any(pc > 0 for pc in percent_changes) else 0
    largest_decrease = min([pc for pc in percent_changes if pc < 0]) if any(pc < 0 for pc in percent_changes) else 0
    print(f"Largest increase: {largest_increase:.1f}%")
    print(f"Largest decrease: {largest_decrease:.1f}%")
    print(f"Average absolute change: {np.mean([abs(pc) for pc in percent_changes]):.1f}%")

if __name__ == "__main__":
    # Set weights for different datasets (can be adjusted based on importance/reliability)
    dataset_weights = [1.0, 1.0, 1.0]  # Equal weighting for plasma, CSF, ISF
    
    # Run multi-objective optimization
    optimal_params, final_error = optimize_parameters(weights=dataset_weights)
    
    # Analyze results
    analysis_results = analyze_multi_dataset_fit(optimal_params, weights=dataset_weights)
    
    print(f"\n" + "="*80)
    print(f"MULTI-DATASET OPTIMIZATION COMPLETE")
    print(f"="*80)
    print(f"Results saved to: simulation_plots/multi_dataset_experimental_fit_results.png")
    print(f"Dataset weights used - Plasma: {dataset_weights[0]}, CSF: {dataset_weights[1]}, ISF: {dataset_weights[2]}")
    print(f"Final weighted error: {final_error:.4e}")
    print(f"Individual dataset RMS errors:")
    print(f"  Plasma ratio: {analysis_results['rms']['plasma']:.4f}")
    print(f"  CSF Aβ42: {analysis_results['rms']['csf']:.2f} pg/ml")
    print(f"  ISF Aβ42: {analysis_results['rms']['isf']:.2f} pg/ml")
    
    # Calculate and display percent changes
    calculate_percent_change_table(optimal_params) 