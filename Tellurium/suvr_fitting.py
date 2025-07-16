import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import re
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from K_rates_extrapolate import calculate_k_rates

# Global variables for tracking optimization
loss_history = []
param_history = []

def load_experimental_data():
    """Load experimental SUVR data from Geerts_2023_Figure_3.csv"""
    data_path = os.path.join(parent_dir, 'Tellurium', 'Geerts_2023_Figure_3.csv')
    df = pd.read_csv(data_path)

    # Clean up column names by stripping quotes and spaces
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    # Clean up 'Condition' column values
    if 'Condition' in df.columns:
        df['Condition'] = df['Condition'].str.strip().str.replace('"', '')

    # Filter for SUVR and non-APOE4 condition
    suvr_data = df[(df['Measure'] == 'SUVR') & (df['Condition'] == 'non APOE4')]
    
    time_years = suvr_data['Time (years)'].values
    suvr_values = suvr_data['y'].values

    print(f"Loaded {len(time_years)} experimental SUVR data points")
    print(f"Time range: {time_years.min():.1f} to {time_years.max():.1f} years")
    print(f"SUVR range: {suvr_values.min():.3f} to {suvr_values.max():.3f}")
    
    return time_years, suvr_values

def calculate_suvr(sol, model, c1=2.52, c2=1.3, c3=3.5, c4=400000, volume_scale_factor_isf=0.2505):
    """
    Calculate SUVR using the weighted sum formula.
    Copied from visualize_tellurium_simulation.py
    """
    y_indexes = model.y_indexes
    n_timepoints = len(sol.ts)
    suvr = np.zeros(n_timepoints)
    
    # Get AB42 oligomer pattern
    ab42_oligomer_pattern = re.compile(r'AB42_Oligomer\d+$')
    
    # Get AB42 protofibril pattern (fibrils 17-23)
    ab42_protofibril_pattern = re.compile(r'AB42_Fibril\d+$')
    
    for t in range(n_timepoints):
        # Calculate AB42 oligomer sum (weighted by size)
        ab42_oligo = 0.0
        for name, idx in y_indexes.items():
            if ab42_oligomer_pattern.match(name):
                # Extract oligomer size from name - handling zero-padded numbers
                size_str = name.split('Oligomer')[1]
                size = int(size_str)
                # Weight by size
                ab42_oligo += size * sol.ys[t, idx]
        
        # Calculate AB42 protofibril sum (fibrils 17-23)
        ab42_proto = 0.0
        for name, idx in y_indexes.items():
            if ab42_protofibril_pattern.match(name):
                # Extract fibril size
                size = int(name.split('Fibril')[1])
                ab42_proto += size * sol.ys[t, idx]
        
        # Get AB42 plaque
        ab42_plaque = sol.ys[t, y_indexes.get('AB42_Plaque_unbound', 0)]
        
        # Calculate the weighted sum
        weighted_sum = (ab42_oligo + ab42_proto + c2 * ab42_plaque) / volume_scale_factor_isf
        
        # Calculate the numerator and denominator for SUVR
        numerator = c1 * (weighted_sum ** c3)
        denominator = (weighted_sum ** c3) + (c4 ** c3)
        
        # Calculate SUVR
        if denominator > 0:
            suvr[t] = 1.0 + (numerator / denominator)
        else:
            suvr[t] = 1.0  # Default value if denominator is zero
    
    return suvr

def objective_function(log_params, exp_years, exp_suvr):
    """
    Objective function: sum of squared differences from experimental SUVR.
    
    Parameters:
    log_params: [log(kb0_fortytwo), log(kb1_fortytwo), log(k_F24_O12_fortytwo)]
    """
    global loss_history, param_history
    
    # Transform back from log space
    kb0_fortytwo = np.exp(log_params[0])  # h^-1
    kb1_fortytwo = np.exp(log_params[1])  # h^-1
    k_F24_O12_fortytwo = np.exp(log_params[2])  # h^-1
    
    try:
        # Load and setup model
        xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model.xml')
        with open(xml_path, "r") as f:
            sbml_str = f.read()
        
        rr = te.loadSBMLModel(sbml_str)
        rr.reset()
        rr.setIntegrator('cvode')
        rr.integrator.absolute_tolerance = 1e-6
        rr.integrator.relative_tolerance = 1e-6
        rr.integrator.setValue('stiff', True)
        
        # Calculate all rates using K_rates_extrapolate
        # Convert to 1/s for the calculate_k_rates function
        Garai_kb0 = kb0_fortytwo / 3600
        Garai_kb1 = kb1_fortytwo / 3600
        
        rates = calculate_k_rates(
            original_kb0_fortytwo=Garai_kb0,
            original_kb1_fortytwo=Garai_kb1
        )
        
        # Set the main parameters
        rr.k_O2_M_fortytwo = kb0_fortytwo
        rr.k_O3_O2_fortytwo = kb1_fortytwo
        rr.k_F24_O12_fortytwo = k_F24_O12_fortytwo
        
        # Update all related AB42 rates (excluding Plaque rates)
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    setattr(rr, key, value)
                except Exception as e:
                    pass  # Skip if parameter doesn't exist
        
        # Run simulation
        t_sim = 100 * 365 * 24  # 100 years in hours
        n_points = 200
        
        # Get all species for proper simulation
        selections = ['time'] + rr.getFloatingSpeciesIds()
        sim_result = rr.simulate(0, t_sim, n_points, selections=selections)
        
        # Check for simulation errors
        if sim_result is None or np.any(np.isnan(sim_result)) or np.any(np.isinf(sim_result)):
            return 1e10  # Large penalty for failed simulations
        
        # Create solution and model objects
        class Solution:
            def __init__(self, ts, ys):
                self.ts = ts
                self.ys = ys
        
        class Model:
            def __init__(self, species_names):
                self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
        
        # Extract time and species data
        sim_time = sim_result[:, 0]
        sim_species = sim_result[:, 1:]
        species_names = rr.getFloatingSpeciesIds()
        
        sol = Solution(sim_time, sim_species)
        model = Model(species_names)
        
        # Calculate SUVR from simulation
        sim_suvr = calculate_suvr(sol, model)
        sim_years = sol.ts / (24 * 365)  # Convert hours to years
        
        # Interpolate simulation results to match experimental time points
        f_sim = interp1d(sim_years, sim_suvr, bounds_error=False, fill_value="extrapolate")
        sim_suvr_interp = f_sim(exp_years)
        
        # Calculate sum of squared residuals
        residuals = exp_suvr - sim_suvr_interp
        error = np.sum(residuals**2)
        
        # Store history
        loss_history.append(error)
        param_history.append([kb0_fortytwo, kb1_fortytwo, k_F24_O12_fortytwo])
        
        # Print progress
        print(f"kb0={kb0_fortytwo:.2e}, kb1={kb1_fortytwo:.2e}, k_F24_O12={k_F24_O12_fortytwo:.2e}, "
              f"final_SUVR={sim_suvr[-1]:.3f}, exp_final={exp_suvr[-1]:.3f}, error={error:.2e}")
        
        return error
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1e10

def optimize_parameters(exp_years, exp_suvr):
    """Optimize parameters using scipy.optimize.minimize"""
    
    print("Starting parameter optimization...")
    
    # Initial parameter values (current/default values)
    # These are rough estimates - you may need to adjust based on your model
    initial_kb0_fortytwo = 45.72  # h^-1 (from agg_rate_sensitivity.py)
    initial_kb1_fortytwo = 1e-05   # h^-1 (from agg_rate_sensitivity.py)
    initial_k_F24_O12_fortytwo = 1.0  # h^-1 (estimate - you may need to adjust)
    
    print(f"Initial parameters:")
    print(f"  kb0_fortytwo = {initial_kb0_fortytwo:.2e} h^-1")
    print(f"  kb1_fortytwo = {initial_kb1_fortytwo:.2e} h^-1")
    print(f"  k_F24_O12_fortytwo = {initial_k_F24_O12_fortytwo:.2e} h^-1")
    
    # Starting point in log space
    initial_params_log = [
        np.log(initial_kb0_fortytwo),
        np.log(initial_kb1_fortytwo),
        np.log(initial_k_F24_O12_fortytwo)
    ]
    
    # Parameter bounds in log space (adjust as needed)
    bounds = [
        (np.log(1e-5), np.log(100.0)),    # kb0_fortytwo
        (np.log(1e-5), np.log(10.0)),     # kb1_fortytwo
        (np.log(1e-2), np.log(10.0)),     # k_F24_O12_fortytwo
    ]
    
    print(f"Parameter bounds:")
    print(f"  kb0_fortytwo: {np.exp(bounds[0][0]):.1e} to {np.exp(bounds[0][1]):.1e}")
    print(f"  kb1_fortytwo: {np.exp(bounds[1][0]):.1e} to {np.exp(bounds[1][1]):.1e}")
    print(f"  k_F24_O12_fortytwo: {np.exp(bounds[2][0]):.1e} to {np.exp(bounds[2][1]):.1e}")
    
    # Clear history
    global loss_history, param_history
    loss_history = []
    param_history = []
    
    # Run optimization
    result = minimize(
        objective_function,
        initial_params_log,
        args=(exp_years, exp_suvr),
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 100,
            'disp': True,
            'gtol': 1e-6,
            'ftol': 1e-9
        }
    )
    
    # Convert optimized parameters back to real space
    optimal_params = [np.exp(param) for param in result.x]
    
    print(f"\nOptimization complete!")
    print(f"Success: {result.success}")
    print(f"Final error: {result.fun:.2e}")
    print(f"Optimized parameters:")
    print(f"  kb0_fortytwo = {optimal_params[0]:.4e} h^-1")
    print(f"  kb1_fortytwo = {optimal_params[1]:.4e} h^-1")
    print(f"  k_F24_O12_fortytwo = {optimal_params[2]:.4e} h^-1")
    
    return optimal_params, result.fun

def run_simulation_with_params(kb0_fortytwo, kb1_fortytwo, k_F24_O12_fortytwo):
    """Run simulation with specified parameters"""
    
    xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model.xml')
    with open(xml_path, "r") as f:
        sbml_str = f.read()
    
    rr = te.loadSBMLModel(sbml_str)
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-6
    rr.integrator.relative_tolerance = 1e-6
    rr.integrator.setValue('stiff', True)
    
    # Calculate all rates using K_rates_extrapolate
    Garai_kb0 = kb0_fortytwo / 3600
    Garai_kb1 = kb1_fortytwo / 3600
    
    rates = calculate_k_rates(
        original_kb0_fortytwo=Garai_kb0,
        original_kb1_fortytwo=Garai_kb1
    )
    
    # Set the main parameters
    rr.k_O2_M_fortytwo = kb0_fortytwo
    rr.k_O3_O2_fortytwo = kb1_fortytwo
    rr.k_F24_O12_fortytwo = k_F24_O12_fortytwo
    
    # Update all related AB42 rates
    for key, value in rates.items():
        if '_fortytwo' in key:
            try:
                setattr(rr, key, value)
            except Exception as e:
                pass
    
    # Run simulation
    t_sim = 100 * 365 * 24  # 100 years in hours
    n_points = 300
    
    selections = ['time'] + rr.getFloatingSpeciesIds()
    sim_result = rr.simulate(0, t_sim, n_points, selections=selections)
    
    # Create solution and model objects
    class Solution:
        def __init__(self, ts, ys):
            self.ts = ts
            self.ys = ys
    
    class Model:
        def __init__(self, species_names):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
    
    sim_time = sim_result[:, 0]
    sim_species = sim_result[:, 1:]
    species_names = rr.getFloatingSpeciesIds()
    
    sol = Solution(sim_time, sim_species)
    model = Model(species_names)
    
    return sol, model

def plot_results(exp_years, exp_suvr, optimal_params, initial_params):
    """Plot experimental vs fitted SUVR results"""
    
    # Run simulation with initial parameters
    initial_sol, initial_model = run_simulation_with_params(*initial_params)
    initial_suvr = calculate_suvr(initial_sol, initial_model)
    initial_years = initial_sol.ts / (24 * 365)
    
    # Run simulation with optimized parameters
    optimal_sol, optimal_model = run_simulation_with_params(*optimal_params)
    optimal_suvr = calculate_suvr(optimal_sol, optimal_model)
    optimal_years = optimal_sol.ts / (24 * 365)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: SUVR comparison
    ax1.scatter(exp_years, exp_suvr, color='red', s=80, alpha=0.7, 
               label='Experimental SUVR', zorder=3)
    ax1.plot(initial_years, initial_suvr, 'g--', linewidth=2, alpha=0.8, 
            label='Initial parameters')
    ax1.plot(optimal_years, optimal_suvr, 'b-', linewidth=3, alpha=0.8, 
            label='Optimized parameters')
    
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('SUVR', fontsize=12)
    ax1.set_title('SUVR Fit Results', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(20, 100)
    
    # Plot 2: Optimization history
    if len(loss_history) > 0:
        ax2.semilogy(range(len(loss_history)), loss_history, 'r-', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Sum of Squared Residuals', fontsize=12)
        ax2.set_title('Optimization Progress', fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter evolution
    if len(param_history) > 0:
        param_array = np.array(param_history)
        ax3.plot(range(len(param_array)), param_array[:, 0], 'r-', label='kb0_fortytwo')
        ax3.plot(range(len(param_array)), param_array[:, 1], 'g-', label='kb1_fortytwo')
        ax3.plot(range(len(param_array)), param_array[:, 2], 'b-', label='k_F24_O12_fortytwo')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Parameter Value (h^-1)', fontsize=12)
        ax3.set_title('Parameter Evolution', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Plot 4: Residuals
    f_initial = interp1d(initial_years, initial_suvr, bounds_error=False, fill_value="extrapolate")
    f_optimal = interp1d(optimal_years, optimal_suvr, bounds_error=False, fill_value="extrapolate")
    
    initial_interp = f_initial(exp_years)
    optimal_interp = f_optimal(exp_years)
    
    initial_residuals = exp_suvr - initial_interp
    optimal_residuals = exp_suvr - optimal_interp
    
    ax4.plot(exp_years, initial_residuals, 'g--', linewidth=2, label='Initial residuals')
    ax4.plot(exp_years, optimal_residuals, 'b-', linewidth=2, label='Optimized residuals')
    ax4.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Time (years)', fontsize=12)
    ax4.set_ylabel('Residuals (exp - sim)', fontsize=12)
    ax4.set_title('Fit Residuals', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(parent_dir, 'Tellurium', 'simulation_plots', 'suvr_optimization_results.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {plot_path}")
    plt.show()
    
    # Print final statistics
    initial_rmse = np.sqrt(np.mean(initial_residuals**2))
    optimal_rmse = np.sqrt(np.mean(optimal_residuals**2))
    
    print(f"\nFit Statistics:")
    print(f"Initial RMSE: {initial_rmse:.4f}")
    print(f"Optimized RMSE: {optimal_rmse:.4f}")
    print(f"Improvement: {((initial_rmse - optimal_rmse)/initial_rmse)*100:.1f}%")

def main():
    """Main function"""
    print("=== SUVR Parameter Optimization ===")
    print("Optimizing: kb0_fortytwo, kb1_fortytwo, k_F24_O12_fortytwo")
    
    # Load experimental data
    exp_years, exp_suvr = load_experimental_data()
    
    # Run optimization
    optimal_params, final_error = optimize_parameters(exp_years, exp_suvr)
    
    # Initial parameters for comparison
    initial_params = [45.72, 1e-05, 1.0]  # h^-1
    
    # Plot results
    plot_results(exp_years, exp_suvr, optimal_params, initial_params)
    
    print("Optimization complete!")

if __name__ == "__main__":
    main() 