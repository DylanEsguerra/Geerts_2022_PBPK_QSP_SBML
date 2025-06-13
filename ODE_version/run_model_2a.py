import os
import jax
os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
from generated_model_2a import array_ode_wrapper, get_state_names
from parameter_loader import load_parameters
import pandas as pd
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path to import K_rates_extrapolate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def time_dependent_parameters(t, original_params, dose_nmol, antibody_type, dose_duration=1.0):
    """
    Update parameters based on time to implement time-dependent dosing using JAX-compatible control flow
    
    Args:
        t: Current time in hours
        original_params: Original parameter dictionary with case-insensitive lookup
        dose_nmol: Dose in nmol
        antibody_type: 'Gant' for Gantenerumab or 'Lec' for Lecanemab
        dose_duration: Duration of dose in hours (default: 1.0)
        
    Returns:
        Updated parameter dictionary
    """
    # Create modified copy of params that preserves the CaseInsensitiveParams structure
    params_dict = original_params.params.copy()
    case_map = original_params.case_map.copy()
    
    # Determine if we're in the dosing period (0 <= t < dose_duration)
    is_dosing_period = jnp.logical_and(t >= 0, t < dose_duration)
    is_gant = antibody_type == 'Gant'
    
    # For Gantenerumab (SC)
    input_sc_value = jnp.where(
        jnp.logical_and(is_dosing_period, is_gant),
        dose_nmol / dose_duration,  # If in dosing period and Gant
        0.0  # Otherwise
    )
    
    # For Lecanemab (IV)
    input_central_value = jnp.where(
        jnp.logical_and(is_dosing_period, jnp.logical_not(is_gant)),
        dose_nmol / dose_duration,  # If in dosing period and not Gant (i.e., Lec)
        0.0  # Otherwise
    )
    
    # Set the parameters in the dictionary
    params_dict['Input_SC'] = input_sc_value
    params_dict['Input_central'] = input_central_value
    
    # Create a new CaseInsensitiveParams object with the updated values
    class CaseInsensitiveParams:
        def __init__(self, params_dict, case_map):
            self.params = params_dict
            self.case_map = case_map
        
        def __getitem__(self, key):
            # First try direct access with the exact key
            if key in self.params:
                return self.params[key]
            
            # If not found, try case-insensitive lookup
            key_lower = key.lower()
            if key_lower in self.case_map:
                original_key, _ = self.case_map[key_lower]
                return self.params[original_key]
            
            # If trying to get a 'k_' parameter, try with 'K_' and vice versa
            if key.startswith('k_') or key.startswith('K_'):
                alternate_key = 'K' + key[1:] if key.startswith('k') else 'k' + key[1:]
                if alternate_key in self.params:
                    return self.params[alternate_key]
                
                # Try case-insensitive lookup for the alternate key
                alternate_key_lower = alternate_key.lower()
                if alternate_key_lower in self.case_map:
                    original_key, _ = self.case_map[alternate_key_lower]
                    return self.params[original_key]
            
            # If we reach here, the parameter is truly not found
            raise KeyError(f"Parameter '{key}' not found, even after case-insensitive lookup")
        
        def __contains__(self, key):
            if key in self.params:
                return True
            
            key_lower = key.lower()
            if key_lower in self.case_map:
                return True
                
            # Check for k_/K_ variants
            if key.startswith('k_') or key.startswith('K_'):
                alternate_key = 'K' + key[1:] if key.startswith('k') else 'k' + key[1:]
                if alternate_key in self.params:
                    return True
                
                alternate_key_lower = alternate_key.lower()
                if alternate_key_lower in self.case_map:
                    return True
            
            return False
    
    return CaseInsensitiveParams(params_dict, case_map)

def run_simulation(antibody_type='Gant', custom_dose=None, custom_time=None, save_dir='results'):
    """
    Run the simulation with the specified antibody and dosing parameters
    
    Args:
        antibody_type: 'Gant' for Gantenerumab (SC) or 'Lec' for Lecanemab (IV)
        custom_dose: Custom dose in nmol (overrides default doses)
        custom_time: Custom simulation time (overrides defaults)
        save_dir: Directory to save results
    """
    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load parameters
    params = load_parameters(antibody_type)
    
    # Initialize input parameters to zero (add to params.params directly)
    params.params['Input_SC'] = jnp.array(0.0)
    params.params['Input_central'] = jnp.array(0.0)
    
    # Fixed dose values as requested
    if custom_dose is not None:
        dose_nmol = custom_dose
    else:
        dose_nmol = 2050.6 if antibody_type == 'Gant' else 4930.6
    
    # Set fixed simulation times as requested
    if custom_time is not None:
        if isinstance(custom_time, int) and antibody_type == 'Gant':
            # If custom time is an integer and antibody is Gant, assume days
            t1 = custom_time * 24.0  # Convert days to hours
            days_for_display = custom_time
        else:
            # Otherwise use the value directly as hours
            t1 = float(custom_time)
            days_for_display = t1 / 24.0
    else:
        if antibody_type == 'Gant':
            t1 = 85 * 24.0  # 85 days in hours
            days_for_display = 85
        else:  # Lecanemab
            t1 = 700.0  # 700 hours
            days_for_display = 700 / 24.0
    
    # Get the state variable names
    state_names = get_state_names()
    
    # Set up initial conditions (all zeros)
    y0 = jnp.zeros(len(state_names))  # Based on the number of equations in the model
    
    # Get indices for specific variables
    fcrn_free_bbb_idx = state_names.index('FcRn_free_BBB')
    fcrn_free_bcsfb_idx = state_names.index('FcRn_free_BcsfB')
    
    # Set initial values
    y0 = y0.at[fcrn_free_bbb_idx].set(4.982e04)  # Fcrn_free_BBB
    y0 = y0.at[fcrn_free_bcsfb_idx].set(4.982e04)  # Fcrn_free_BcsfB
    
    # Set up time points for steady state simulation
    t0 = 0.0
    t_steady = 1000.0 #* 24.0  # Run for 1000 days to reach steady state
    dt = None  # Initial time step
    n_steps = 1000  # Number of points to save
    
    # Create saveat points for steady state
    saveat_steady = diffrax.SaveAt(ts=jnp.linspace(t0, t_steady, n_steps))
    
    # Create the ODE solver using the array_ode_wrapper
    term = diffrax.ODETerm(array_ode_wrapper)
    solver = diffrax.Dopri5()
    rtol = 1.4e-8
    atol = 1.4e-8
    
    # Create stepsize controller
    controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff=0.4, icoeff=0.3)
    
    print("\nRunning steady state simulation without drug...")
    start_time = time.time()
    
    # Run the steady state simulation
    steady_state_solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t_steady,
        dt0=dt,
        y0=y0,
        args=params,
        saveat=saveat_steady,
        max_steps=10000000000,
        stepsize_controller=controller,
        adjoint=diffrax.BacksolveAdjoint(),
        progress_meter=diffrax.TextProgressMeter()
    )
    
    end_time = time.time()
    print(f"Steady state simulation completed in {end_time - start_time:.2f} seconds")
    
    # Use the final state from steady state as initial condition for drug simulation
    y0_drug = steady_state_solution.ys[-1]
    
    # Create saveat points for drug simulation
    n_steps = min(1000, int(t1/24) + 1)  # Save at least daily points
    saveat_drug = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))
    
    # Create a time-dependent parameter function with the specified dose
    param_func = lambda t: time_dependent_parameters(t, params, dose_nmol, antibody_type)
    
    # Create a wrapper for array_ode_wrapper that uses time-dependent parameters
    def time_dependent_ode_wrapper(t, y, _):
        return array_ode_wrapper(t, y, param_func(t))
    
    # Print simulation info
    antibody_name = "Gantenerumab" if antibody_type == 'Gant' else "Lecanemab"
    dose_route = "SC" if antibody_type == 'Gant' else "IV"
    time_unit = "days" if antibody_type == 'Gant' else "hours"
    time_value = days_for_display if antibody_type == 'Gant' else t1
    
    print(f"\nSimulating {antibody_name} {dose_nmol} nmol {dose_route} for {time_value} {time_unit}...")
    
    start_time = time.time()
    
    # Create a new term with the time-dependent wrapper
    time_dep_term = diffrax.ODETerm(time_dependent_ode_wrapper)
    
    # Run the drug simulation with time-dependent parameters
    solution = diffrax.diffeqsolve(
        time_dep_term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0_drug,
        args=None,  # Parameters are now handled in the wrapper
        saveat=saveat_drug,
        max_steps=10000000000,
        stepsize_controller=controller,
        adjoint=diffrax.BacksolveAdjoint(),
        progress_meter=diffrax.TextProgressMeter()
    )
    
    end_time = time.time()
    print(f"Drug simulation completed in {end_time - start_time:.2f} seconds")
    
    # Return the solution and save the results
    return plot_and_save_results(solution, antibody_type, dose_nmol, antibody_type == 'Gant', t1, save_path, state_names)

def plot_and_save_results(solution, antibody_type, dose_nmol, subcutaneous, t1, save_path, state_names):
    """
    Plot and save the simulation results
    """
    # Convert time from hours to days for better readability if Gantenerumab, otherwise keep in hours
    if antibody_type == 'Gant':
        time_values = solution.ts / 24.0
        time_label = 'Time (days)'
    else:  # Lecanemab
        time_values = solution.ts
        time_label = 'Time (hours)'
    
    # Create a dictionary to store results
    results = {'time': time_values}
    
    # Dictionary of important state variables
    important_vars = [
        'AB40_Monomer',  
        'AB42_Monomer',  
        'AB40_monomer_ISF_total',  
        'AB42_monomer_ISF_total',  
        'AB40_Plaque_unbound',  
        'AB42_Plaque_unbound',  
        'PK_central_compartment',  
        'PK_Brain_Plasma',  
        'Antibody_ISF_total',  
        'Microglia_Hi_Fract',  
        'SUVR',  
    ]
    
    # Create a mapping of important variables to display names
    display_names = {
        'AB40_Monomer': 'AB40_Monomer',
        'AB42_Monomer': 'AB42_Monomer',
        'AB40_monomer_ISF_total': 'AB40_Monomer_ISF_total',
        'AB42_monomer_ISF_total': 'AB42_Monomer_ISF_total',
        'AB40_Plaque_unbound': 'AB40_Plaque_unbound',
        'AB42_Plaque_unbound': 'AB42_Plaque_unbound',
        'PK_central_compartment': 'PK_Central',
        'PK_Brain_Plasma': 'PK_Brain',
        'Antibody_ISF_total': 'Antibody_ISF',
        'Microglia_Hi_Fract': 'Microglia_Hi_Fract',
        'SUVR': 'SUVR',
    }
    
    # Extract and save all monitored state variables
    for var_name in important_vars:
        idx = state_names.index(var_name)
        results[display_names[var_name]] = solution.ys[:, idx]
    
    # Save results to CSV
    antibody_name = "Gantenerumab" if antibody_type == 'Gant' else "Lecanemab"
    dose_route = "SC" if subcutaneous else "IV"
    time_unit = "d" if antibody_type == 'Gant' else "h"
    time_value = int(t1 / 24.0) if antibody_type == 'Gant' else int(t1)
    
    base_filename = f"{antibody_name}_{dose_nmol}nmol_{dose_route}_{time_value}{time_unit}"
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path / f"{base_filename}_results.csv", index=False)
    
    # Create plots
    
    # 0. Brain Plasma Plot (similar to combined_master_model)
    fig_brain_plasma = plt.figure(figsize=(10, 6))
    plt.semilogy(time_values, results['PK_Brain'], color='black', linewidth=3, label='Brain Plasma')
    
    # Load experimental data if available
    data_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 f"parameters/Geerts_{'Lec' if antibody_type == 'Lec' else 'Gant'}_Data.csv"))
    if data_path.exists():
        exp_data = pd.read_csv(data_path)
        plt.scatter(exp_data['Time'], exp_data['Concentration'], 
                   color='red', s=100, marker='o', edgecolors='red', linewidths=2, 
                   facecolors='none', label='Experimental Data')
    
    # Set axis labels and title
    plt.xlabel(time_label, fontsize=14)
    plt.ylabel('Concentration (nM)', fontsize=14)
    dose_info = "10 mg/kg IV" if antibody_type == 'Lec' else f"{dose_nmol} nmol {dose_route}"
    plt.title(f'{antibody_name} Plasma PK: {dose_info}', fontsize=16, fontweight='bold')
    
    # Set y-axis limits based on drug
    if antibody_type == 'Lec':
        plt.ylim(1, 10000)  # 10^0 to 10^4
    else:
        plt.ylim(1, 1000)   # 10^0 to 10^3
    
    # Set x-axis limits
    if antibody_type == 'Lec':
        plt.xlim(0, 700)  # 700 hours for lecanemab
    else:
        plt.xlim(0, 85)   # 85 days for gantenerumab
    
    # Add grid and legend
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, framealpha=1, loc='upper right')
    plt.tight_layout()
    
    plt.savefig(save_path / f"{base_filename}_brain_plasma.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1. Plot antibody concentrations (central, brain, ISF)
    plt.figure(figsize=(10, 6))
    plt.semilogy(time_values, results['PK_Central'], label='Central Compartment', linewidth=2)
    plt.semilogy(time_values, results['PK_Brain'], label='Brain Plasma', linewidth=2)
    plt.semilogy(time_values, results['Antibody_ISF'], label='Brain ISF', linewidth=2)
    plt.xlabel(time_label)
    plt.ylabel('Antibody Concentration (nM)')
    plt.title(f'{antibody_name} {dose_nmol} nmol {dose_route} - Concentration Profile')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.savefig(save_path / f"{base_filename}_antibody_concentration.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot Aβ monomers
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, results['AB40_Monomer'], label='Aβ40 Monomer', linewidth=2)
    plt.plot(time_values, results['AB42_Monomer'], label='Aβ42 Monomer', linewidth=2)
    plt.xlabel(time_label)
    plt.ylabel('Concentration (nM)')
    plt.title(f'{antibody_name} {dose_nmol} nmol {dose_route} - Aβ Monomer Dynamics')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f"{base_filename}_monomer_concentrations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot plaque levels and SUVR
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, results['AB40_Plaque_unbound'], label='Aβ40 Plaque', linewidth=2)
    plt.plot(time_values, results['AB42_Plaque_unbound'], label='Aβ42 Plaque', linewidth=2)
    plt.plot(time_values, results['SUVR'], label='SUVR', linewidth=3, color='black')
    plt.xlabel(time_label)
    plt.ylabel('Level')
    plt.title(f'{antibody_name} {dose_nmol} nmol {dose_route} - Plaque and SUVR')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / f"{base_filename}_plaque_suvr.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot microglia activation
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, results['Microglia_Hi_Fract'], label='Activated Microglia Fraction', linewidth=2)
    plt.xlabel(time_label)
    plt.ylabel('Fraction')
    plt.title(f'{antibody_name} {dose_nmol} nmol {dose_route} - Microglia Activation')
    plt.grid(True)
    plt.savefig(save_path / f"{base_filename}_microglia.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {save_path}")
    return solution, results_df

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run PBPK-QSP model simulation')
    parser.add_argument('--antibody', type=str, choices=['Gant', 'Lec'], default='Gant',
                        help='Antibody type: Gant (Gantenerumab, SC) or Lec (Lecanemab, IV)')
    parser.add_argument('--dose', type=float, default=None,
                        help='Optional custom dose in nmol (overrides defaults)')
    parser.add_argument('--time', type=float, default=None,
                        help='Optional custom simulation time (days for Gant, hours for Lec)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run simulation
    run_simulation(
        antibody_type=args.antibody,
        custom_dose=args.dose,
        custom_time=args.time,
        save_dir=args.outdir
    ) 