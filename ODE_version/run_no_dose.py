import os
os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import numpy as np
from generated_model_2a import array_ode_wrapper, state_array_to_dict, get_state_names
from parameter_loader import load_parameters
import pandas as pd
import sys
import time
import argparse
from pathlib import Path
import traceback
import re

# Add parent directory to path to import K_rates_extrapolate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from K_rates_extrapolate import calculate_k_rates

def calculate_suvr(states, state_names, c1=2.52, c2=400000, c3=1.3, c4=3.5):
    """
    Calculate SUVR using the weighted sum formula:
    SUVR_w = 1 + (C₁(Ab42ᵒˡⁱᵍᵒ + Ab42ᵖʳᵒᵗᵒ + C₂*Ab42ᵖˡᵃᑫᵘᵉ))^C₃ / [(Ab42ᵒˡⁱᵍᵒ + Ab42ᵖʳᵒᵗᵒ + C₂*Ab42ᵖˡᵃᑫᵘᵉ)^C₃ + C₄^C₃]
    
    Args:
        states: List of state dictionaries
        state_names: List of all state names
        c1, c2, c3, c4: Parameters from PK_Geerts.csv
        
    Returns:
        SUVR array
    """
    # Initialize arrays
    n_timepoints = len(states)
    suvr = np.zeros(n_timepoints)
    
    # Get AB42 oligomer pattern
    ab42_oligomer_pattern = re.compile(r'AB42_Oligomer\d+$')
    
    # Get AB42 protofibril pattern (fibrils 17-23)
    ab42_protofibril_pattern = re.compile(r'AB42_Fibril(1[7-9]|2[0-3])$')
    
    for t in range(n_timepoints):
        state = states[t]
        
        # Calculate AB42 oligomer sum (weighted by size)
        ab42_oligo = 0.0
        for name in state_names:
            if ab42_oligomer_pattern.match(name):
                # Extract oligomer size from name - handling zero-padded numbers
                size_str = name.split('Oligomer')[1]
                size = int(size_str)
                # Weight by size
                ab42_oligo += size * state.get(name, 0.0)
        
        # Calculate AB42 protofibril sum (fibrils 17-23)
        ab42_proto = 0.0
        for name in state_names:
            if ab42_protofibril_pattern.match(name):
                # Extract fibril size
                size = int(name.split('Fibril')[1])
                ab42_proto += size * state.get(name, 0.0)
        
        # Get AB42 plaque
        ab42_plaque = state.get('AB42_Plaque_unbound', 0.0)
        
        # Calculate the weighted sum
        weighted_sum = ab42_oligo + ab42_proto + c2 * ab42_plaque
        
        # Calculate the numerator and denominator for SUVR
        numerator = c1 * (weighted_sum ** c3)
        denominator = (weighted_sum ** c3) + (c4 ** c3)
        
        # Calculate SUVR
        if denominator > 0:
            suvr[t] = 1.0 + (numerator / denominator)
        else:
            suvr[t] = 1.0  # Default value if denominator is zero
    
    return suvr

def run_simulation(antibody_type='Gant', simulation_years=10, save_dir='results/no_dose'):
    """
    Run the no-dose simulation for the specified number of years
    
    Args:
        antibody_type: 'Gant' for Gantenerumab or 'Lec' for Lecanemab (affects parameters)
        simulation_years: Number of years to simulate
        save_dir: Directory to save results
    """
    # Set up JAX
    jax.config.update("jax_enable_x64", True)
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load parameters
    params = load_parameters(antibody_type)
    
    # Get state names
    state_names = get_state_names()
    
    # Convert years to hours for simulation
    simulation_hours = simulation_years * 365 * 24
    
    # Print simulation info
    antibody_name = "Gantenerumab" if antibody_type == 'Gant' else "Lecanemab"
    print(f"\nSimulating {antibody_name} model for {simulation_years} years (no drug dosing)...")
    
    # Set up initial conditions (all zeros)
    y0 = jnp.zeros(len(state_names))
    
    # Get indices for specific variables
    fcrn_free_bbb_idx = state_names.index('FcRn_free_BBB')
    fcrn_free_bcsfb_idx = state_names.index('FcRn_free_BcsfB')
    microglia_cell_count_idx = state_names.index('Microglia_cell_count')
    cl_ab40_ide_idx = state_names.index('CL_AB40_IDE')
    cl_ab42_ide_idx = state_names.index('CL_AB42_IDE')
    
    # Set initial values
    y0 = y0.at[fcrn_free_bbb_idx].set(4.982e04)  # Fcrn_free_BBB
    y0 = y0.at[fcrn_free_bcsfb_idx].set(4.982e04)  # Fcrn_free_BcsfB
    y0 = y0.at[microglia_cell_count_idx].set(1.0)  # Microglia_Cell_Count
    # Can Increase clearance to values from Geerts 2024 publication 
    y0 = y0.at[cl_ab40_ide_idx].set(1500)  # CL_AB40_IDE
    y0 = y0.at[cl_ab42_ide_idx].set(50)  # CL_AB42_IDE
    
    # Set up time points for simulation
    t0 = 0.0
    t1 = float(simulation_hours)
    dt = 0.001  # Initial time step (changed from 0.0002 to match combined model)
    n_steps = min(1000, int(simulation_years * 12))  # Save approximately monthly points
    # Create saveat points
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))
    
    # Create the ODE solver
    term = diffrax.ODETerm(array_ode_wrapper)
    solver = diffrax.Tsit5()  # Using Tsit5 instead of Dopri5 to match combined model
    rtol = 1e-6  # Changed from 1e-10 to match combined model
    atol = 1e-9  # Changed from 1e-10 to match combined model
    
    # Create stepsize controller
    controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff=0.4, icoeff=0.3)
    
    print(f"\nRunning simulation for {simulation_years} years ({simulation_hours} hours)...")
    start_time = time.time()
    # Run the simulation
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=params,
        saveat=saveat,
        max_steps=10000000000,
        stepsize_controller=controller,
        progress_meter=diffrax.TextProgressMeter()
    )
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Return the solution and save the results
    return create_plots(solution, antibody_type, simulation_years, save_path, state_names)

def plot_biomarkers_and_ratios(results, years, states, save_path, antibody_type, simulation_years):
    """
    Create plots showing CSF biomarkers, brain monomer concentrations, and plasma ratios
    
    Args:
        results: Dictionary containing simulation results
        years: Array of time points in years
        states: List of state dictionaries at each time point
        save_path: Path to save the plots
        antibody_type: Type of antibody used in the simulation
        simulation_years: Number of years simulated
    """
    antibody_name = "Gantenerumab" if antibody_type == 'Gant' else "Lecanemab"
    base_filename = f"{antibody_name}_no_dose_{simulation_years}yr"
    
    # Convert from nM to pg/ml
    # AB40 MolWt ~4330 g/mol, AB42 MolWt ~4514 g/mol
    AB40_MOLWT = 4330  # g/mol
    AB42_MOLWT = 4514  # g/mol
    
    # Extract time series for species of interest
    ab40_csf_sas = jnp.array([state.get('AB40Mu_CSF_SAS', 0) for state in states])
    ab42_csf_sas = jnp.array([state.get('AB42Mu_CSF_SAS', 0) for state in states])
    ab40_monomer = results['AB40_Monomer']
    ab42_monomer = results['AB42_Monomer']
    ab40_brain_plasma = jnp.array([state.get('AB40Mu_Brain_Plasma', 0) for state in states])
    ab42_brain_plasma = jnp.array([state.get('AB42Mu_Brain_Plasma', 0) for state in states])
    
    # Calculate concentrations in pg/ml
    ab40_csf_sas_pg_ml = ab40_csf_sas * AB40_MOLWT
    ab42_csf_sas_pg_ml = ab42_csf_sas * AB42_MOLWT
    ab40_monomer_pg_ml = ab40_monomer * AB40_MOLWT
    ab42_monomer_pg_ml = ab42_monomer * AB42_MOLWT
    
    # Calculate ratios
    brain_plasma_ratio = jnp.where(ab40_brain_plasma > 0, 
                                  ab42_brain_plasma / ab40_brain_plasma, 
                                  jnp.zeros_like(ab40_brain_plasma))
    csf_ratio = jnp.where(ab40_csf_sas > 0,
                         ab42_csf_sas / ab40_csf_sas,
                         jnp.zeros_like(ab40_csf_sas))
    
    # Create plots
    
    # 1. CSF concentrations
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(years, ab40_csf_sas_pg_ml, label='AB40 CSF SAS', linewidth=2, color='blue')
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('Concentration (pg/ml)', fontsize=12)
    ax1.set_title('AB40 in CSF SAS', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(years, ab42_csf_sas_pg_ml, label='AB42 CSF SAS', linewidth=2, color='red')
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Concentration (pg/ml)', fontsize=12)
    ax2.set_title('AB42 in CSF SAS', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    fig1.savefig(save_path / f"{base_filename}_csf_concentrations.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Brain monomer concentrations
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(years, ab40_monomer_pg_ml, label='AB40 Monomer', linewidth=2, color='blue')
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('Concentration (pg/ml)', fontsize=12)
    ax1.set_title('AB40 Monomer in ISF', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(years, ab42_monomer_pg_ml, label='AB42 Monomer', linewidth=2, color='red')
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Concentration (pg/ml)', fontsize=12)
    ax2.set_title('AB42 Monomer in ISF', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    fig2.savefig(save_path / f"{base_filename}_brain_monomer_concentrations.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Ratio plots
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(years, brain_plasma_ratio, label='AB42/AB40 in Brain Plasma', linewidth=2, color='purple')
    ax1.set_xlabel('Time (years)', fontsize=12)
    ax1.set_ylabel('Ratio', fontsize=12)
    ax1.set_title('AB42/AB40 Ratio in Brain Plasma', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.plot(years, csf_ratio, label='AB42/AB40 in CSF', linewidth=2, color='green')
    ax2.set_xlabel('Time (years)', fontsize=12)
    ax2.set_ylabel('Ratio', fontsize=12)
    ax2.set_title('AB42/AB40 Ratio in CSF SAS', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    fig3.savefig(save_path / f"{base_filename}_ab42_ab40_ratios_detailed.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

def create_plots(solution, antibody_type, simulation_years, save_path, state_names):
    """
    Create plots similar to those in the combined master model
    """
    if solution is None:
        print("No solution data available for plotting.")
        return None, None
    
    # Set global matplotlib parameters for better readability (from compare_no_dose_models.py)
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 3,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.minor.size': 5,
        'ytick.minor.size': 5,
    })
    
    # Convert time from hours to years for better readability
    years = solution.ts / (24.0 * 365.0)
    
    # Create a dictionary to store results
    results = {'time_years': years}
    
    # Convert state array to dictionaries for each time point
    states = []
    for t_idx in range(len(solution.ts)):
        state_dict = state_array_to_dict(solution.ys[t_idx], state_names)
        states.append(state_dict)
    
    # Extract main state variables
    results['AB40_Monomer'] = jnp.array([state['AB40_Monomer'] for state in states])
    results['AB42_Monomer'] = jnp.array([state['AB42_Monomer'] for state in states])
    results['AB40_Plaque_unbound'] = jnp.array([state['AB40_Plaque_unbound'] for state in states])
    results['AB42_Plaque_unbound'] = jnp.array([state['AB42_Plaque_unbound'] for state in states])
    results['PK_central_compartment'] = jnp.array([state['PK_central_compartment'] for state in states])
    results['PK_Brain_Plasma'] = jnp.array([state['PK_Brain_Plasma'] for state in states])
    results['Antibody_unbound_ISF'] = jnp.array([state['Antibody_unbound_ISF'] for state in states])
    results['Microglia_Hi_Fract'] = jnp.array([state['Microglia_Hi_Fract'] for state in states])
    # Add CL_AB40_IDE and CL_AB42_IDE
    results['CL_AB40_IDE'] = jnp.array([state['CL_AB40_IDE'] for state in states])
    results['CL_AB42_IDE'] = jnp.array([state['CL_AB42_IDE'] for state in states])
    
    # Calculate SUVR using the weighted sum formula instead of using the stored value
    results['SUVR_calculated'] = calculate_suvr(states, state_names)
    
    # Keep the original SUVR from the model for comparison
    results['SUVR'] = jnp.array([state['SUVR'] for state in states])
    
    # Identify and sum oligomers
    ab40_oligomer_names = [name for name in state_names if re.match(r'AB40_Oligomer\d+$', name)]
    ab42_oligomer_names = [name for name in state_names if re.match(r'AB42_Oligomer\d+$', name)]
    
    # Identify and sum fibrils
    ab40_fibril_names = [name for name in state_names if re.match(r'AB40_Fibril\d+$', name)]
    ab42_fibril_names = [name for name in state_names if re.match(r'AB42_Fibril\d+$', name)]
    
    # Calculate oligomer loads
    ab40_oligomers = jnp.zeros(len(solution.ts))
    for name in ab40_oligomer_names:
        values = jnp.array([state[name] for state in states])
        ab40_oligomers += values
    
    ab42_oligomers = jnp.zeros(len(solution.ts))
    for name in ab42_oligomer_names:
        values = jnp.array([state[name] for state in states])
        ab42_oligomers += values
    
    # Calculate fibril loads
    ab40_fibrils = jnp.zeros(len(solution.ts))
    for name in ab40_fibril_names:
        values = jnp.array([state[name] for state in states])
        ab40_fibrils += values
    
    ab42_fibrils = jnp.zeros(len(solution.ts))
    for name in ab42_fibril_names:
        values = jnp.array([state[name] for state in states])
        ab42_fibrils += values
    
    # Add to results
    results['AB40_Oligomers'] = ab40_oligomers
    results['AB42_Oligomers'] = ab42_oligomers
    results['AB40_Fibrils'] = ab40_fibrils
    results['AB42_Fibrils'] = ab42_fibrils
    
    # Add biomarker and ratio plots
    plot_biomarkers_and_ratios(results, years, states, save_path, antibody_type, simulation_years)
    
    # Save results to CSV (all state variables)
    antibody_name = "Gantenerumab" if antibody_type == 'Gant' else "Lecanemab"
    base_filename = f"{antibody_name}_no_dose_{simulation_years}yr"
    # Build a DataFrame with all state variables at each time point
    all_states_df = pd.DataFrame(states)
    all_states_df.insert(0, 'time_years', years)
    all_states_df.to_csv(save_path / f"{base_filename}_results.csv", index=False)
    
    try:
        # 1. AB42/AB40 Ratios Plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        # Calculate ratios
        monomer_ratio = results['AB42_Monomer'] / results['AB40_Monomer']
        oligomer_ratio = results['AB42_Oligomers'] / results['AB40_Oligomers']
        plaque_ratio = results['AB42_Plaque_unbound'] / results['AB40_Plaque_unbound']
        fibril_ratio = results['AB42_Fibrils'] / results['AB40_Fibrils']
        # Plot ratios
        ax1.plot(years, monomer_ratio, label='Monomer Ratio', linewidth=3)
        ax1.plot(years, oligomer_ratio, label='Oligomer Ratio', linewidth=3)
        ax1.plot(years, plaque_ratio, label='Plaque Ratio', linewidth=3)
        ax1.plot(years, fibril_ratio, label='Fibril Ratio', linewidth=3)
        ax1.set_xlabel('Time (years)', fontsize=20)
        ax1.set_ylabel('AB42/AB40 Ratio', fontsize=20)
        ax1.set_title('AB42/AB40 Ratios Over Time', fontsize=22)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        fig1.savefig(save_path / f"{base_filename}_ab42_ab40_ratios.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        # 2. Oligomer and Monomer Loads
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 8))
        ax2a.plot(years, results['AB40_Monomer'], label='AB40 Monomer', linewidth=3, color='C0')
        ax2a.plot(years, results['AB40_Oligomers'], label='AB40 Oligomers', linewidth=3, color='C1')
        ax2a.set_xlabel('Time (years)', fontsize=20)
        ax2a.set_ylabel('Concentration (nM)', fontsize=20)
        ax2a.set_title('AB40 Loads', fontsize=22)
        ax2a.legend()
        ax2a.grid(True, alpha=0.3)
        ax2b.plot(years, results['AB42_Monomer'], label='AB42 Monomer', linewidth=3, color='C0')
        ax2b.plot(years, results['AB42_Oligomers'], label='AB42 Oligomers', linewidth=3, color='C1')
        ax2b.set_xlabel('Time (years)', fontsize=20)
        ax2b.set_ylabel('Concentration (nM)', fontsize=20)
        ax2b.set_title('AB42 Loads', fontsize=22)
        ax2b.legend()
        ax2b.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(save_path / f"{base_filename}_oligomer_monomer_loads.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        # 3. Plaque and Fibril Dynamics
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 8))
        ax3a.plot(years, results['AB40_Plaque_unbound'], label='AB40 Plaque', linewidth=3, color='C2')
        ax3a.plot(years, results['AB40_Fibrils'], label='AB40 Fibrils', linewidth=3, color='C3')
        ax3a.set_xlabel('Time (years)', fontsize=20)
        ax3a.set_ylabel('Concentration (nM)', fontsize=20)
        ax3a.set_title('AB40 Plaque and Fibril Dynamics', fontsize=22)
        ax3a.legend()
        ax3a.grid(True, alpha=0.3)
        ax3b.plot(years, results['AB42_Plaque_unbound'], label='AB42 Plaque', linewidth=3, color='C2')
        ax3b.plot(years, results['AB42_Fibrils'], label='AB42 Fibrils', linewidth=3, color='C3')
        ax3b.set_xlabel('Time (years)', fontsize=20)
        ax3b.set_ylabel('Concentration (nM)', fontsize=20)
        ax3b.set_title('AB42 Plaque and Fibril Dynamics', fontsize=22)
        ax3b.legend()
        ax3b.grid(True, alpha=0.3)
        plt.tight_layout()
        fig3.savefig(save_path / f"{base_filename}_plaque_fibril_dynamics.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        # 4. SUVR Plot
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        # Plot both the calculated SUVR and the model's SUVR
        ax4.plot(years, results['SUVR_calculated'], label='SUVR (Calculated)', linewidth=3, color='blue')
        ax4.plot(years, results['SUVR'], label='SUVR (Model)', linewidth=3, color='red', linestyle='--')
        ax4.set_xlabel('Time (years)', fontsize=20)
        ax4.set_ylabel('SUVR', fontsize=20)
        ax4.set_title('SUVR Progression Over Time', fontsize=22)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        fig4.savefig(save_path / f"{base_filename}_suvr.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
        # 5. Microglia Activation
        fig5, ax5 = plt.subplots(figsize=(12, 8))
        ax5.plot(years, results['Microglia_Hi_Fract'], label='Activated Microglia Fraction', linewidth=3)
        ax5.set_xlabel('Time (years)', fontsize=20)
        ax5.set_ylabel('Fraction', fontsize=20)
        ax5.set_title('Microglia Activation Over Time', fontsize=22)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        fig5.savefig(save_path / f"{base_filename}_microglia.png", dpi=300, bbox_inches='tight')
        plt.close(fig5)
        # 6. Total Oligomer and Fibril Composition
        fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(16, 8))
        ax6a.stackplot(years, 
                      results['AB40_Monomer'],
                      results['AB40_Oligomers'],
                      results['AB40_Fibrils'],
                      results['AB40_Plaque_unbound'],
                      labels=['Monomer', 'Oligomers', 'Fibrils', 'Plaque'],
                      alpha=0.7)
        ax6a.set_xlabel('Time (years)', fontsize=20)
        ax6a.set_ylabel('Concentration (nM)', fontsize=20)
        ax6a.set_title('AB40 Species Composition', fontsize=22)
        ax6a.legend()
        ax6a.grid(True, alpha=0.3)
        ax6b.stackplot(years, 
                      results['AB42_Monomer'],
                      results['AB42_Oligomers'],
                      results['AB42_Fibrils'],
                      results['AB42_Plaque_unbound'],
                      labels=['Monomer', 'Oligomers', 'Fibrils', 'Plaque'],
                      alpha=0.7)
        ax6b.set_xlabel('Time (years)', fontsize=20)
        ax6b.set_ylabel('Concentration (nM)', fontsize=20)
        ax6b.set_title('AB42 Species Composition', fontsize=22)
        ax6b.legend()
        ax6b.grid(True, alpha=0.3)
        plt.tight_layout()
        fig6.savefig(save_path / f"{base_filename}_species_composition.png", dpi=300, bbox_inches='tight')
        plt.close(fig6)
        # 7. CL IDE Dynamics
        '''
        fig7, ax7 = plt.subplots(figsize=(12, 8))
        ax7.plot(years, results['CL_AB40_IDE'], label='CL_AB40_IDE', linewidth=3, color='blue')
        ax7.plot(years, results['CL_AB42_IDE'], label='CL_AB42_IDE', linewidth=3, color='red')
        ax7.set_xlabel('Time (years)', fontsize=20)
        ax7.set_ylabel('CL IDE Value', fontsize=20)
        ax7.set_title('CL_AB40_IDE and CL_AB42_IDE Dynamics Over Time', fontsize=22)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        fig7.savefig(save_path / f"{base_filename}_CL_IDE_dynamics.png", dpi=300, bbox_inches='tight')
        plt.close(fig7)
        '''

        print(f"Plots saved to {save_path}")
    except Exception as e:
        print(f"\nError creating plots: {e}")
        traceback.print_exc()
    return solution, results

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run QSP model without drug dosing for long-term analysis')
    parser.add_argument('--drug', type=str, choices=['lecanemab', 'gantenerumab'], default='gantenerumab',
                        help='Drug type parameters to use: lecanemab or gantenerumab')
    parser.add_argument('--years', type=float, default=10.0,
                        help='Number of years to simulate')
    parser.add_argument('--outdir', type=str, default='results/no_dose',
                        help='Directory to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Map drug argument to antibody_type for internal use
    antibody_type = 'Lec' if args.drug.lower() == 'lecanemab' else 'Gant'
    
    # Run simulation
    solution, results = run_simulation(
        antibody_type=antibody_type,
        simulation_years=args.years,
        save_dir=args.outdir
    ) 