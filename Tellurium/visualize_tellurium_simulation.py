"""
Script to visualize steady state data from run_default_simulation.py using plotting functions
adapted from visualize_steady_state.py.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
import traceback
import re

# Set global matplotlib parameters for better readability
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

script_dir = Path(__file__).resolve().parent

def calculate_suvr(sol, model, c1=2.52, c2=1.3, c3=3.5, c4=400000, volume_scale_factor_isf=0.2505):
    """
    Calculate SUVR using the weighted sum formula:
    SUVR_w = 1 + (C₁(Ab42ᵒˡⁱᵍᵒ + Ab42ᵖʳᵒᵗᵒ + C₂*Ab42ᵖˡ))^C₃ / [(Ab42ᵒˡⁱᵍᵒ + Ab42ᵖʳᵒᵗᵒ + C₂*Ab42ᵖˡ)^C₃ + C₄^C₃]
    
    Args:
        sol: Solution object with ts and ys attributes
        model: Model object with y_indexes attribute
        c1, c2, c3, c4: Parameters from PK_Geerts.csv
        
    Returns:
        SUVR array
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
                ab42_oligo += (size-1) * sol.ys[t, idx]
        
        # Calculate AB42 protofibril sum (fibrils 17-23)
        ab42_proto = 0.0
        for name, idx in y_indexes.items():
            if ab42_protofibril_pattern.match(name):
                # Extract fibril size
                size = int(name.split('Fibril')[1])
                ab42_proto += (size-1) * sol.ys[t, idx]
        
        # Get AB42 plaque
        ab42_plaque = sol.ys[t, y_indexes.get('AB42_Plaque_unbound', 0)]
        
        #print("Oligomer load: ", ab42_oligo/volume_scale_factor_isf)
        #print("Protofibril load: ", ab42_proto/volume_scale_factor_isf)
        #print("Plaque load: ", ab42_plaque/volume_scale_factor_isf)

        # Calculate the weighted sum
        weighted_sum = (ab42_oligo + ab42_proto + c2 * 24 * ab42_plaque)/volume_scale_factor_isf
        
        # Calculate the numerator and denominator for SUVR
        numerator = c1 * (weighted_sum ** c3)
        denominator = (weighted_sum ** c3) + (c4 ** c3)
        
        # Calculate SUVR
        if denominator > 0:
            suvr[t] = 1.0 + (numerator / denominator)
        else:
            suvr[t] = 1.0  # Default value if denominator is zero
    
    return suvr

def load_tellurium_data(years=100.0):
    """Load Tellurium simulation data from the saved CSV file.
    
    Args:
        years: Number of years simulated (used to select the correct CSV file).
        
    Returns:
        Tuple of (time_points, species_data, model).
    """
    data_path = script_dir / f"default_simulation_results_{years}yr_all_vars.csv"

    if not data_path.exists():
        print(f"Error: No data found at {data_path}")
        print(f"Please run run_default_simulation.py with --years {years} first")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    
    # Extract time points and species data
    time_points = df['time'].values
    data = df.drop('time', axis=1)
    species_data = data.values
    
    class SimpleModel:
        def __init__(self, species_names, initial_conditions):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
            self.y0 = initial_conditions
    
    model = SimpleModel(data.columns, species_data[0])
    
    return time_points, species_data, model

def create_solution_object(time_points, species_data):
    """Create a solution object compatible with the plotting functions.
    
    Args:
        time_points: Array of time points.
        species_data: Array of species concentrations.
        
    Returns:
        Solution object with ts and ys attributes.
    """
    class Solution:
        def __init__(self, ts, ys):
            self.ts = ts
            self.ys = ys
    
    return Solution(time_points, species_data)

def create_plots(sol, model):
    """Create plots of the simulation results."""
    if sol is None:
        print("No solution data available for plotting.")
        return
    
    figures_dir = script_dir / "simulation_plots/tellurium_steady_state"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    y_indexes = model.y_indexes
    
    # Volume scaling factor for ISF compartment
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    
    try:
        years = sol.ts / (24 * 365)
        
        # 1. AB42/AB40 Ratios Plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        # For ratios, volume scaling cancels out, so we can use raw values
        ab40_monomer = sol.ys[:, model.y_indexes['AB40_Monomer']]
        ab42_monomer = sol.ys[:, model.y_indexes['AB42_Monomer']]
        monomer_ratio = ab42_monomer / ab40_monomer
        
        ab40_oligomers = sum(sol.ys[:, model.y_indexes[f'AB40_Oligomer{i:02d}']] for i in range(2, 17) if f'AB40_Oligomer{i:02d}' in model.y_indexes)
        ab42_oligomers = sum(sol.ys[:, model.y_indexes[f'AB42_Oligomer{i:02d}']] for i in range(2, 17) if f'AB42_Oligomer{i:02d}' in model.y_indexes)
        oligomer_ratio = ab42_oligomers / ab40_oligomers
        
        ab40_fibrils = sum(sol.ys[:, model.y_indexes[f'AB40_Fibril{i:02d}']] for i in range(17, 24) if f'AB40_Fibril{i:02d}' in model.y_indexes)
        ab42_fibrils = sum(sol.ys[:, model.y_indexes[f'AB42_Fibril{i:02d}']] for i in range(17, 24) if f'AB42_Fibril{i:02d}' in model.y_indexes)
        fibril_ratio = ab42_fibrils / ab40_fibrils
        
        ab40_plaque = sol.ys[:, model.y_indexes['AB40_Plaque_unbound']]
        ab42_plaque = sol.ys[:, model.y_indexes['AB42_Plaque_unbound']]
        plaque_ratio = ab42_plaque / ab40_plaque
        
        ax1.plot(years, monomer_ratio, label='Monomer Ratio', linewidth=2)
        ax1.plot(years, oligomer_ratio, label='Oligomer Ratio', linewidth=2)
        ax1.plot(years, fibril_ratio, label='Fibril Ratio', linewidth=2)
        ax1.plot(years, plaque_ratio, label='Plaque Ratio', linewidth=2)
        
        ax1.set_xlabel('Time (years)', fontsize=12)
        ax1.set_ylabel('AB42/AB40 Ratio', fontsize=12)
        ax1.set_title('AB42/AB40 Ratios Over Time', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig1.savefig(figures_dir / 'ab42_ab40_ratios.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Oligomer Loads (apply volume scaling for proper concentration units)
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 8))
        
        
        
        # Apply volume scaling for proper concentration units
        ab40_oligomer_load = ab40_oligomers / volume_scale_factor_isf
        ab42_oligomer_load = ab42_oligomers / volume_scale_factor_isf
        
        ax2a.plot(years, ab40_oligomer_load, label='Oligomer Load', linewidth=2, color='C0')
        ax2a.set_xlabel('Time (years)', fontsize=12)
        ax2a.set_ylabel('Load (nM)', fontsize=12)
        ax2a.set_title('AB40 Oligomer Load', fontsize=14)
        ax2a.legend(fontsize=10)
        ax2a.grid(True, alpha=0.3)
        
        ax2b.plot(years, ab42_oligomer_load, label='Oligomer Load', linewidth=2, color='C0')
        ax2b.set_xlabel('Time (years)', fontsize=12)
        ax2b.set_ylabel('Load (nM)', fontsize=12)
        ax2b.set_title('AB42 Oligomer Load', fontsize=14)
        ax2b.legend(fontsize=10)
        ax2b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2.savefig(figures_dir / 'oligomer_load.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Protofibril Loads (apply volume scaling for proper concentration units)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Apply volume scaling for proper concentration units
        ab40_fibrils_scaled = ab40_fibrils / volume_scale_factor_isf
        ab42_fibrils_scaled = ab42_fibrils / volume_scale_factor_isf
        
        ax3a.plot(years, ab40_fibrils_scaled, label='Protofibril Load', linewidth=2, color='C1')
        ax3a.set_xlabel('Time (years)', fontsize=12)
        ax3a.set_ylabel('Load (nM)', fontsize=12)
        ax3a.set_title('AB40 Protofibril Load', fontsize=14)
        ax3a.legend(fontsize=10)
        ax3a.grid(True, alpha=0.3)
        
        ax3b.plot(years, ab42_fibrils_scaled, label='Protofibril Load', linewidth=2, color='C1')
        ax3b.set_xlabel('Time (years)', fontsize=12)
        ax3b.set_ylabel('Load (nM)', fontsize=12)
        ax3b.set_title('AB42 Protofibril Load', fontsize=14)
        ax3b.legend(fontsize=10)
        ax3b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig3.savefig(figures_dir / 'protofibril_load.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Plaque Dynamics (apply volume scaling for proper concentration units)
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Apply volume scaling for proper concentration units
        ab40_plaque_scaled = ab40_plaque / volume_scale_factor_isf
        ab42_plaque_scaled = ab42_plaque / volume_scale_factor_isf
        
        ax4a.plot(years, ab40_plaque_scaled, 
                 label='Unbound', linewidth=2, color='C2')
        if 'AB40_Plaque_Antibody_bound' in model.y_indexes:
            ab40_plaque_bound_raw = sol.ys[:, model.y_indexes['AB40_Plaque_Antibody_bound']]
            ab40_plaque_bound_scaled = ab40_plaque_bound_raw / volume_scale_factor_isf
            ax4a.plot(years, ab40_plaque_bound_scaled, 
                     label='Antibody-bound', linewidth=2, color='C3')
        ax4a.set_xlabel('Time (years)', fontsize=12)
        ax4a.set_ylabel('Concentration (nM)', fontsize=12)
        ax4a.set_title('AB40 Plaque Dynamics', fontsize=14)
        ax4a.legend(fontsize=10)
        ax4a.grid(True, alpha=0.3)
        
        ax4b.plot(years, ab42_plaque_scaled, 
                 label='Unbound', linewidth=2, color='C2')
        if 'AB42_Plaque_Antibody_bound' in model.y_indexes:
            ab42_plaque_bound_raw = sol.ys[:, model.y_indexes['AB42_Plaque_Antibody_bound']]
            ab42_plaque_bound_scaled = ab42_plaque_bound_raw / volume_scale_factor_isf
            ax4b.plot(years, ab42_plaque_bound_scaled, 
                     label='Antibody-bound', linewidth=2, color='C3')
        ax4b.set_xlabel('Time (years)', fontsize=12)
        ax4b.set_ylabel('Concentration (nM)', fontsize=12)
        ax4b.set_title('AB42 Plaque Dynamics', fontsize=14)
        ax4b.legend(fontsize=10)
        ax4b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig4.savefig(figures_dir / 'plaque_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        plot_microglia_dynamics(sol, model, plots_dir=figures_dir)
        
        print(f"Figures saved to {figures_dir}")
        
    except Exception as e:
        print(f"\nError creating plots: {e}")
        traceback.print_exc()

def plot_individual_oligomers(sol, model, drug_type="gantenerumab", plots_dir=None):
    """Plot individual oligomer sizes for both AB40 and AB42"""
    if plots_dir is None:
        plots_dir = script_dir / "simulation_plots/tellurium_steady_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), sharex=True)
    
    ab40_oligomers = {}
    ab42_oligomers = {}
    ab40_oligomers_bound = {}
    ab42_oligomers_bound = {}
    
    y_indexes = model.y_indexes
    
    # Volume scaling factor for ISF compartment
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    
    for species_name in y_indexes.keys():
        for i in range(2, 17):
            if f'AB40_Oligomer{i:02d}' == species_name:
                ab40_oligomers[i] = species_name
            elif f'AB40_Oligomer{i:02d}_Antibody_bound' == species_name:
                ab40_oligomers_bound[i] = species_name
        
        for i in range(2, 17):
            if f'AB42_Oligomer{i:02d}' == species_name:
                ab42_oligomers[i] = species_name
            elif f'AB42_Oligomer{i:02d}_Antibody_bound' == species_name:
                ab42_oligomers_bound[i] = species_name
    
    x_values = sol.ts / 24.0 / 365.0
    
    cmap_ab40 = plt.cm.viridis(np.linspace(0, 1, max(len(ab40_oligomers), 1)))
    cmap_ab42 = plt.cm.plasma(np.linspace(0, 1, max(len(ab42_oligomers), 1)))
    
    for i, (size, species) in enumerate(sorted(ab40_oligomers.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax1.semilogy(x_values, concentration_nM, 
                    label=f'AB40 Oligomer {size}', 
                    color=cmap_ab40[i], linewidth=2)
    
    for i, (size, species) in enumerate(sorted(ab40_oligomers_bound.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax3.semilogy(x_values, concentration_nM, 
                    label=f'AB40 Oligomer {size} (Bound)', 
                    color=cmap_ab40[i], linewidth=2)
    
    for i, (size, species) in enumerate(sorted(ab42_oligomers.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax2.semilogy(x_values, concentration_nM, 
                    label=f'AB42 Oligomer {size}', 
                    color=cmap_ab42[i], linewidth=2)
    
    for i, (size, species) in enumerate(sorted(ab42_oligomers_bound.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax4.semilogy(x_values, concentration_nM, 
                    label=f'AB42 Oligomer {size} (Bound)', 
                    color=cmap_ab42[i], linewidth=2)
    
    ax1.set_title('Unbound AB40 Oligomers', fontsize=30)
    ax1.set_ylabel('Concentration (nM)', fontsize=26)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    if len(ab40_oligomers) > 0:
        ax1.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_title('Unbound AB42 Oligomers', fontsize=30)
    ax2.set_ylabel('Concentration (nM)', fontsize=26)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    if len(ab42_oligomers) > 0:
        ax2.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax3.set_title('Antibody-Bound AB40 Oligomers', fontsize=30)
    ax3.set_xlabel('Time (years)', fontsize=26)
    ax3.set_ylabel('Concentration (nM)', fontsize=26)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    if len(ab40_oligomers_bound) > 0:
        ax3.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax4.set_title('Antibody-Bound AB42 Oligomers', fontsize=30)
    ax4.set_xlabel('Time (years)', fontsize=26)
    ax4.set_ylabel('Concentration (nM)', fontsize=26)
    ax4.tick_params(axis='both', which='major', labelsize=22)
    if len(ab42_oligomers_bound) > 0:
        ax4.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    fig.savefig(plots_dir / f'{drug_type.lower()}_oligomer_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_fibrils_and_plaques(sol, model, drug_type="gantenerumab", plots_dir=None):
    """Plot individual fibril sizes and plaque dynamics"""
    if plots_dir is None:
        plots_dir = script_dir / "simulation_plots/tellurium_steady_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig_fibrils, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), sharex=True)
    fig_plaques, ax_plaque = plt.subplots(figsize=(12, 8))
    
    ab40_fibrils = {}
    ab42_fibrils = {}
    ab40_fibrils_bound = {}
    ab42_fibrils_bound = {}
    plaque_species = []
    
    y_indexes = model.y_indexes
    
    # Volume scaling factors
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    volume_scale_factor_csf = 0.09875  # Division by volume for SAS concentration units
    
    for species_name in y_indexes.keys():
        for i in range(17, 25):
            if f'AB40_Fibril{i:02d}' == species_name:
                ab40_fibrils[i] = species_name
            elif f'AB40_Fibril{i:02d}_Antibody_bound' == species_name:
                ab40_fibrils_bound[i] = species_name
        
        for i in range(17, 25):
            if f'AB42_Fibril{i:02d}' == species_name:
                ab42_fibrils[i] = species_name
            elif f'AB42_Fibril{i:02d}_Antibody_bound' == species_name:
                ab42_fibrils_bound[i] = species_name
                
        if "plaque" in species_name.lower():
            plaque_species.append(species_name)
    
    x_values = sol.ts / 24.0 / 365.0
    
    cmap_ab40 = plt.cm.viridis(np.linspace(0, 1, max(len(ab40_fibrils), 1)))
    cmap_ab42 = plt.cm.plasma(np.linspace(0, 1, max(len(ab42_fibrils), 1)))
    cmap_plaques = plt.cm.tab10(np.linspace(0, 1, max(len(plaque_species), 1)))
    
    for i, (size, species) in enumerate(sorted(ab40_fibrils.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax1.semilogy(x_values, concentration_nM, 
                    label=f'AB40 Fibril {size}', 
                    color=cmap_ab40[i], linewidth=2)
    
    for i, (size, species) in enumerate(sorted(ab40_fibrils_bound.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax3.semilogy(x_values, concentration_nM, 
                    label=f'AB40 Fibril {size} (Bound)', 
                    color=cmap_ab40[i], linewidth=2)
    
    for i, (size, species) in enumerate(sorted(ab42_fibrils.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax2.semilogy(x_values, concentration_nM, 
                    label=f'AB42 Fibril {size}', 
                    color=cmap_ab42[i], linewidth=2)
    
    for i, (size, species) in enumerate(sorted(ab42_fibrils_bound.items())):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax4.semilogy(x_values, concentration_nM, 
                    label=f'AB42 Fibril {size} (Bound)', 
                    color=cmap_ab42[i], linewidth=2)
    
    for i, species in enumerate(sorted(plaque_species)):
        # Apply volume scaling for proper concentration units
        raw_concentration = sol.ys[:, y_indexes[species]]
        
        concentration_nM = raw_concentration / volume_scale_factor_isf
        ax_plaque.semilogy(x_values, concentration_nM, 
                         label=species, 
                         color=cmap_plaques[i % len(cmap_plaques)], linewidth=2)
    
    ax1.set_title('Unbound AB40 Fibrils', fontsize=30)
    ax1.set_ylabel('Concentration (nM)', fontsize=26)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    if len(ab40_fibrils) > 0:
        ax1.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_title('Unbound AB42 Fibrils', fontsize=30)
    ax2.set_ylabel('Concentration (nM)', fontsize=26)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    if len(ab42_fibrils) > 0:
        ax2.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax3.set_title('Antibody-Bound AB40 Fibrils', fontsize=30)
    ax3.set_xlabel('Time (years)', fontsize=26)
    ax3.set_ylabel('Concentration (nM)', fontsize=26)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    if len(ab40_fibrils_bound) > 0:
        ax3.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax4.set_title('Antibody-Bound AB42 Fibrils', fontsize=30)
    ax4.set_xlabel('Time (years)', fontsize=26)
    ax4.set_ylabel('Concentration (nM)', fontsize=26)
    ax4.tick_params(axis='both', which='major', labelsize=22)
    if len(ab42_fibrils_bound) > 0:
        ax4.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax_plaque.set_title('Plaque Dynamics', fontsize=30)
    ax_plaque.set_xlabel('Time (years)', fontsize=26)
    ax_plaque.set_ylabel('Concentration (nM)', fontsize=26)
    ax_plaque.tick_params(axis='both', which='major', labelsize=22)
    if len(plaque_species) > 0:
        ax_plaque.legend(fontsize=20)
    
    plt.figure(fig_fibrils.number)
    plt.tight_layout()
    plt.figure(fig_plaques.number)
    plt.tight_layout()
    
    fig_fibrils.savefig(plots_dir / f'{drug_type.lower()}_fibril_dynamics.png', 
                       dpi=300, bbox_inches='tight')
    fig_plaques.savefig(plots_dir / f'{drug_type.lower()}_plaque_dynamics.png', 
                       dpi=300, bbox_inches='tight')
    plt.close(fig_fibrils)
    plt.close(fig_plaques)

def print_model_info(sol, model):
    """Print information about the model to help with debugging"""
    if sol is None or model is None:
        print("No model information available.")
        return
    
    try:
        y_indexes = model.y_indexes
        
        print("\nModel Information:")
        print(f"Number of variables: {len(y_indexes)}")
        
        species_groups = {
            "AB Production": [], "AB40 Oligomers": [], "AB42 Oligomers": [],
            "AB40 Fibrils": [], "AB42 Fibrils": [], "Plaque/Antibody": [], "Other": []
        }
        
        for species in sorted(y_indexes.keys()):
            if species in ['APP', 'C99']: species_groups["AB Production"].append(species)
            elif "AB40" in species and "Fibril" in species: species_groups["AB40 Fibrils"].append(species)
            elif "AB42" in species and "Fibril" in species: species_groups["AB42 Fibrils"].append(species)
            elif "AB40" in species or "ABeta40" in species: species_groups["AB40 Oligomers"].append(species)
            elif "AB42" in species or "ABeta42" in species: species_groups["AB42 Oligomers"].append(species)
            elif "plaque" in species.lower() or "antibody" in species.lower(): species_groups["Plaque/Antibody"].append(species)
            else: species_groups["Other"].append(species)
        
        for group, species_list in species_groups.items():
            if species_list:
                print(f"\n{group}:")
                for species in sorted(species_list):
                    try:
                        final_conc = sol.ys[-1, y_indexes[species]]
                        print(f"  {species:<30}: {final_conc:.6e}")
                    except:
                        print(f"  {species:<30}: (error getting concentration)")
        
        print("\nInitial Conditions (selected):")
        key_species = ['AB40_Monomer', 'AB42_Monomer']
        for species in key_species:
            if species in y_indexes:
                try:
                    print(f"  {species:<30}: {model.y0[y_indexes[species]]:.6e}")
                except:
                    print(f"  {species:<30}: (error getting initial condition)")
        
    except Exception as e:
        print(f"\nError printing model information: {e}")
        traceback.print_exc()

def _setup_year_axis(ax, x_data):
    """Set up the x-axis to display years with appropriate formatting"""
    ax.set_xlim(min(x_data), max(x_data))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(True, which='major', alpha=0.3)

def plot_ab42_ratios_and_concentrations(sol, model, drug_type="gantenerumab", plots_dir=None):
    """Plot AB42/AB40 ratios and concentrations in different compartments with SUVR."""
    if plots_dir is None:
        plots_dir = script_dir / "simulation_plots/tellurium_steady_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 subplot layout: SUVR, ratio, ISF, CSF
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    x_values = sol.ts / 24.0 / 365.0
    start_idx = np.where(x_values >= (max(x_values) - 80))[0][0]
    x_filtered = x_values[start_idx:]
    
    y_indexes = model.y_indexes
    nM_to_pg = 4514.0
    
    # Volume scaling factors (same as in multi_dataset_experimental_fit.py)
    volume_scale_factor_csf = 0.09875  # Division by volume for SAS concentration units
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    
    # 1. SUVR Plot (top left)
    suvr = calculate_suvr(sol, model)
    suvr_filtered = suvr[start_idx:]
    
    ax1.plot(x_filtered, suvr_filtered, linewidth=3, color='blue', label='SUVR')
    ax1.set_ylabel('SUVR', fontsize=26)
    ax1.set_xlabel('Time (years)', fontsize=26)
    ax1.set_title(f'SUVR Progression', fontsize=30)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    
    # 2. Brain plasma ratios (top right)
    ab42_brain_plasma = sol.ys[:, y_indexes['AB42Mu_Brain_Plasma']]
    ab40_brain_plasma = sol.ys[:, y_indexes['AB40Mu_Brain_Plasma']]
    brain_plasma_ratio = np.where(ab40_brain_plasma > 0, ab42_brain_plasma / ab40_brain_plasma, 0)
    brain_plasma_ratio_filtered = brain_plasma_ratio[start_idx:]
    
    ax2.plot(x_filtered, brain_plasma_ratio_filtered, linewidth=2, color='blue', label='AB42/AB40 Ratio')
    ax2.set_ylabel('Ratio', fontsize=26)
    ax2.set_xlabel('Time (years)', fontsize=26)
    ax2.set_title('Brain Plasma AB42/AB40 Ratio', fontsize=30)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    
    # 3. ISF AB42 (bottom left)
    ab42_isf_raw = sol.ys[:, y_indexes['AB42_Monomer']]
    ab42_isf_nM = ab42_isf_raw / volume_scale_factor_isf
    ab42_isf = ab42_isf_nM * nM_to_pg
    ab42_isf_filtered = ab42_isf[start_idx:]
    
    ax3.semilogy(x_filtered, ab42_isf_filtered, linewidth=2, color='blue', label='Total ISF AB42')
    ax3.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax3.set_xlabel('Time (years)', fontsize=26)
    ax3.set_title('Total ISF AB42', fontsize=30)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    
    # 4. CSF SAS AB42 (bottom right)
    ab42_sas_raw = sol.ys[:, y_indexes['AB42Mu_SAS']]
    ab42_sas_nM = ab42_sas_raw / volume_scale_factor_csf
    ab42_sas = ab42_sas_nM * nM_to_pg
    ab42_sas_filtered = ab42_sas[start_idx:]
    
    ax4.plot(x_filtered, ab42_sas_filtered, linewidth=2, color='blue', label='CSF SAS AB42')
    ax4.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax4.set_xlabel('Time (years)', fontsize=26)
    ax4.set_title('CSF SAS AB42', fontsize=30)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout()
    
    fig.savefig(plots_dir / f'{drug_type.lower()}_ab42_ratios_and_concentrations.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Also create separate AB40 plots
    fig_ab40, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ISF AB40 (apply volume scaling)
    ab40_isf_raw = sol.ys[:, y_indexes['AB40_Monomer']]
    ab40_isf_nM = ab40_isf_raw / volume_scale_factor_isf
    ab40_isf = ab40_isf_nM * nM_to_pg
    ab40_isf_filtered = ab40_isf[start_idx:]
    
    ax5.semilogy(x_filtered, ab40_isf_filtered, linewidth=2, color='blue', label='Total ISF AB40')
    ax5.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax5.set_xlabel('Time (years)', fontsize=26)
    ax5.set_title('Total ISF AB40', fontsize=30)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=22)
    
    # CSF SAS AB40 (apply volume scaling)
    ab40_sas_raw = sol.ys[:, y_indexes['AB40Mu_SAS']]
    ab40_sas_nM = ab40_sas_raw / volume_scale_factor_csf
    ab40_sas = ab40_sas_nM * nM_to_pg
    ab40_sas_filtered = ab40_sas[start_idx:]
    
    ax6.semilogy(x_filtered, ab40_sas_filtered, linewidth=2, color='blue', label='CSF SAS AB40')
    ax6.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax6.set_xlabel('Time (years)', fontsize=26)
    ax6.set_title('CSF SAS AB40', fontsize=30)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=20)
    ax6.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout()
    
    fig_ab40.savefig(plots_dir / f'{drug_type.lower()}_ab40_concentrations.png', 
                     dpi=300, bbox_inches='tight')
    plt.close(fig_ab40)

def plot_microglia_dynamics(sol, model, plots_dir=None):
    """Plot microglia high fraction and cell count over time"""
    if plots_dir is None:
        plots_dir = script_dir / "simulation_plots/tellurium_steady_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    y_indexes = model.y_indexes
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    x_values = sol.ts / 24.0 / 365.0
    
    if 'Microglia_Hi_Fract' in y_indexes:
        ax1.plot(x_values, sol.ys[:, y_indexes['Microglia_Hi_Fract']], 
                linewidth=2.5, color='blue', label='High Activity Fraction')
        ax1.set_ylabel('Fraction', fontsize=14)
        ax1.set_title('Microglia High Activity Fraction', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
    else:
        print("Warning: Microglia_Hi_Fract not found in model")
    
    if 'Microglia_cell_count' in y_indexes:
        ax2.plot(x_values, sol.ys[:, y_indexes['Microglia_cell_count']], 
                linewidth=2.5, color='green', label='Cell Count')
        ax2.set_xlabel('Time (years)', fontsize=14)
        ax2.set_ylabel('Cell Count', fontsize=14)
        ax2.set_title('Microglia Cell Count', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
    else:
        print("Warning: Microglia_cell_count not found in model")
    
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    fig.savefig(plots_dir / 'microglia_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_suvr(sol, model, drug_type="gantenerumab", plots_dir=None):
    """Plot SUVR progression over time"""
    if plots_dir is None:
        plots_dir = script_dir / "simulation_plots/tellurium_steady_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate SUVR
    suvr = calculate_suvr(sol, model)
    
    # Convert time to years
    years = sol.ts / (24 * 365)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(years, suvr, linewidth=3, color='black', label='SUVR')
    ax.set_xlabel('Time (years)', fontsize=20)
    ax.set_ylabel('SUVR', fontsize=20)
    ax.set_title(f'{drug_type.capitalize()} SUVR Progression Over Time', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    fig.savefig(plots_dir / f'{drug_type.lower()}_suvr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a combined plot with plaque dynamics
    fig_combined, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot plaque dynamics
    y_indexes = model.y_indexes
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    
    ab40_plaque = sol.ys[:, y_indexes['AB40_Plaque_unbound']] / volume_scale_factor_isf
    ab42_plaque = sol.ys[:, y_indexes['AB42_Plaque_unbound']] / volume_scale_factor_isf
    
    ax1.plot(years, ab40_plaque, label='AB40 Plaque', linewidth=2, color='blue')
    ax1.plot(years, ab42_plaque, label='AB42 Plaque', linewidth=2, color='red')
    ax1.set_ylabel('Concentration (nM)', fontsize=16)
    ax1.set_title('Plaque Dynamics', fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot SUVR
    ax2.plot(years, suvr, label='SUVR', linewidth=3, color='black')
    ax2.set_xlabel('Time (years)', fontsize=16)
    ax2.set_ylabel('SUVR', fontsize=16)
    ax2.set_title('SUVR Progression', fontsize=18)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    fig_combined.savefig(plots_dir / f'{drug_type.lower()}_plaque_and_suvr.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return suvr

def get_ab42_ratios_and_concentrations_final_values(sol, model):
    """Get the final values of AB42/AB40 ratios and concentrations."""
    y_indexes = model.y_indexes
    
    nM_to_pg = 4514.0
    
    # Volume scaling factors (same as in multi_dataset_experimental_fit.py)
    volume_scale_factor_csf = 0.09875  # Division by volume for SAS concentration units
    volume_scale_factor_isf = 0.2505   # Division by volume for ISF concentration units
    
    # Brain plasma ratios (no volume scaling needed for ratios)
    ab42_brain_plasma = sol.ys[-1, y_indexes['AB42Mu_Brain_Plasma']]
    ab40_brain_plasma = sol.ys[-1, y_indexes['AB40Mu_Brain_Plasma']]
    brain_plasma_ratio = ab42_brain_plasma / ab40_brain_plasma if ab40_brain_plasma > 0 else 0
    
    # ISF AB42 (apply volume scaling)
    ab42_isf_raw = sol.ys[-1, y_indexes['AB42_Monomer']]
    ab42_isf_nM = ab42_isf_raw / volume_scale_factor_isf
    ab42_isf = ab42_isf_nM * nM_to_pg
    
    # CSF SAS AB42 (apply volume scaling)
    ab42_sas_raw = sol.ys[-1, y_indexes['AB42Mu_SAS']]
    ab42_sas_nM = ab42_sas_raw / volume_scale_factor_csf
    ab42_sas = ab42_sas_nM * nM_to_pg
    
    # ISF AB40 (apply volume scaling)
    ab40_isf_raw = sol.ys[-1, y_indexes['AB40_Monomer']]
    ab40_isf_nM = ab40_isf_raw / volume_scale_factor_isf
    ab40_isf = ab40_isf_nM * nM_to_pg
    
    # CSF SAS AB40 (apply volume scaling)
    ab40_sas_raw = sol.ys[-1, y_indexes['AB40Mu_SAS']]
    ab40_sas_nM = ab40_sas_raw / volume_scale_factor_csf
    ab40_sas = ab40_sas_nM * nM_to_pg
    
    return {
        'brain_plasma_ratio': brain_plasma_ratio,
        'ab42_isf_pg_ml': ab42_isf,
        'ab42_sas_pg_ml': ab42_sas,
        'ab40_isf_pg_ml': ab40_isf,
        'ab40_sas_pg_ml': ab40_sas
    }

def get_suvr_final_value(sol, model):
    """Get the final SUVR value."""
    suvr = calculate_suvr(sol, model)
    return suvr[-1]

def main():
    """Main function to visualize Tellurium simulation data"""
    parser = argparse.ArgumentParser(description="Visualize Tellurium simulation data")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                        default="gantenerumab", help="Drug type to visualize")
    parser.add_argument("--years", type=float, default=100.0,
                        help="Number of years simulated (used to select the correct CSV file)")
    args = parser.parse_args()
    
    print(f"\n=== TELLURIUM SIMULATION VISUALIZATION ===")
    print(f"Drug: {args.drug.upper()}")
    print(f"Years: {args.years}")
    print("=" * 40)
    
    print("\nLoading Tellurium simulation data...")
    time_points, species_data, model = load_tellurium_data(years=args.years)
    
    sol = create_solution_object(time_points, species_data)
    
    plots_dir = script_dir / "simulation_plots/tellurium_steady_state"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        #print_model_info(sol, model)
        
        final_values = get_ab42_ratios_and_concentrations_final_values(sol, model)
        if final_values:
            print("\nFinal AB42/AB40 Ratios and Concentrations:")
            print(f"Brain Plasma AB42/AB40 Ratio: {final_values['brain_plasma_ratio']:.4f}")
            print(f"Total ISF AB42: {final_values['ab42_isf_pg_ml']:.2f} pg/mL")
            print(f"CSF SAS AB42: {final_values['ab42_sas_pg_ml']:.2f} pg/mL")
            print(f"Total ISF AB40: {final_values['ab40_isf_pg_ml']:.2f} pg/mL")
            print(f"CSF SAS AB40: {final_values['ab40_sas_pg_ml']:.2f} pg/mL")
        
        # Calculate and display final SUVR value
        final_suvr = get_suvr_final_value(sol, model)
        print(f"Final SUVR: {final_suvr:.4f}")
        
        print("\nGenerating plots...")
        create_plots(sol, model)
        plot_individual_oligomers(sol, model, drug_type=args.drug, plots_dir=plots_dir)
        plot_fibrils_and_plaques(sol, model, drug_type=args.drug, plots_dir=plots_dir)
        plot_ab42_ratios_and_concentrations(sol, model, drug_type=args.drug, plots_dir=plots_dir)
        plot_suvr(sol, model, drug_type=args.drug, plots_dir=plots_dir)
        
        print(f"\nPlots saved to {plots_dir}")
        
    except Exception as e:
        print(f"\nError during visualization: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 