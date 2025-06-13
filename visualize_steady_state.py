"""
Script to visualize steady state data from run_combined_master_model.py using plotting functions
from run_no_dose_combined_master_model.py.
"""
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt
import traceback

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

# Add the project root to Python path
root_dir = Path(__file__).parents[1]  # Go up 1 level to reach models directory
sys.path.append(str(root_dir))

def load_steady_state_data(drug_type="gantenerumab", years=20):
    """Load steady state data from the saved CSV file
    
    Args:
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        years: Number of years simulated (used to select the correct CSV file)
        
    Returns:
        Tuple of (time_points, species_data, model)
    """
    # Load the saved data
    data_path = Path(f"generated/{years}_year_simulation_results_{drug_type.lower()}.csv")

    if not data_path.exists():
        print(f"Error: No steady state data found at {data_path}")
        print(f"Please run run_no_dose_combined_master_model.py with --years {years} first")
        sys.exit(1)
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Extract time points and species data
    time_points = df['time'].values
    species_data = df.drop('time', axis=1).values
    
    # Create a simple model object with y_indexes
    class SimpleModel:
        def __init__(self, species_names):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
            self.y0 = species_data[0]  # Use first time point as initial conditions
    
    # Create model object with species names
    model = SimpleModel(df.drop('time', axis=1).columns)
    
    return time_points, species_data, model

def create_solution_object(time_points, species_data):
    """Create a solution object compatible with the plotting functions
    
    Args:
        time_points: Array of time points
        species_data: Array of species concentrations
        
    Returns:
        Solution object with ts and ys attributes
    """
    class Solution:
        def __init__(self, ts, ys):
            self.ts = ts
            self.ys = ys
    
    return Solution(time_points, species_data)

def create_plots(sol, c, model):
    """Create plots of the simulation results similar to QSP model"""
    if sol is None:
        print("No solution data available for plotting.")
        return
    
    # Create figures directory if it doesn't exist
    figures_dir = Path("generated/figures/steady_state")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
       # Import the generated model to get species indices
        module_name = "generated.jax.combined_master_jax"
        import importlib
        try:
            jax_module = importlib.import_module(module_name)
            y_indexes = jax_module.y_indexes
            c_indexes = jax_module.c_indexes
            c = jax_module.c
        except ImportError:
            print(f"Error: Could not import module {module_name}")
            return
        
        # Convert time from hours to years
        years = sol.ts / (24 * 365)
        
        # 1. AB42/AB40 Ratios Plot
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        # Calculate total monomers
        ab40_monomer = sol.ys[:, y_indexes['AB40_Monomer']]
        ab42_monomer = sol.ys[:, y_indexes['AB42_Monomer']]
        monomer_ratio = ab42_monomer / ab40_monomer
        
        # Calculate total oligomers (2-16)
        ab40_oligomers = np.zeros_like(sol.ts)
        ab42_oligomers = np.zeros_like(sol.ts)
        for i in range(2, 17):
            name40 = f'AB40_Oligomer{i:02d}'
            name42 = f'AB42_Oligomer{i:02d}'
            if name40 in y_indexes:
                ab40_oligomers += sol.ys[:, y_indexes[name40]]
            if name42 in y_indexes:
                ab42_oligomers += sol.ys[:, y_indexes[name42]]
        oligomer_ratio = ab42_oligomers / ab40_oligomers
        
        # Calculate total fibrils (17-23)
        ab40_fibrils = np.zeros_like(sol.ts)
        ab42_fibrils = np.zeros_like(sol.ts)
        for i in range(17, 24):
            name40 = f'AB40_Fibril{i:02d}'
            name42 = f'AB42_Fibril{i:02d}'
            if name40 in y_indexes:
                ab40_fibrils += sol.ys[:, y_indexes[name40]]
            if name42 in y_indexes:
                ab42_fibrils += sol.ys[:, y_indexes[name42]]
        fibril_ratio = ab42_fibrils / ab40_fibrils
        
        # Calculate plaque ratio
        ab40_plaque = sol.ys[:, y_indexes['AB40_Plaque_unbound']]
        ab42_plaque = sol.ys[:, y_indexes['AB42_Plaque_unbound']]
        plaque_ratio = ab42_plaque / ab40_plaque
        
        # Plot ratios
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
        
        # 2. Oligomer Loads
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate oligomer loads
        ab40_oligomer_load = ab40_monomer + ab40_oligomers
        ab42_oligomer_load = ab42_monomer + ab42_oligomers
        
        # Plot AB40 oligomer load
        ax2a.plot(years, ab40_oligomer_load / c[c_indexes['VIS_brain']], label='Oligomer Load', linewidth=2, color='C0')
        ax2a.set_xlabel('Time (years)', fontsize=12)
        ax2a.set_ylabel('Load (nM)', fontsize=12)
        ax2a.set_title('AB40 Oligomer Load', fontsize=14)
        ax2a.legend(fontsize=10)
        ax2a.grid(True, alpha=0.3)
        
        # Plot AB42 oligomer load
        ax2b.plot(years, ab42_oligomer_load / c[c_indexes['VIS_brain']], label='Oligomer Load', linewidth=2, color='C0')
        ax2b.set_xlabel('Time (years)', fontsize=12)
        ax2b.set_ylabel('Load (nM)', fontsize=12)
        ax2b.set_title('AB42 Oligomer Load', fontsize=14)
        ax2b.legend(fontsize=10)
        ax2b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig2.savefig(figures_dir / 'oligomer_load.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Protofibril Loads
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot AB40 protofibril load
        ax3a.plot(years, ab40_fibrils / c[c_indexes['VIS_brain']], label='Protofibril Load', linewidth=2, color='C1')
        ax3a.set_xlabel('Time (years)', fontsize=12)
        ax3a.set_ylabel('Load (nM)', fontsize=12)
        ax3a.set_title('AB40 Protofibril Load', fontsize=14)
        ax3a.legend(fontsize=10)
        ax3a.grid(True, alpha=0.3)
        
        # Plot AB42 protofibril load
        ax3b.plot(years, ab42_fibrils / c[c_indexes['VIS_brain']], label='Protofibril Load', linewidth=2, color='C1')
        ax3b.set_xlabel('Time (years)', fontsize=12)
        ax3b.set_ylabel('Load (nM)', fontsize=12)
        ax3b.set_title('AB42 Protofibril Load', fontsize=14)
        ax3b.legend(fontsize=10)
        ax3b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig3.savefig(figures_dir / 'protofibril_load.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Plaque Dynamics
        fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot AB40 plaque dynamics
        ax4a.plot(years, sol.ys[:, y_indexes['AB40_Plaque_unbound']] / c[c_indexes['VIS_brain']], 
                 label='Unbound', linewidth=2, color='C2')
        if 'AB40_Plaque_Antibody_bound' in y_indexes:
            ax4a.plot(years, sol.ys[:, y_indexes['AB40_Plaque_Antibody_bound']], 
                     label='Antibody-bound', linewidth=2, color='C3')
        ax4a.set_xlabel('Time (years)', fontsize=12)
        ax4a.set_ylabel('Concentration (nM)', fontsize=12)
        ax4a.set_title('AB40 Plaque Dynamics', fontsize=14)
        ax4a.legend(fontsize=10)
        ax4a.grid(True, alpha=0.3)
        
        # Plot AB42 plaque dynamics
        ax4b.plot(years, sol.ys[:, y_indexes['AB42_Plaque_unbound']] / c[c_indexes['VIS_brain']], 
                 label='Unbound', linewidth=2, color='C2')
        if 'AB42_Plaque_Antibody_bound' in y_indexes:
            ax4b.plot(years, sol.ys[:, y_indexes['AB42_Plaque_Antibody_bound']], 
                     label='Antibody-bound', linewidth=2, color='C3')
        ax4b.set_xlabel('Time (years)', fontsize=12)
        ax4b.set_ylabel('Concentration (nM)', fontsize=12)
        ax4b.set_title('AB42 Plaque Dynamics', fontsize=14)
        ax4b.legend(fontsize=10)
        ax4b.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig4.savefig(figures_dir / 'plaque_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # Add microglia dynamics plot
        plot_microglia_dynamics(sol, plots_dir=figures_dir)
        
        print(f"Figures saved to {figures_dir}")
        
    except Exception as e:
        print(f"\nError creating plots: {e}")
        traceback.print_exc()

def plot_individual_oligomers(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot individual oligomer sizes for both AB40 and AB42"""
    if plots_dir is None:
        plots_dir = Path("generated/figures/steady_state")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), sharex=True)
    
    # Dictionary to store oligomer species found in the model
    ab40_oligomers = {}
    ab42_oligomers = {}
    ab40_oligomers_bound = {}
    ab42_oligomers_bound = {}
    
    # Find all oligomer species
    for species_name in sol.model.y_indexes.keys():
        # AB40 oligomers
        for i in range(2, 17):
            if f'AB40_Oligomer{i:02d}' == species_name:
                ab40_oligomers[i] = species_name
            elif f'AB40_Oligomer{i:02d}_Antibody_bound' == species_name:
                ab40_oligomers_bound[i] = species_name
        
        # AB42 oligomers
        for i in range(2, 17):
            if f'AB42_Oligomer{i:02d}' == species_name:
                ab42_oligomers[i] = species_name
            elif f'AB42_Oligomer{i:02d}_Antibody_bound' == species_name:
                ab42_oligomers_bound[i] = species_name
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Create color maps for different oligomer sizes
    cmap_ab40 = plt.cm.viridis(np.linspace(0, 1, max(len(ab40_oligomers), 1)))
    cmap_ab42 = plt.cm.plasma(np.linspace(0, 1, max(len(ab42_oligomers), 1)))
    
    # Plot unbound AB40 oligomers
    for i, (size, species) in enumerate(sorted(ab40_oligomers.items())):
        ax1.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB40 Oligomer {size}', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot bound AB40 oligomers
    for i, (size, species) in enumerate(sorted(ab40_oligomers_bound.items())):
        ax3.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB40 Oligomer {size} (Bound)', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot unbound AB42 oligomers
    for i, (size, species) in enumerate(sorted(ab42_oligomers.items())):
        ax2.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB42 Oligomer {size}', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Plot bound AB42 oligomers
    for i, (size, species) in enumerate(sorted(ab42_oligomers_bound.items())):
        ax4.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB42 Oligomer {size} (Bound)', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Set titles and labels
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
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_type.lower()}_oligomer_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_fibrils_and_plaques(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot individual fibril sizes and plaque dynamics"""
    if plots_dir is None:
        plots_dir = Path("generated/figures/steady_state")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create two figures: one for fibrils, one for plaques
    fig_fibrils, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), sharex=True)
    fig_plaques, ax_plaque = plt.subplots(figsize=(12, 8))
    
    # Dictionary to store fibril species found in the model
    ab40_fibrils = {}
    ab42_fibrils = {}
    ab40_fibrils_bound = {}
    ab42_fibrils_bound = {}
    plaque_species = []
    
    # Find all fibril and plaque species
    for species_name in sol.model.y_indexes.keys():
        # AB40 fibrils
        for i in range(17, 25):
            if f'AB40_Fibril{i:02d}' == species_name:
                ab40_fibrils[i] = species_name
            elif f'AB40_Fibril{i:02d}_Antibody_bound' == species_name:
                ab40_fibrils_bound[i] = species_name
        
        # AB42 fibrils
        for i in range(17, 25):
            if f'AB42_Fibril{i:02d}' == species_name:
                ab42_fibrils[i] = species_name
            elif f'AB42_Fibril{i:02d}_Antibody_bound' == species_name:
                ab42_fibrils_bound[i] = species_name
                
        # Plaque species
        if "plaque" in species_name.lower():
            plaque_species.append(species_name)
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Create color maps for different fibril sizes
    cmap_ab40 = plt.cm.viridis(np.linspace(0, 1, max(len(ab40_fibrils), 1)))
    cmap_ab42 = plt.cm.plasma(np.linspace(0, 1, max(len(ab42_fibrils), 1)))
    cmap_plaques = plt.cm.tab10(np.linspace(0, 1, max(len(plaque_species), 1)))
    
    # Plot unbound AB40 fibrils
    for i, (size, species) in enumerate(sorted(ab40_fibrils.items())):
        ax1.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB40 Fibril {size}', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot bound AB40 fibrils
    for i, (size, species) in enumerate(sorted(ab40_fibrils_bound.items())):
        ax3.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB40 Fibril {size} (Bound)', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot unbound AB42 fibrils
    for i, (size, species) in enumerate(sorted(ab42_fibrils.items())):
        ax2.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB42 Fibril {size}', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Plot bound AB42 fibrils
    for i, (size, species) in enumerate(sorted(ab42_fibrils_bound.items())):
        ax4.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                    label=f'AB42 Fibril {size} (Bound)', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Plot plaque species
    for i, species in enumerate(sorted(plaque_species)):
        ax_plaque.semilogy(x_values, sol.ys[:, sol.model.y_indexes[species]], 
                         label=species, 
                         color=cmap_plaques[i % len(cmap_plaques)], linewidth=2)
    
    # Set titles and labels for fibril plot
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
    
    # Set titles and labels for plaque plot
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
    
    # Save figures
    fig_fibrils.savefig(plots_dir / f'{drug_type.lower()}_fibril_dynamics.png', 
                       dpi=300, bbox_inches='tight')
    fig_plaques.savefig(plots_dir / f'{drug_type.lower()}_plaque_dynamics.png', 
                       dpi=300, bbox_inches='tight')
    plt.close(fig_fibrils)
    plt.close(fig_plaques)

def print_model_info(sol, model):
    """Print information about the model to help with debugging
    
    Args:
        sol: Solution from the ODE solver
        model: The model object with species indices
    """
    if sol is None or model is None:
        print("No model information available.")
        return
    
    try:
        # Get species indices
        y_indexes = model.y_indexes
        
        # Print the number of species and their names
        print("\nModel Information:")
        print(f"Number of species: {len(y_indexes)}")
        
        # Group species by type for better readability
        species_groups = {
            "AB Production": [],
            "AB40 Oligomers": [],
            "AB42 Oligomers": [],
            "AB40 Fibrils": [],
            "AB42 Fibrils": [],
            "Plaque/Antibody": [],
            "Other": []
        }
        
        for species in sorted(y_indexes.keys()):
            if species in ['APP', 'C99']:
                species_groups["AB Production"].append(species)
            elif "AB40" in species and "Fibril" in species:
                species_groups["AB40 Fibrils"].append(species)
            elif "AB42" in species and "Fibril" in species:
                species_groups["AB42 Fibrils"].append(species)
            elif "AB40" in species or "ABeta40" in species:
                species_groups["AB40 Oligomers"].append(species)
            elif "AB42" in species or "ABeta42" in species:
                species_groups["AB42 Oligomers"].append(species)
            elif "plaque" in species.lower() or "antibody" in species.lower():
                species_groups["Plaque/Antibody"].append(species)
            else:
                species_groups["Other"].append(species)
        
        # Print each group
        for group, species_list in species_groups.items():
            if species_list:
                print(f"\n{group}:")
                for species in sorted(species_list):
                    # Try to get final concentration
                    try:
                        final_conc = sol.ys[-1, y_indexes[species]]
                        print(f"  {species:<30}: {final_conc:.6e} M")
                    except:
                        print(f"  {species:<30}: (error getting concentration)")
        
        # Print initial conditions for key species
        print("\nInitial Conditions (selected):")
        key_species = ['AB40_Monomer', 'AB42_Monomer']
        for species in key_species:
            if species in y_indexes:
                try:
                    print(f"  {species:<30}: {model.y0[y_indexes[species]]:.6e} M")
                except:
                    print(f"  {species:<30}: (error getting initial condition)")
        
    except Exception as e:
        print(f"\nError printing model information: {e}")
        traceback.print_exc()

def _setup_year_axis(ax, x_data):
    """Set up the x-axis to display years with appropriate formatting
    
    Args:
        ax: Matplotlib axis object
        x_data: Array of x-axis values in years
    """
    # Set x-axis limits to span the data
    ax.set_xlim(min(x_data), max(x_data))
    
    # Format x-axis ticks to show years
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Show 5 major ticks
    
    # Add grid
    ax.grid(True, which='major', alpha=0.3)

def plot_ab42_ratios_and_concentrations(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot AB42/AB40 ratios and concentrations in different compartments in two separate figures"""
    if plots_dir is None:
        plots_dir = Path("generated/figures/steady_state")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create two figures: one for AB42 and one for AB40
    fig_ab42, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig_ab40, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Filter data to show only the last 80 years
    start_idx = np.where(x_values >= (max(x_values) - 80))[0][0]
    x_filtered = x_values[start_idx:]
    
    # Import the generated model to get species indices
    module_name = "generated.jax.combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
        c_indexes = jax_module.c_indexes
        c = jax_module.c
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # AB 1-42 MolWt 4514 g/mol REF  https://pubchem.ncbi.nlm.nih.gov/compound/beta-Amyloid-Peptide-_1-42_-_human
    # units all cancel out in g/mol to pg/ml
    nM_to_pg = 4514.0
    
    # Calculate AB42/AB40 ratio in Brain Plasma
    ab42_brain_plasma = sol.ys[:, y_indexes['AB42Mu_Brain_Plasma']]
    ab40_brain_plasma = sol.ys[:, y_indexes['AB40Mu_Brain_Plasma']]
    brain_plasma_ratio = np.where(ab40_brain_plasma > 0, ab42_brain_plasma / ab40_brain_plasma, 0)
    
    # Filter the ratio data for plotting
    brain_plasma_ratio_filtered = brain_plasma_ratio[start_idx:]
    
    # Plot Brain Plasma ratio with filtered data
    ax1.plot(x_filtered, brain_plasma_ratio_filtered, linewidth=2, color='blue', label='AB42/AB40 Ratio')
    ax1.set_ylabel('Ratio', fontsize=26)
    ax1.set_xlabel('Time (years)', fontsize=26)
    ax1.set_title('Brain Plasma AB42/AB40 Ratio', fontsize=30)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=20)
    ax1.set_ylim(0.06, 0.15)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    
    # Calculate total ISF AB42 and convert to pg/mL
    ab42_isf = (sol.ys[:, y_indexes['AB42_Monomer']]/c[c_indexes['VIS_brain']]) * nM_to_pg
    # Filter for plotting
    ab42_isf_filtered = ab42_isf[start_idx:]
    
    ax2.semilogy(x_filtered, ab42_isf_filtered, linewidth=2, color='blue', label='Total ISF AB42')
    ax2.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax2.set_xlabel('Time (years)', fontsize=26)
    ax2.set_title('Total ISF AB42', fontsize=30)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    
    # Get AB42 in SAS and convert to pg/mL
    ab42_sas = (sol.ys[:, y_indexes['AB42Mu_SAS']]/c[c_indexes['V_SAS_brain']]) * nM_to_pg
    # Filter for plotting
    ab42_sas_filtered = ab42_sas[start_idx:]
    
    ax3.plot(x_filtered, ab42_sas_filtered, linewidth=2, color='blue', label='CSF SAS AB42')
    ax3.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax3.set_xlabel('Time (years)', fontsize=26)
    ax3.set_title('CSF SAS AB42', fontsize=30)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    
    # Second figure: AB40 concentrations
    # Calculate total ISF AB40 and convert to pg/mL
    ab40_isf = (sol.ys[:, y_indexes['AB40_Monomer']]/c[c_indexes['VIS_brain']]) * nM_to_pg
    ab40_isf_filtered = ab40_isf[start_idx:]
    
    ax4.semilogy(x_filtered, ab40_isf_filtered, linewidth=2, color='blue', label='Total ISF AB40')
    ax4.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax4.set_xlabel('Time (years)', fontsize=26)
    ax4.set_title('Total ISF AB40', fontsize=30)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=22)
    
    # Get AB40 in SAS and convert to pg/mL
    ab40_sas = (sol.ys[:, y_indexes['AB40Mu_SAS']]/c[c_indexes['V_SAS_brain']]) * nM_to_pg  
    ab40_sas_filtered = ab40_sas[start_idx:]
    
    ax5.semilogy(x_filtered, ab40_sas_filtered, linewidth=2, color='blue', label='CSF SAS AB40')
    ax5.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax5.set_xlabel('Time (years)', fontsize=26)
    ax5.set_title('CSF SAS AB40', fontsize=30)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=22)
    
    plt.figure(fig_ab42.number)
    plt.tight_layout()
    plt.figure(fig_ab40.number)
    plt.tight_layout()
    
    # Save figures
    fig_ab42.savefig(plots_dir / f'{drug_type.lower()}_ab42_ratios_and_concentrations.png', 
                     dpi=300, bbox_inches='tight')
    fig_ab40.savefig(plots_dir / f'{drug_type.lower()}_ab40_concentrations.png', 
                     dpi=300, bbox_inches='tight')
    plt.close(fig_ab42)
    plt.close(fig_ab40)

def plot_microglia_dynamics(sol, plots_dir=None):
    """Plot microglia high fraction and cell count over time"""
    if plots_dir is None:
        plots_dir = Path("generated/figures/steady_state")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "generated.jax.combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Create figure with two subplots (one for fraction, one for count)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Plot Microglia High Fraction
    if 'Microglia_Hi_Fract' in y_indexes:
        ax1.plot(x_values, sol.ys[:, y_indexes['Microglia_Hi_Fract']], 
                linewidth=2.5, color='blue', label='High Activity Fraction')
        ax1.set_ylabel('Fraction', fontsize=14)
        ax1.set_title('Microglia High Activity Fraction', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
    else:
        print("Warning: Microglia_Hi_Fract not found in model")
    
    # Plot Microglia Cell Count
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
    
    # Set x-axis limits and formatting
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(plots_dir / 'microglia_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

def get_ab42_ratios_and_concentrations_final_values(sol):
    """Get the final values of AB42/AB40 ratios and concentrations
    
    Args:
        sol: Solution object containing time points and species data
        
    Returns:
        Dictionary containing final values for:
        - brain_plasma_ratio: AB42/AB40 ratio in brain plasma
        - ab42_isf_pg_ml: Total ISF AB42 concentration in pg/mL
        - ab42_sas_pg_ml: CSF SAS AB42 concentration in pg/mL
        - ab40_isf_pg_ml: Total ISF AB40 concentration in pg/mL
        - ab40_sas_pg_ml: CSF SAS AB40 concentration in pg/mL
    """
    # Import the generated model to get species indices
    module_name = "generated.jax.combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
        c_indexes = jax_module.c_indexes
        c = jax_module.c
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return None
    
    # AB 1-42 MolWt 4514 g/mol REF  https://pubchem.ncbi.nlm.nih.gov/compound/beta-Amyloid-Peptide-_1-42_-_human
    # units all cancel out in g/mol to pg/ml
    nM_to_pg =  4514.0
    
    # Calculate final values
    ab42_brain_plasma = sol.ys[-1, y_indexes['AB42Mu_Brain_Plasma']]
    ab40_brain_plasma = sol.ys[-1, y_indexes['AB40Mu_Brain_Plasma']]
    brain_plasma_ratio = ab42_brain_plasma / ab40_brain_plasma if ab40_brain_plasma > 0 else 0
    
    ab42_isf = (sol.ys[-1, y_indexes['AB42_Monomer']] / c[c_indexes['VIS_brain']]) * nM_to_pg   
    ab42_sas = (sol.ys[-1, y_indexes['AB42Mu_SAS']] / c[c_indexes['V_SAS_brain']]) * nM_to_pg
    ab40_isf = (sol.ys[-1, y_indexes['AB40_Monomer']] / c[c_indexes['VIS_brain']]) * nM_to_pg
    ab40_sas = (sol.ys[-1, y_indexes['AB40Mu_SAS']] / c[c_indexes['V_SAS_brain']]) * nM_to_pg
    
    return {
        'brain_plasma_ratio': brain_plasma_ratio,
        'ab42_isf_pg_ml': ab42_isf,
        'ab42_sas_pg_ml': ab42_sas,
        'ab40_isf_pg_ml': ab40_isf,
        'ab40_sas_pg_ml': ab40_sas
    }

def main():
    """Main function to visualize steady state data"""
    parser = argparse.ArgumentParser(description="Visualize steady state data from combined master model")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                        default="gantenerumab", help="Drug type to visualize")
    parser.add_argument("--years", type=int, default=20,
                        help="Number of years simulated (used to select the correct CSV file)")
    args = parser.parse_args()
    
    # Print summary of visualization settings
    print(f"\n=== STEADY STATE DATA VISUALIZATION ===")
    print(f"Drug: {args.drug.upper()}")
    print(f"Years: {args.years}")
    print("=" * 40)
    
    # Load steady state data
    print("\nLoading steady state data...")
    time_points, species_data, model = load_steady_state_data(drug_type=args.drug, years=args.years)
    
    # Create solution object
    sol = create_solution_object(time_points, species_data)
    sol.model = model  # Add model to solution object for plotting functions
    
    # Create output directory for plots
    plots_dir = Path("generated/figures/steady_state")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Print model information
        print_model_info(sol, model)
        
        # Get and print final AB42/AB40 ratios and concentrations
        final_values = get_ab42_ratios_and_concentrations_final_values(sol)
        if final_values:
            print("\nFinal AB42/AB40 Ratios and Concentrations:")
            print(f"Brain Plasma AB42/AB40 Ratio: {final_values['brain_plasma_ratio']:.4f}")
            print(f"Total ISF AB42: {final_values['ab42_isf_pg_ml']:.2f} pg/mL")
            print(f"CSF SAS AB42: {final_values['ab42_sas_pg_ml']:.2f} pg/mL")
            print(f"Total ISF AB40: {final_values['ab40_isf_pg_ml']:.2f} pg/mL")
            print(f"CSF SAS AB40: {final_values['ab40_sas_pg_ml']:.2f} pg/mL")
        
        # Create plots
        print("\nGenerating plots...")
        create_plots(sol, None, model)  # Pass None for constants as they're not needed for these plots
        plot_individual_oligomers(sol, drug_type=args.drug, plots_dir=plots_dir)
        plot_fibrils_and_plaques(sol, drug_type=args.drug, plots_dir=plots_dir)
        plot_ab42_ratios_and_concentrations(sol, drug_type=args.drug, plots_dir=plots_dir)
        
        print(f"\nPlots saved to {plots_dir}")
        
    except Exception as e:
        print(f"\nError creating plots: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 