"""
Run script for the combined master model.
This script runs the combined model that includes both AB_Master_Model and Geerts_Master_Model components.
It first runs to steady state without antibody dosing, then uses those results as initial conditions
for a simulation with antibody dosing.
"""
import os
os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import diffrax
from diffrax import Tsit5, ODETerm, SaveAt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import argparse
import time
from sbmltoodejax import parse
from sbmltoodejax.modulegeneration import GenerateModel
import libsbml
import traceback

# Add the project root to Python path
root_dir = Path(__file__).parents[1]  # Go up 1 level to reach models directory
sys.path.append(str(root_dir))

# Import the master model
from Geerts.Modules.Combined_Master_Model import create_combined_master_model, load_parameters_from_csv, save_model

def generate_jax_model(drug_type="gantenerumab"):
    """Generate JAX model from the combined master model
    
    Args:
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        
    Returns:
        Tuple of (document, jax_path)
    """
    xml_path = Path("generated/sbml/combined_master_model.xml")
    jax_path = Path("generated/jax/combined_master_jax.py")
    
    # Create output directories if they don't exist
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    jax_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a new SBML model
    print("\nCreating new SBML model for combined dynamics...")
    
    try:
        # Load parameters
        params, params_with_units = load_parameters_from_csv(
            "parameters/PK_Geerts.csv",
            drug_type=drug_type,
        )
        
        # Generate the combined master model
        document = create_combined_master_model(
            params,
            params_with_units,
            drug_type=drug_type,
        )
        
        # Save SBML file
        save_model(document, str(xml_path))
        print(f"SBML model successfully created at {xml_path}")
        
        # Parse SBML and generate JAX model
        print("\nParsing combined master model SBML...")
        model_data = parse.ParseSBMLFile(str(xml_path))
        
        print("\nGenerating JAX model...")
        GenerateModel(model_data, str(jax_path))
        print(f"JAX model successfully generated at {jax_path}")
        
        return document, jax_path
        
    except Exception as e:
        print(f"\nError generating model: {e}")
        # Add this line to exit when model generation fails
        sys.exit(f"Model generation failed. Cannot continue with simulation.")

def run_simulation(time_hours=2040, drug_type="gantenerumab"):
    """Run the combined master model simulation without drug dosing
    
    Args:
        time_hours: Simulation duration in hours
        drug_type: Either "lecanemab" or "gantenerumab" (affects other parameters)
        
    Returns:
        Tuple of (solution, constants)
    """
    # Import the generated model
    sys.path.append(str(Path("generated/jax").absolute()))
    
    print(f"\n=== Running {drug_type.upper()} simulation for {time_hours} hours (no drug dosing) ===")

    # Generate JAX model
    _, jax_path = generate_jax_model(drug_type=drug_type)
    
    # Add generated JAX module directory to path
    jax_dir = Path("generated/jax").absolute()
    if str(jax_dir) not in sys.path:
        sys.path.append(str(jax_dir))
    
    # Import the generated model
    module_name = "combined_master_jax"
    print(f"Importing generated JAX model: {module_name}...")
    
    # Dynamic import of the generated module
    import importlib
    importlib.invalidate_caches()
    combined_model = importlib.import_module(module_name)
    
    # Access model components
    RateofSpeciesChange = combined_model.RateofSpeciesChange
    AssignmentRule = combined_model.AssignmentRule
    y0 = combined_model.y0
    w0 = combined_model.w0
    t0 = combined_model.t0
    c = combined_model.c
    y_indexes = combined_model.y_indexes
    c_indexes = combined_model.c_indexes
    
    # Print species names and indices for verification
    print("\nSpecies names and indices:")
    for species_name, idx in sorted(y_indexes.items()):
        print(f"{species_name:<30}: {idx}")
    
    # Create rate of change function
    rate_of_change = RateofSpeciesChange()
    assignment_rule = AssignmentRule()
    
    @jit
    def combined_ode_func(t, y, args):
        w, c = args
        w = assignment_rule(y, w, c, t)
        dy_dt = rate_of_change(y, t, w, c)
        return dy_dt
    
    # Create mutable copy of constants and set all dosing parameters to zero
    c_mutable = c.copy()
    c_mutable = c_mutable.at[c_indexes['MaxDosingTime']].set(0.0)
    c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(0.0)
    c_mutable = c_mutable.at[c_indexes['SC_DoseAmount']].set(0.0)
    c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(0.0)
    c_mutable = c_mutable.at[c_indexes['SC_NumDoses']].set(0.0)
    
    # Set initial conditions to zero for drug compartments
    current_y0 = y0.copy()
    if drug_type.lower() == "lecanemab":
        current_y0 = current_y0.at[y_indexes['PK_central']].set(0.0)
    else:
        current_y0 = current_y0.at[y_indexes['SubCut_absorption']].set(0.0)
    
    # Simulation parameters
    t1 = float(time_hours)
    dt = 0.001
    # Harmonize n_steps to monthly save points (same as run_no_dose.py)
    years = time_hours / (24 * 365)
    n_steps = 1000
    
    # Create diffrax solver
    term = ODETerm(combined_ode_func)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))
    
    print("\nRunning simulation...")
    start_time = time.time()
    
    # Solve the ODE system with harmonized PIDController
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=current_y0,
        args=(w0, c_mutable),
        saveat=saveat,
        max_steps=10000000000,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8, pcoeff=0.4, icoeff=0.3),
        progress_meter=diffrax.TextProgressMeter()
    )
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return sol, c_mutable, combined_model

def plot_individual_oligomers(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot individual oligomer sizes for both AB40 and AB42"""
    # Use provided plots_dir or default to generated/figures
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    
    # Dynamic import of the correct module
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), sharex=True)
    
    # Dictionary to store oligomer species found in the model
    ab40_oligomers = {}
    ab42_oligomers = {}
    ab40_oligomers_bound = {}
    ab42_oligomers_bound = {}
    
    # Find all oligomer species
    for species_name in y_indexes.keys():
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
        ax1.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB40 Oligomer {size}', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot bound AB40 oligomers
    for i, (size, species) in enumerate(sorted(ab40_oligomers_bound.items())):
        ax3.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB40 Oligomer {size} (Bound)', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot unbound AB42 oligomers
    for i, (size, species) in enumerate(sorted(ab42_oligomers.items())):
        ax2.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB42 Oligomer {size}', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Plot bound AB42 oligomers
    for i, (size, species) in enumerate(sorted(ab42_oligomers_bound.items())):
        ax4.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB42 Oligomer {size} (Bound)', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Set titles and labels
    ax1.set_title('Unbound AB40 Oligomers', fontsize=14)
    ax1.set_ylabel('Concentration (nM)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    if len(ab40_oligomers) > 0:
        ax1.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_title('Unbound AB42 Oligomers', fontsize=14)
    ax2.set_ylabel('Concentration (nM)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if len(ab42_oligomers) > 0:
        ax2.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax3.set_title('Antibody-Bound AB40 Oligomers', fontsize=14)
    ax3.set_xlabel('Time (years)', fontsize=12)
    ax3.set_ylabel('Concentration (nM)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if len(ab40_oligomers_bound) > 0:
        ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax4.set_title('Antibody-Bound AB42 Oligomers', fontsize=14)
    ax4.set_xlabel('Time (years)', fontsize=12)
    ax4.set_ylabel('Concentration (nM)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    if len(ab42_oligomers_bound) > 0:
        ax4.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_type.lower()}_oligomer_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_fibrils_and_plaques(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot individual fibril sizes and plaque dynamics"""
    # Use provided plots_dir or default to generated/figures
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    
    # Dynamic import of the correct module
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
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
    for species_name in y_indexes.keys():
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
        ax1.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB40 Fibril {size}', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot bound AB40 fibrils
    for i, (size, species) in enumerate(sorted(ab40_fibrils_bound.items())):
        ax3.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB40 Fibril {size} (Bound)', 
                    color=cmap_ab40[i], linewidth=2)
    
    # Plot unbound AB42 fibrils
    for i, (size, species) in enumerate(sorted(ab42_fibrils.items())):
        ax2.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB42 Fibril {size}', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Plot bound AB42 fibrils
    for i, (size, species) in enumerate(sorted(ab42_fibrils_bound.items())):
        ax4.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                    label=f'AB42 Fibril {size} (Bound)', 
                    color=cmap_ab42[i], linewidth=2)
    
    # Plot plaque species
    for i, species in enumerate(sorted(plaque_species)):
        ax_plaque.semilogy(x_values, sol.ys[:, y_indexes[species]], 
                         label=species, 
                         color=cmap_plaques[i % len(cmap_plaques)], linewidth=2)
    
    # Set titles and labels for fibril plot
    ax1.set_title('Unbound AB40 Fibrils', fontsize=14)
    ax1.set_ylabel('Concentration (nM)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    if len(ab40_fibrils) > 0:
        ax1.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_title('Unbound AB42 Fibrils', fontsize=14)
    ax2.set_ylabel('Concentration (nM)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if len(ab42_fibrils) > 0:
        ax2.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax3.set_title('Antibody-Bound AB40 Fibrils', fontsize=14)
    ax3.set_xlabel('Time (years)', fontsize=12)
    ax3.set_ylabel('Concentration (nM)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if len(ab40_fibrils_bound) > 0:
        ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax4.set_title('Antibody-Bound AB42 Fibrils', fontsize=14)
    ax4.set_xlabel('Time (years)', fontsize=12)
    ax4.set_ylabel('Concentration (nM)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    if len(ab42_fibrils_bound) > 0:
        ax4.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set titles and labels for plaque plot
    ax_plaque.set_title('Plaque Dynamics', fontsize=14)
    ax_plaque.set_xlabel('Time (years)', fontsize=12)
    ax_plaque.set_ylabel('Concentration (nM)', fontsize=12)
    ax_plaque.grid(True, alpha=0.3)
    if len(plaque_species) > 0:
        ax_plaque.legend(fontsize=10)
    
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

def create_plots(sol, c, model):
    """Create plots of the simulation results similar to QSP model"""
    if sol is None:
        print("No solution data available for plotting.")
        return
    
    # Create figures directory if it doesn't exist
    figures_dir = Path("generated/figures/no_dose")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get index mappings
        y_indexes = model.y_indexes
        
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
        ax2a.plot(years, ab40_oligomer_load, label='Oligomer Load', linewidth=2, color='C0')
        ax2a.set_xlabel('Time (years)', fontsize=12)
        ax2a.set_ylabel('Load (nM)', fontsize=12)
        ax2a.set_title('AB40 Oligomer Load', fontsize=14)
        ax2a.legend(fontsize=10)
        ax2a.grid(True, alpha=0.3)
        
        # Plot AB42 oligomer load
        ax2b.plot(years, ab42_oligomer_load, label='Oligomer Load', linewidth=2, color='C0')
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
        ax3a.plot(years, ab40_fibrils, label='Protofibril Load', linewidth=2, color='C1')
        ax3a.set_xlabel('Time (years)', fontsize=12)
        ax3a.set_ylabel('Load (nM)', fontsize=12)
        ax3a.set_title('AB40 Protofibril Load', fontsize=14)
        ax3a.legend(fontsize=10)
        ax3a.grid(True, alpha=0.3)
        
        # Plot AB42 protofibril load
        ax3b.plot(years, ab42_fibrils, label='Protofibril Load', linewidth=2, color='C1')
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
        ax4a.plot(years, sol.ys[:, y_indexes['AB40_Plaque_unbound']], 
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
        ax4b.plot(years, sol.ys[:, y_indexes['AB42_Plaque_unbound']], 
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
        
        print(f"Figures saved to {figures_dir}")
        
    except Exception as e:
        print(f"\nError creating plots: {e}")
        traceback.print_exc()

def print_model_info(sol, model):
    """Print information about the model to help with debugging
    
    Args:
        sol: Solution from the ODE solver
        model: The imported JAX model module
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


def main():
    """Main function to run the combined master model simulation"""
    parser = argparse.ArgumentParser(description="Run combined master model simulation")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                        default="gantenerumab", help="Drug type to simulate (affects parameters)")
    parser.add_argument("--years", type=float, default=10.0, help="Number of years to simulate")
    args = parser.parse_args()
    
    # Print summary of simulation settings
    print(f"\n=== COMBINED MASTER MODEL SIMULATION (NO DRUG DOSING) ===")
    print(f"Drug parameters: {args.drug.upper()}")
    print(f"Simulation years: {args.years}")
    print("=" * 40)
    
    # Run simulation
    simulation_time = args.years * 24 * 365  # years to hours
    sol, c, model = run_simulation(simulation_time, drug_type=args.drug)
    
    if sol is not None:
        # Print model information
        print_model_info(sol, model)
        
       
        # Create figures directory if it doesn't exist
        figures_dir = Path("generated/figures/no_dose")
        figures_dir.mkdir(parents=True, exist_ok=True)
        create_plots(sol, c, model)
        plot_individual_oligomers(sol, drug_type=args.drug, plots_dir=figures_dir)
        plot_fibrils_and_plaques(sol, drug_type=args.drug, plots_dir=figures_dir)
        # Save simulation data to file
        data = {'time': sol.ts, **{name: sol.ys[:, idx] for name, idx in model.y_indexes.items()}}
        pd.DataFrame(data).to_csv(f"generated/100_year_simulation_results_{args.drug}.csv", index=False)
    else:
        print("\nSimulation failed. Check error messages above.")

if __name__ == "__main__":
    main()