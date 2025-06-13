"""
Run script for the combined master model.
This script runs the combined model that includes both AB_Master_Model and Geerts_Master_Model components.
It first runs to steady state without antibody dosing, then uses those results as initial conditions
for a simulation with antibody dosing.
"""
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
        print(f"Loading parameters for {drug_type}...")
        params, params_with_units = load_parameters_from_csv(
            "parameters/PK_Geerts.csv",
            drug_type=drug_type,
        )
        print(f"Parameters loaded successfully. Total: {len(params)}")
        
        # Generate the combined master model
        print("Generating combined master model...")
        document = create_combined_master_model(
            params,
            params_with_units,
            drug_type=drug_type,
        )
        print("Combined master model created successfully.")
        
        # Save SBML file
        print(f"Saving SBML model to {xml_path}...")
        save_model(document, str(xml_path))
        print(f"SBML model successfully created at {xml_path}")
        
        # Verify file was written
        if xml_path.exists():
            file_size = xml_path.stat().st_size
            print(f"File written successfully. Size: {file_size} bytes")
        else:
            print("ERROR: File was not created!")
        
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

def run_simulation(time_hours=2040, drug_type="gantenerumab", steady_state=False, y0_ss=None):
    """Run the combined master model simulation
    
    Args:
        time_hours: Simulation duration in hours
        drug_type: Either "lecanemab" or "gantenerumab"
        steady_state: If True, run to steady state without dosing
        y0_ss: Initial state (optional, for continuing from steady state)
        
    Returns:
        Tuple of (solution, constants)
    """
    # Import the generated model
    sys.path.append(str(Path("generated/jax").absolute()))
    
    print(f"\n=== Running {drug_type.upper()} simulation for {time_hours} hours ===")

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

    print(f"Run Simulation Vcent: {c[c_indexes['Vcent']]}")
    print(f"Run Simulation Vp_brain: {c[c_indexes['Vp_brain']]}")
    
    # Create rate of change function
    rate_of_change = RateofSpeciesChange()
    assignment_rule = AssignmentRule()
    
    @jit
    def combined_ode_func(t, y, args):
        # This combines assignment rule and rate of change in one jitted function
        w, c = args
        w = assignment_rule(y, w, c, t)
        dy_dt = rate_of_change(y, t, w, c)
        return dy_dt
    
    # Create mutable copy of constants
    c_mutable = c.copy()
    
    # Determine drug-specific parameters
    is_lecanemab = drug_type.lower() == "lecanemab"
    
    if steady_state:
        # For steady state, set all dosing parameters to zero
        print("\nConfiguring steady state simulation (no drug dosing)...")
        c_mutable = c_mutable.at[c_indexes['MaxDosingTime']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['SC_DoseAmount']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['SC_NumDoses']].set(0.0)
        
        # Set initial conditions to zero for drug compartments
        current_y0 = y0.copy()
        if is_lecanemab:
            current_y0 = current_y0.at[y_indexes['PK_central']].set(0.0)
        else:
            current_y0 = current_y0.at[y_indexes['SubCut_absorption']].set(0.0)
        
        # For steady state, we don't need dosing times
        dosing_times = jnp.array([])
        
    else:
        # Set maximum dosing time
        max_dosing_time = min(13140, time_hours)  # Don't exceed simulation time
        c_mutable = c_mutable.at[c_indexes['MaxDosingTime']].set(max_dosing_time)
        
        if is_lecanemab:
            # FOR SINGLE DOSE: Just use a single time point at t=0
            dosing_times = jnp.array([0.0])  # Single dose at start
            
            # Set Lecanemab IV dosing parameters (10 mg/kg)
            c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(4930.6)  # nM
            c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(1.0)  # CHANGED: Set to 1 dose
            
            # Zero out SC dosing for Lecanemab
            c_mutable = c_mutable.at[c_indexes['SC_DoseAmount']].set(0.0)
            c_mutable = c_mutable.at[c_indexes['SC_NumDoses']].set(0.0)
        else:
            # FOR SINGLE DOSE: Just use a single time point at t=0
            dosing_times = jnp.array([0.0])  # Single dose at start
            
            # Set Gantenerumab SC dosing parameters (1200 mg)
            c_mutable = c_mutable.at[c_indexes['SC_DoseAmount']].set(2050.6)  # nM
            c_mutable = c_mutable.at[c_indexes['SC_NumDoses']].set(1.0)  # CHANGED: Set to 1 dose
            
            # Zero out IV dosing for Gantenerumab
            c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(0.0)
            c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(0.0)
        
        # Use steady state solution if provided
        current_y0 = y0_ss if y0_ss is not None else y0.copy()
    
    # Print current parameters
    print(f"\nSimulation parameters for {drug_type}:")
    print(f"Steady state: {steady_state}")
    print(f"Time: {time_hours} hours")
    print(f"MaxDosingTime: {c_mutable[c_indexes['MaxDosingTime']]}")
    print(f"IV_DoseAmount: {c_mutable[c_indexes['IV_DoseAmount']]} nM")
    print(f"SC_DoseAmount: {c_mutable[c_indexes['SC_DoseAmount']]} nM")
    print(f"IV_NumDoses: {c_mutable[c_indexes['IV_NumDoses']]}")
    print(f"SC_NumDoses: {c_mutable[c_indexes['SC_NumDoses']]}")
    
    # Simulation parameters
    t1 = float(time_hours)
    dt = 0.01  # 10x larger, let the adaptive stepper handle it
    n_steps = min(1000, int(time_hours/10))
    
    # Create diffrax solver
    term = ODETerm(combined_ode_func)
    solver = Tsit5()
    rtol = 1e-3
    atol = 1e-6
    print("Using Tsit5 solver")
    
    # Create saveat points
    saveat = SaveAt(ts=jnp.linspace(t0, t1, n_steps))
    
    # Create stepsize controller
    controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff=0.4, icoeff=0.3)
    
    print(f"\nSimulating for {time_hours} hours...")
    start_time = time.time()
    
    # Solve the ODE system
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
        stepsize_controller=controller,
        progress_meter=diffrax.TextProgressMeter()
    )
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return sol, c_mutable, dosing_times

def plot_brain_plasma_dynamics(sol, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot brain plasma dynamics"""
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
        c_indexes = jax_module.c_indexes
        y_indexes = jax_module.y_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV" if is_lecanemab else "300 mg subcutaneous"
    
    # Get brain plasma volume from constants
    brain_plasma_volume = c[c_indexes['Vp_brain']]
    
    # Create figure for Brain Plasma
    fig_brain_plasma = plt.figure(figsize=(10, 6))
    ax_brain_plasma = fig_brain_plasma.add_subplot(1, 1, 1)
    
    # For Lecanemab, use hours for x-axis, for Gantenerumab use days
    if is_lecanemab:
        x_values = sol.ts
        x_label = 'Time (hours)'
    else:
        x_values = sol.ts / 24.0
        x_label = 'Time after first dose (days)'
    
    # Plot Brain Plasma with bold line
    ax_brain_plasma.semilogy(x_values, sol.ys[:, y_indexes['PK_p_brain']]/brain_plasma_volume, 
                    color='black', linewidth=3, label='Brain Plasma (Master)')
    
    # Load experimental data if available
    data_path = Path(f"parameters/Geerts_{'Lec' if is_lecanemab else 'Gant'}_Data.csv")
    if data_path.exists():
        exp_data = pd.read_csv(data_path)
        ax_brain_plasma.scatter(exp_data['Time'], exp_data['Concentration'], 
                     color='red', s=100, marker='o', edgecolors='red', linewidths=2, 
                     facecolors='none', label='Experimental Data')
    
    # Set axis labels and title with larger font
    ax_brain_plasma.set_xlabel(x_label, fontsize=14)
    ax_brain_plasma.set_ylabel('Concentration (nM)', fontsize=14)
    ax_brain_plasma.set_title(f'{drug_name} Plasma PK: {dose_info}', fontsize=16, fontweight='bold')
    
    # Set y-axis limits based on drug
    if is_lecanemab:
        ax_brain_plasma.set_ylim(1, 10000)  # 10^0 to 10^4
    else:
        ax_brain_plasma.set_ylim(1, 1000)   # 10^0 to 10^3
    
    # Set x-axis limits
    if is_lecanemab:
        ax_brain_plasma.set_xlim(0, 700)  # 700 hours for lecanemab
    else:
        ax_brain_plasma.set_xlim(0, 85)   # 85 days for gantenerumab
    
    # Increase tick label size
    ax_brain_plasma.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid with light alpha
    ax_brain_plasma.grid(True, alpha=0.3)
    
    # Create custom legend with larger font
    ax_brain_plasma.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save Brain Plasma figure with high resolution
    fig_brain_plasma.savefig(plots_dir / f'{drug_name.lower()}_pk_brain_plasma_master.png', 
                    dpi=300, bbox_inches='tight')
    plt.close()

def plot_amyloid_dynamics(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot total oligomers, fibrils, and plaque dynamics"""
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Initialize arrays for total concentrations
    total_ab40_oligomers = np.zeros_like(sol.ts)
    total_ab42_oligomers = np.zeros_like(sol.ts)
    total_ab40_fibrils = np.zeros_like(sol.ts)
    total_ab42_fibrils = np.zeros_like(sol.ts)
    
    # Find and sum all AB40 oligomers
    ab40_oligomer_count = 0
    for i in range(2, 17):
        # Try different naming patterns that might be in the model
        possible_names = [
            f'AB40_Oligomer{i:02d}',
            f'AB40_Oligomer{i}',
            f'ABeta40_oligomer{i:02d}',
            f'ABeta40_oligomer{i}',
            f'AB40_oligomer{i:02d}',
            f'AB40_oligomer{i}'
        ]
        for name in possible_names:
            if name in y_indexes:
                total_ab40_oligomers += sol.ys[:, y_indexes[name]]
                ab40_oligomer_count += 1
                break
    
    # Find and sum all AB42 oligomers
    ab42_oligomer_count = 0
    for i in range(2, 17):
        # Try different naming patterns that might be in the model
        possible_names = [
            f'AB42_Oligomer{i:02d}',
            f'AB42_Oligomer{i}',
            f'ABeta42_oligomer{i:02d}',
            f'ABeta42_oligomer{i}',
            f'AB42_oligomer{i:02d}',
            f'AB42_oligomer{i}'
        ]
        for name in possible_names:
            if name in y_indexes:
                total_ab42_oligomers += sol.ys[:, y_indexes[name]]
                ab42_oligomer_count += 1
                break
                
    # Find and sum all AB40 fibrils
    ab40_fibril_count = 0
    for i in range(17, 25):  # Fibrils 17-24
        possible_names = [
            f'AB40_Fibril{i:02d}',
            f'AB40_Fibril{i}'
        ]
        for name in possible_names:
            if name in y_indexes:
                total_ab40_fibrils += sol.ys[:, y_indexes[name]]
                ab40_fibril_count += 1
                break
                
    # Find and sum all AB42 fibrils
    ab42_fibril_count = 0
    for i in range(17, 25):  # Fibrils 17-24
        possible_names = [
            f'AB42_Fibril{i:02d}',
            f'AB42_Fibril{i}'
        ]
        for name in possible_names:
            if name in y_indexes:
                total_ab42_fibrils += sol.ys[:, y_indexes[name]]
                ab42_fibril_count += 1
                break
    
    # Plot total oligomer and fibril concentrations
    ax.plot(sol.ts / 24.0, total_ab40_oligomers, 
                label=f'Total AB40 Oligomers (n={ab40_oligomer_count})', 
                linewidth=2, color='C0')
    ax.plot(sol.ts / 24.0, total_ab42_oligomers, 
                label=f'Total AB42 Oligomers (n={ab42_oligomer_count})', 
                linewidth=2, color='C1')
    ax.plot(sol.ts / 24.0, total_ab40_fibrils,
                label=f'Total AB40 Fibrils (n={ab40_fibril_count})',
                linewidth=2, color='C2')
    ax.plot(sol.ts / 24.0, total_ab42_fibrils,
                label=f'Total AB42 Fibrils (n={ab42_fibril_count})',
                linewidth=2, color='C3')
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Total Concentration (nM)', fontsize=12)
    ax.set_title(f'Total AB40 and AB42 Oligomers and Fibrils ({drug_type.capitalize()})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_type.lower()}_total_oligomer_fibril_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_microglia_dynamics(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot microglia high fraction and cell count over time"""
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
    
    # Create figure with two subplots (one for fraction, one for count)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot using days on x-axis for better readability
    x_values = sol.ts / 24.0  # Convert hours to days
    
    # Plot Microglia High Fraction
    if 'Microglia_Hi_Fract' in y_indexes:
        ax1.plot(x_values, sol.ys[:, y_indexes['Microglia_Hi_Fract']], 
                linewidth=2.5, color='blue', label='Microglia High Fraction')
        ax1.set_ylabel('Fraction', fontsize=14)
        ax1.set_title('Microglia High Fraction', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
    else:
        print("Warning: Microglia_Hi_Fract not found in model")
    
    # Plot Microglia Cell Count
    if 'Microglia_cell_count' in y_indexes:
        ax2.plot(x_values, sol.ys[:, y_indexes['Microglia_cell_count']], 
                linewidth=2.5, color='green', label='Microglia Cell Count')
        ax2.set_xlabel('Time (days)', fontsize=14)
        ax2.set_ylabel('Cell Count', fontsize=14)
        ax2.set_title('Microglia Cell Count', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
    else:
        print("Warning: Microglia_cell_count not found in model")
    
    plt.tight_layout()
    
    # Save figure using the provided directory
    fig.savefig(plots_dir / f'{drug_type.lower()}_microglia_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Plot using days on x-axis for better readability
    x_values = sol.ts / 24.0  # Convert hours to days
    
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
    ax3.set_xlabel('Time (days)', fontsize=12)
    ax3.set_ylabel('Concentration (nM)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if len(ab40_oligomers_bound) > 0:
        ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax4.set_title('Antibody-Bound AB42 Oligomers', fontsize=14)
    ax4.set_xlabel('Time (days)', fontsize=12)
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
    
    # Plot using days on x-axis for better readability
    x_values = sol.ts / 24.0  # Convert hours to days
    
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
    ax3.set_xlabel('Time (days)', fontsize=12)
    ax3.set_ylabel('Concentration (nM)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    if len(ab40_fibrils_bound) > 0:
        ax3.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax4.set_title('Antibody-Bound AB42 Fibrils', fontsize=14)
    ax4.set_xlabel('Time (days)', fontsize=12)
    ax4.set_ylabel('Concentration (nM)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    if len(ab42_fibrils_bound) > 0:
        ax4.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set titles and labels for plaque plot
    ax_plaque.set_title('Plaque Dynamics', fontsize=14)
    ax_plaque.set_xlabel('Time (days)', fontsize=12)
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

def plot_compartment_dynamics(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot brain, CSF, and peripheral compartment dynamics"""
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
    
    # Create figure with three subplots
    fig_combined = plt.figure(figsize=(20, 6))
    
    # Brain compartments subplot
    ax_brain = fig_combined.add_subplot(1, 3, 1)
    ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['PK_central']], label='Central', linewidth=2)
    ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['PK_p_brain']], label='Brain Plasma', linewidth=2)
    ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['Ab_t']], label='Brain ISF (Ab_t)', linewidth=2)
    ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['PK_BBB_unbound_brain']], label='BBB Unbound', linewidth=2)
    ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['PK_BBB_bound_brain']], label='BBB Bound', linewidth=2)
    
    # Add PVS/ARIA specific plots
    if 'C_Antibody_unbound_PVS' in y_indexes:
        ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['C_Antibody_unbound_PVS']], 
                         label='Antibody in PVS', linewidth=2, color='purple')
    
    if 'AB40_plaque_bound_PVS' in y_indexes:
        ax_brain.semilogy(sol.ts, sol.ys[:, y_indexes['AB40_plaque_bound_PVS']], 
                         label='AB40 Plaque Bound PVS', linewidth=2, color='orange')
    
    ax_brain.set_xlabel('Time (hours)', fontsize=12)
    ax_brain.set_ylabel('Concentration (nM)', fontsize=12)
    ax_brain.set_title('Brain Compartments', fontsize=14)
    ax_brain.legend(fontsize=10)
    ax_brain.grid(True, alpha=0.3)
    ax_brain.set_xlim(0, sol.ts[-1])
    
    # CSF compartments subplot
    ax_csf = fig_combined.add_subplot(1, 3, 2)
    ax_csf.semilogy(sol.ts, sol.ys[:, y_indexes['PK_LV_brain']], label='Lateral Ventricle', linewidth=2)
    ax_csf.semilogy(sol.ts, sol.ys[:, y_indexes['PK_TFV_brain']], label='Third/Fourth Ventricle', linewidth=2)
    ax_csf.semilogy(sol.ts, sol.ys[:, y_indexes['PK_CM_brain']], label='Cisterna Magna', linewidth=2)
    ax_csf.semilogy(sol.ts, sol.ys[:, y_indexes['PK_SAS_brain']], label='Subarachnoid Space', linewidth=2)
    ax_csf.semilogy(sol.ts, sol.ys[:, y_indexes['PK_BCSFB_unbound_brain']], label='BCSFB Unbound', linewidth=2, color='red')
    ax_csf.semilogy(sol.ts, sol.ys[:, y_indexes['PK_BCSFB_bound_brain']], label='BCSFB Bound', linewidth=2, color='red', linestyle='--')
    
    ax_csf.set_xlabel('Time (hours)', fontsize=12)
    ax_csf.set_ylabel('Concentration (nM)', fontsize=12)
    ax_csf.set_title('CSF Compartments', fontsize=14)
    ax_csf.legend(fontsize=10)
    ax_csf.grid(True, alpha=0.3)
    ax_csf.set_xlim(0, sol.ts[-1])
    
    # Peripheral compartments subplot
    ax_per = fig_combined.add_subplot(1, 3, 3)
    ax_per.semilogy(sol.ts, sol.ys[:, y_indexes['PK_central']], label='Central', linewidth=2)
    ax_per.semilogy(sol.ts, sol.ys[:, y_indexes['PK_per']], label='Peripheral', linewidth=2)
    
    # Only plot SubCut for gantenerumab
    if drug_type.lower() == "gantenerumab":
        ax_per.semilogy(sol.ts, sol.ys[:, y_indexes['SubCut_absorption']], label='SubCut', linewidth=2)
    
    ax_per.set_xlabel('Time (hours)', fontsize=12)
    ax_per.set_ylabel('Concentration (nM)', fontsize=12)
    ax_per.set_title('Peripheral Compartments', fontsize=14)
    ax_per.legend(fontsize=10)
    ax_per.grid(True, alpha=0.3)
    ax_per.set_xlim(0, sol.ts[-1])
    
    plt.tight_layout()
    
    # Save figure
    fig_combined.savefig(plots_dir / f'{drug_type.lower()}_compartment_dynamics.png', 
                        dpi=300, bbox_inches='tight')
    plt.close(fig_combined)

def main():
    """Main function to run the combined master model simulation"""
    parser = argparse.ArgumentParser(description="Run combined master model simulation")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                        default="gantenerumab", help="Drug type to simulate")
    args = parser.parse_args()
    
    # Print summary of simulation settings
    print(f"\n=== COMBINED MASTER MODEL SIMULATION ===")
    print(f"Drug: {args.drug.upper()}")
    print("=" * 40)
    
    # Run simulation until steady state
    print("Running to steady state (no drug dose)...")
    sol_ss, c_ss, _ = run_simulation(
        1000,  # Run for 2000 hours to reach steady state
        drug_type=args.drug, 
        steady_state=True
    )
    
    # Generate plots for steady state
    print("\nGenerating steady state plots...")
    
    # Create steady state directory
    steady_state_dir = Path("generated/figures/steady_state")
    steady_state_dir.mkdir(parents=True, exist_ok=True)
    
    # Original plots for steady state with correct directory
    plot_brain_plasma_dynamics(sol_ss, c_ss, [], drug_type=f"{args.drug}_steady_state", plots_dir=steady_state_dir)
    plot_amyloid_dynamics(sol_ss, drug_type=f"{args.drug}_steady_state", plots_dir=steady_state_dir)
    plot_microglia_dynamics(sol_ss, drug_type=f"{args.drug}_steady_state", plots_dir=steady_state_dir)
    plot_individual_oligomers(sol_ss, drug_type=f"{args.drug}_steady_state", plots_dir=steady_state_dir)
    plot_fibrils_and_plaques(sol_ss, drug_type=f"{args.drug}_steady_state", plots_dir=steady_state_dir)
    plot_compartment_dynamics(sol_ss, drug_type=f"{args.drug}_steady_state", plots_dir=steady_state_dir)
    
    # Run simulation with steady state initial conditions
    print("\nRunning with drug dose from steady state...")
    if args.drug.lower() == "lecanemab":
        simulation_time = 700  # For lecanemab, run for 700 hours
    else:
        simulation_time = 2040  # For gantenerumab, run for 2040 hours (85 days)
    
    sol, c, dosing_times = run_simulation(
        simulation_time,
        drug_type=args.drug, 
        steady_state=False, 
        y0_ss=sol_ss.ys[-1]
    )
    
    # Create drug simulation directory
    drug_sim_dir = Path("generated/figures/drug_simulation")
    drug_sim_dir.mkdir(parents=True, exist_ok=True)
    
    # Plots for drug simulation with correct directory
    plot_brain_plasma_dynamics(sol, c, dosing_times, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_amyloid_dynamics(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_microglia_dynamics(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_individual_oligomers(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_fibrils_and_plaques(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_compartment_dynamics(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    
    print("\nAll plots saved in generated/figures/steady_state and generated/figures/drug_simulation")

if __name__ == "__main__":
    main() 