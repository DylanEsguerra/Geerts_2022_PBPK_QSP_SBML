"""
Run script for the combined master model.
This script runs the combined model that includes both AB_Master_Model and Geerts_Master_Model components.
It uses the saved data from run_no_dose_combined_master_model.py as initial conditions
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

# Add the project root to Python path
root_dir = Path(__file__).parents[1]  # Go up 1 level to reach models directory
sys.path.append(str(root_dir))

# Import the master model
from Modules.Combined_Master_Model import create_combined_master_model, load_parameters_from_csv, save_model

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

def run_simulation(time_hours=26280, drug_type="gantenerumab", steady_state=False, y0_ss=None, age=0.0):
    """Run the combined master model simulation
    
    Args:
        time_hours: Simulation duration in hours (default 26280 = 3 years)
        drug_type: Either "lecanemab" or "gantenerumab"
        steady_state: If True, run to steady state without dosing
        y0_ss: Initial state (optional, for continuing from steady state)
        age: Initial age in years (default 0.0)
        
    Returns:
        Tuple of (solution, constants)
    """
    # Import the generated model
    sys.path.append(str(Path("generated/jax").absolute()))
    
    print(f"\n=== Running {drug_type.upper()} simulation for {time_hours} hours ===")
    print(f"Initial age: {age} years")

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
    
    # Set the initial age parameter
    if 'age_init' in c_indexes:
        c_mutable = c_mutable.at[c_indexes['age_init']].set(float(age))
        print(f"Setting initial age to: {age} years")
    
    # Determine drug-specific parameters
    is_lecanemab = drug_type.lower() == "lecanemab"
    
    if steady_state:
        # For steady state, set all dosing parameters to zero
        print("\nConfiguring steady state simulation (no drug dosing)...")
        c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['SC_DoseAmount']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['SC_NumDoses']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['IV_DoseInterval']].set(0.0)
        c_mutable = c_mutable.at[c_indexes['SC_DoseInterval']].set(0.0)
        
        # Set initial conditions to zero for drug compartments
        current_y0 = y0.copy()
        if is_lecanemab:
            current_y0 = current_y0.at[y_indexes['PK_central']].set(0.0)
        else:
            current_y0 = current_y0.at[y_indexes['SubCut_absorption']].set(0.0)
        
        # For steady state, we don't need dosing times
        dosing_times = jnp.array([])
        dosing_interval = 0
        num_doses = 0
        
    else:
        # Set maximum dosing time
        max_dosing_time = min(1.5*365*24, time_hours)  # stop dosing after 1.5 years
        c_mutable = c_mutable.at[c_indexes['MaxDosingTime']].set(max_dosing_time)
        
        # Determine drug-specific parameters
        is_lecanemab = drug_type.lower() == "lecanemab"
        
        if is_lecanemab:
            # Lecanemab: every 2 weeks (336 hours)
            dosing_interval = 336.0  # 2 weeks in hours
            # Ensure at least one dose
            num_doses = max(1, int(jnp.ceil(max_dosing_time / dosing_interval)))
            
            # Generate all dosing times
            dosing_times = jnp.arange(0, min(max_dosing_time, num_doses * dosing_interval), dosing_interval)
            
            # Set Lecanemab IV dosing parameters (10 mg/kg)
            c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(4930.6)  # nM
            c_mutable = c_mutable.at[c_indexes['IV_DoseInterval']].set(dosing_interval)
            c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(float(num_doses))
            
        else:
            # Gantenerumab: every 4 weeks (672 hours)
            dosing_interval = 672.0  # 4 weeks in hours
            num_doses = max(1, int(jnp.ceil(max_dosing_time / dosing_interval)))
            
            # Generate all dosing times
            dosing_times = jnp.arange(0, min(max_dosing_time, num_doses * dosing_interval), dosing_interval)
            
            # Set Gantenerumab SC dosing parameters (1200 mg)
            c_mutable = c_mutable.at[c_indexes['SC_DoseAmount']].set(2050.6 * 4)  # nM
            c_mutable = c_mutable.at[c_indexes['SC_DoseInterval']].set(dosing_interval)
            c_mutable = c_mutable.at[c_indexes['SC_NumDoses']].set(float(num_doses))
            
            # Zero out IV dosing for Gantenerumab
            c_mutable = c_mutable.at[c_indexes['IV_DoseAmount']].set(0.0)
            c_mutable = c_mutable.at[c_indexes['IV_DoseInterval']].set(0.0)
            c_mutable = c_mutable.at[c_indexes['IV_NumDoses']].set(0.0)
            print(f"Dosing time: {max_dosing_time} hours ({max_dosing_time/24/365:.1f} years)")
            print(f"Number of doses: {len(dosing_times)}")
            print(f"Dosing interval: {dosing_interval} hours ({dosing_interval/24:.1f} days)")

        # Use steady state solution if provided
        current_y0 = y0_ss if y0_ss is not None else y0.copy()
    
    # Print current parameters
    print(f"\nSimulation parameters for {drug_type}:")
    print(f"Steady state: {steady_state}")
    print(f"Total simulation time: {time_hours} hours ({time_hours/24/365:.1f} years)")
    print(f"IV_DoseAmount: {c_mutable[c_indexes['IV_DoseAmount']]} nM")
    print(f"SC_DoseAmount: {c_mutable[c_indexes['SC_DoseAmount']]} nM")
    print(f"IV_NumDoses: {c_mutable[c_indexes['IV_NumDoses']]}")
    print(f"SC_NumDoses: {c_mutable[c_indexes['SC_NumDoses']]}")
    
    # Simulation parameters
    t1 = float(time_hours)
    dt = 0.0002  # Increased initial step size
    n_steps = min(5000, int(time_hours/50))  # Significantly reduce saved points
    
    # Create diffrax solver
    term = ODETerm(combined_ode_func)
    solver = diffrax.Tsit5()
    rtol = 1e-10
    atol = 1e-10
    print("Using Tsit5 solver")
    
    if not steady_state:
        # Create more detailed steps around doses, with a scale based on dosing interval
        step_ts = []
        for dose_time in dosing_times:
            if 0 < dose_time < t1:
                # Scale offsets proportionally to dosing interval
                step_ts.extend([
                    float(dose_time - 0.1),  
                    float(dose_time + 0.1),
                ])
        step_ts = jnp.array(step_ts)
        
        # Optimize saveat by reducing number of points while maintaining resolution around doses
        base_points = jnp.linspace(t0, t1, n_steps // 2)  # Reduced from n_steps
        save_ts = jnp.sort(jnp.concatenate([
            base_points,
            dosing_times,
            step_ts
        ]))
        saveat = diffrax.SaveAt(ts=save_ts)
        
        # Create stepsize controller with proper handling of discontinuities
        pid_controller = diffrax.PIDController(rtol=rtol, atol=atol, pcoeff=0.4, icoeff=0.3)
        
        # Only use ClipStepSizeController if we have step times
        if len(step_ts) > 0:
            controller = diffrax.ClipStepSizeController(
                controller=pid_controller,
                step_ts=step_ts)
        else:
            controller = pid_controller
    else:
        # For steady state, use simple uniform time points
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps))
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
    
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
        max_steps=int(1e12),
        stepsize_controller=controller,
        progress_meter=diffrax.TextProgressMeter()
    )
    
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # If this is a steady state simulation, save the full solution to CSV
    if steady_state:
        # Create a DataFrame with time and all species values
        data = {"time": sol.ts}
        for species_name, idx in y_indexes.items():
            data[species_name] = sol.ys[:, idx]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        output_dir = Path("generated/steady_state")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_file = output_dir / f"steady_state_solution_{drug_type.lower()}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nFull steady state solution saved to {output_file}")
    
    return sol, c_mutable, dosing_times

def _setup_year_axis(ax, x_data):
    """Helper function to set up year-based x-axis with 0.5 year marks
    
    Args:
        ax: matplotlib axis object
        x_data: array of x-values in years to determine appropriate range
    """
    # Get the data range
    x_max = np.max(x_data)
    
    # Set x-axis limits
    ax.set_xlim(0, x_max)
    ax.set_xlabel('Time (years)', fontsize=14)
    
    # Calculate appropriate tick spacing
    if x_max <= 0.5:
        tick_spacing = 0.1  # For very short simulations
    elif x_max <= 1:
        tick_spacing = 0.2  # For simulations up to 1 year
    else:
        tick_spacing = 0.5  # For longer simulations
    
    # Set x-ticks with calculated spacing
    ax.set_xticks(np.arange(0, x_max + tick_spacing, tick_spacing))
    
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

def plot_Ab_t_dynamics(sol, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot Ab_t (ISF) dynamics and total bound amyloid over time"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Calculate total bound amyloid in ISF
    isf_bound = np.zeros_like(sol.ts)
    isf_bound += sol.ys[:, y_indexes['AB40_monomer_antibody_bound']]
    isf_bound += sol.ys[:, y_indexes['AB42_monomer_antibody_bound']]
    
    # Sum bound oligomers in ISF
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Oligomer{i:02d}_Antibody_bound"
            if species_name in y_indexes:
                isf_bound += sol.ys[:, y_indexes[species_name]] 
    
    # Sum bound fibrils in ISF
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Fibril{i:02d}_Antibody_bound"
            if species_name in y_indexes:
                isf_bound += sol.ys[:, y_indexes[species_name]] 
    
    # Add bound plaques in ISF
    for species_name in ["AB40_Plaque_Antibody_bound", "AB42_Plaque_Antibody_bound"]:
        if species_name in y_indexes:
            isf_bound += sol.ys[:, y_indexes[species_name]] 
    
    # Calculate total bound amyloid in PVS
    pvs_bound = np.zeros_like(sol.ts)
    for ab_type in ['AB40', 'AB42']:
        # Add bound monomers
        species_name = f'{ab_type}_Monomer_PVS_bound'
        if species_name in y_indexes:
            pvs_bound += sol.ys[:, y_indexes[species_name]]
        
        # Add bound oligomers
        for i in range(2, 17):
            species_name = f'{ab_type}_Oligomer{i:02d}_PVS_bound'
            if species_name in y_indexes:
                pvs_bound += sol.ys[:, y_indexes[species_name]]
        
        # Add bound fibrils
        for i in range(17, 25):
            species_name = f'{ab_type}_Fibril{i:02d}_PVS_bound'
            if species_name in y_indexes:
                pvs_bound += sol.ys[:, y_indexes[species_name]]
        
        # Add bound plaques
        species_name = f'{ab_type}_Plaque_bound_PVS'
        if species_name in y_indexes:
            pvs_bound += sol.ys[:, y_indexes[species_name]]
    
    # Calculate total antibody concentrations
    isf_total = sol.ys[:, y_indexes['Ab_t']] + isf_bound
    pvs_total = sol.ys[:, y_indexes['C_Antibody_unbound_PVS']] + pvs_bound
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Plot ISF unbound antibody (top left)
    axs[0, 0].plot(x_values, sol.ys[:, y_indexes['Ab_t']] / c[c_indexes['VIS_brain']], 
                   label='Unbound', linewidth=2, color='blue')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('Brain ISF Unbound Antibody', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)
    
    # Plot PVS unbound antibody (top right)
    axs[0, 1].plot(x_values, sol.ys[:, y_indexes['C_Antibody_unbound_PVS']] / c[c_indexes['V_PVS']], 
                   label='Unbound', linewidth=2, color='green')
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('PVS Unbound Antibody', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)
    
    # Plot ISF total antibody (bottom left)
    axs[1, 0].plot(x_values, isf_total / c[c_indexes['VIS_brain']],
                   label='Total', linewidth=2, color='red')
    axs[1, 0].plot(x_values, sol.ys[:, y_indexes['Ab_t']] / c[c_indexes['VIS_brain']],
                   label='Unbound', linewidth=2, color='blue')
    axs[1, 0].plot(x_values, isf_bound / c[c_indexes['VIS_brain']],
                   label='Bound', linewidth=2, color='green')
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('Brain ISF Total Antibody', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)
    
    # Plot PVS total antibody (bottom right)
    axs[1, 1].plot(x_values, pvs_total / c[c_indexes['V_PVS']],
                   label='Total', linewidth=2, color='red')
    axs[1, 1].plot(x_values, sol.ys[:, y_indexes['C_Antibody_unbound_PVS']] / c[c_indexes['V_PVS']],
                   label='Unbound', linewidth=2, color='green')
    axs[1, 1].plot(x_values, pvs_bound / c[c_indexes['V_PVS']],
                   label='Bound', linewidth=2, color='blue')
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('PVS Total Antibody', fontsize=14)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=10)
    
    # Update x-axis and add markers for all subplots
    for ax in axs.flat:
        _setup_year_axis(ax, x_values)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add dosing markers
        for dose_time in dosing_times:
            ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
        
        # Add end of dosing line
        ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
                  label='End of Dosing (1.5 years)')
    
    # Add overall title
    plt.suptitle(f'{drug_name} Antibody Dynamics: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_ab_t_and_bound_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_amyloid_dynamics(sol, drug_type="gantenerumab", plots_dir=None):
    """Plot total AB40 and AB42 for oligomers, fibrils, and plaques in separate subplots"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        y_indexes = jax_module.y_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Create figure with 3x2 subplots (AB40 left column, AB42 right column)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18), sharex=True)
    
    # Initialize arrays for total concentrations
    total_ab40_oligomers = np.zeros_like(sol.ts)
    total_ab42_oligomers = np.zeros_like(sol.ts)
    total_ab40_fibrils = np.zeros_like(sol.ts)
    total_ab42_fibrils = np.zeros_like(sol.ts)
    total_ab40_plaques = np.zeros_like(sol.ts)
    total_ab42_plaques = np.zeros_like(sol.ts)
    

    ab40_oligomer_count = 0
    for i in range(2, 17):
        species_name = f'AB40_Oligomer{i:02d}'
        if species_name in y_indexes:
            total_ab40_oligomers += sol.ys[:, y_indexes[species_name]]
            ab40_oligomer_count += 1
        
    
    # Sum all AB42 oligomers (both free and bound)
    ab42_oligomer_count = 0
    for i in range(2, 17):
        species_name = f'AB42_Oligomer{i:02d}'
        if species_name in y_indexes:
            total_ab42_oligomers += sol.ys[:, y_indexes[species_name]]
            ab42_oligomer_count += 1
       
    
    # Sum all AB40 fibrils (both free and bound)
    ab40_fibril_count = 0
    for i in range(17, 25):
        species_name = f'AB40_Fibril{i:02d}'
        if species_name in y_indexes:
            total_ab40_fibrils += sol.ys[:, y_indexes[species_name]]
            ab40_fibril_count += 1
        
    
    # Sum all AB42 fibrils (both free and bound)
    ab42_fibril_count = 0
    for i in range(17, 25):
        species_name = f'AB42_Fibril{i:02d}'
        if species_name in y_indexes:
            total_ab42_fibrils += sol.ys[:, y_indexes[species_name]]
            ab42_fibril_count += 1
       
    
    # Sum AB40 plaques (free and bound)
    if "AB40_Plaque_unbound" in y_indexes:
        total_ab40_plaques += sol.ys[:, y_indexes["AB40_Plaque_unbound"]]
    if "AB40_Plaque_Antibody_bound" in y_indexes:
        total_ab40_plaques += sol.ys[:, y_indexes["AB40_Plaque_Antibody_bound"]]
    
    # Sum AB42 plaques (free and bound)
    if "AB42_Plaque_unbound" in y_indexes:
        total_ab42_plaques += sol.ys[:, y_indexes["AB42_Plaque_unbound"]]
    if "AB42_Plaque_Antibody_bound" in y_indexes:
        total_ab42_plaques += sol.ys[:, y_indexes["AB42_Plaque_Antibody_bound"]]
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Plot AB40 oligomers (top left)
    ax1.plot(x_values, total_ab40_oligomers, 
                 label=f'Total (n={ab40_oligomer_count})', 
                 linewidth=2, color='blue')
    ax1.set_ylabel('Concentration (nM)', fontsize=12)
    ax1.set_title('AB40 Oligomers', fontsize=14)
    
    # Plot AB42 oligomers (top right)
    ax2.plot(x_values, total_ab42_oligomers, 
                 label=f'Total (n={ab42_oligomer_count})', 
                 linewidth=2, color='red')
    ax2.set_title('AB42 Oligomers', fontsize=14)
    
    # Plot AB40 fibrils (middle left)
    ax3.plot(x_values, total_ab40_fibrils,
                 label=f'Total (n={ab40_fibril_count})',
                 linewidth=2, color='blue')
    ax3.set_ylabel('Concentration (nM)', fontsize=12)
    ax3.set_title('AB40 Fibrils', fontsize=14)
    
    # Plot AB42 fibrils (middle right)
    ax4.plot(x_values, total_ab42_fibrils,
                 label=f'Total (n={ab42_fibril_count})',
                 linewidth=2, color='red')
    ax4.set_title('AB42 Fibrils', fontsize=14)
    
    # Plot AB40 plaques (bottom left)
    ax5.plot(x_values, total_ab40_plaques,
                 label='Total',
                 linewidth=2, color='blue')
    ax5.set_ylabel('Concentration (nM)', fontsize=12)
    ax5.set_xlabel('Time (years)', fontsize=12)
    ax5.set_title('AB40 Plaques', fontsize=14)
    
    # Plot AB42 plaques (bottom right)
    ax6.plot(x_values, total_ab42_plaques,
                 label='Total',
                 linewidth=2, color='red')
    ax6.set_xlabel('Time (years)', fontsize=12)
    ax6.set_title('AB42 Plaques', fontsize=14)
    
    # Set x-axis limits and formatting for all subplots
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        _setup_year_axis(ax, x_values)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
    
    # Add overall title
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    plt.suptitle(f'{drug_name} Amyloid Dynamics: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_type.lower()}_amyloid_dynamics.png', 
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
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
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
        ax2.set_xlabel('Time (years)', fontsize=14)
        ax2.set_ylabel('Cell Count', fontsize=14)
        ax2.set_title('Microglia Cell Count', fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
    else:
        print("Warning: Microglia_cell_count not found in model")
    
    # Update x-axis
    _setup_year_axis(ax2, x_values)  # Only need to set for bottom plot since they share x
    
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
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


def plot_bbb_bcsfb_subplots(sol, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot BBB and BCSFB bound/unbound concentrations in subplots"""
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
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Plot BBB bound concentration (top left)
    axs[0, 0].plot(x_values, sol.ys[:, y_indexes['PK_BBB_bound_brain']]/c[c_indexes['VBBB_brain']], 
                   color='blue', linewidth=2, label='BBB Bound')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('BBB Bound', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot BBB unbound concentration (top right)
    axs[0, 1].plot(x_values, sol.ys[:, y_indexes['PK_BBB_unbound_brain']]/c[c_indexes['VBBB_brain']], 
                   color='green', linewidth=2, label='BBB Unbound')
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('BBB Unbound', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot BCSFB bound concentration (bottom left)
    axs[1, 0].plot(x_values, sol.ys[:, y_indexes['PK_BCSFB_bound_brain']]/c[c_indexes['V_BCSFB_brain']], 
                   color='purple', linewidth=2, label='BCSFB Bound')
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('BCSFB Bound', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot BCSFB unbound concentration (bottom right)
    axs[1, 1].plot(x_values, sol.ys[:, y_indexes['PK_BCSFB_unbound_brain']]/c[c_indexes['V_BCSFB_brain']], 
                   color='orange', linewidth=2, label='BCSFB Unbound')
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('BCSFB Unbound', fontsize=14)
    axs[1, 1].grid(True, alpha=0.3)
    
    # Set x-axis limits and add markers for all subplots
    for ax in axs.flat:
        _setup_year_axis(ax, x_values)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add dosing markers
        for dose_time in dosing_times:
            ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.2)
        
        # Add end of dosing line
        ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'{drug_name} BBB and BCSFB Concentrations: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_bbb_bcsfb_subplots.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_csf_subplots(sol, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot CSF compartment concentrations in subplots, including total CSF concentration"""
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
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x3 subplots (added one for total CSF)
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharex=True)
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Calculate total bound amyloid in CSF compartments
    bound_amyloid = np.zeros_like(sol.ts)
    for compartment in ['LV', 'TFV', 'CM', 'SAS']:
        for ab_type in ['AB40', 'AB42']:
            species_name = f'{ab_type}Mb_{compartment}'
            if species_name in y_indexes:
                bound_amyloid += sol.ys[:, y_indexes[species_name]]
    
    # Calculate total antibody mass in all CSF compartments
    total_antibody_mass = (
        sol.ys[:, y_indexes['PK_LV_brain']] +
        sol.ys[:, y_indexes['PK_TFV_brain']] +
        sol.ys[:, y_indexes['PK_CM_brain']] +
        sol.ys[:, y_indexes['PK_SAS_brain']]
    )
    
    # Calculate total CSF volume
    total_csf_volume = (
        c[c_indexes['V_LV_brain']] +
        c[c_indexes['V_TFV_brain']] +
        c[c_indexes['V_CM_brain']] +
        c[c_indexes['V_SAS_brain']]
    )
    
    # Calculate average concentrations (mass/volume)
    total_antibody = total_antibody_mass / total_csf_volume
    total_bound = bound_amyloid / total_csf_volume
    total_csf = total_antibody + total_bound
    
    # Plot LV concentration
    axs[0, 0].plot(x_values, sol.ys[:, y_indexes['PK_LV_brain']] / c[c_indexes['V_LV_brain']], 
                   color='blue', linewidth=2, label='LV')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('Lateral Ventricles (LV)', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot TFV concentration
    axs[0, 1].plot(x_values, sol.ys[:, y_indexes['PK_TFV_brain']] / c[c_indexes['V_TFV_brain']], 
                   color='green', linewidth=2, label='TFV')
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('Third & Fourth Ventricles (TFV)', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot Total CSF concentration (including bound amyloid)
    axs[0, 2].plot(x_values, total_csf, 
                   color='red', linewidth=2, label='Total CSF (Incl. Bound)')
    axs[0, 2].plot(x_values, total_antibody, 
                   color='blue', linewidth=2, label='Free Antibody')
    axs[0, 2].plot(x_values, total_bound, 
                   color='green', linewidth=2, label='Bound Amyloid')
    axs[0, 2].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 2].set_title('Total CSF Concentration', fontsize=14)
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].legend(fontsize=10)
    
    # Plot CM concentration
    axs[1, 0].plot(x_values, sol.ys[:, y_indexes['PK_CM_brain']] / c[c_indexes['V_CM_brain']], 
                   color='purple', linewidth=2, label='CM')
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('Cisterna Magna (CM)', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot SAS concentration
    axs[1, 1].plot(x_values, sol.ys[:, y_indexes['PK_SAS_brain']] / c[c_indexes['V_SAS_brain']], 
                   color='orange', linewidth=2, label='SAS')
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('Subarachnoid Space (SAS)', fontsize=14)
    axs[1, 1].grid(True, alpha=0.3)
    
    # Plot BCSFB concentrations
    axs[1, 2].plot(x_values, sol.ys[:, y_indexes['PK_BCSFB_bound_brain']] / c[c_indexes['V_BCSFB_brain']], 
                   color='brown', linewidth=2, label='BCSFB Bound')
    axs[1, 2].plot(x_values, sol.ys[:, y_indexes['PK_BCSFB_unbound_brain']] / c[c_indexes['V_BCSFB_brain']] , 
                   color='pink', linewidth=2, label='BCSFB Unbound')
    axs[1, 2].set_xlabel('Time (years)', fontsize=12)
    axs[1, 2].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 2].set_title('BCSFB Concentrations', fontsize=14)
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].legend(fontsize=10)
    
    # Set x-axis limits and add markers for all subplots
    for ax in axs.flat:
        _setup_year_axis(ax, x_values)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add dosing markers
        for dose_time in dosing_times:
            ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.2)
        
        # Add end of dosing line
        ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'{drug_name} CSF Compartment Concentrations: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_csf_subplots.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate figure for total CSF concentration
    fig_total = plt.figure(figsize=(12, 8))
    ax_total = fig_total.add_subplot(1, 1, 1)
    
    # Plot total CSF concentration components
    ax_total.plot(x_values, total_csf, 
                 color='red', linewidth=3, label='Total CSF (Incl. Bound)')
    ax_total.plot(x_values, total_antibody, 
                 color='blue', linewidth=2, label='Free Antibody')
    ax_total.plot(x_values, total_bound, 
                 color='green', linewidth=2, label='Bound Amyloid')
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax_total.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax_total.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
                     label='End of Dosing (1.5 years)')
    
    # Set axis labels and title
    ax_total.set_xlabel('Time (years)', fontsize=14)
    ax_total.set_ylabel('Concentration (nM)', fontsize=14)
    ax_total.set_title(f'{drug_name} Total CSF Concentration: {dose_info}',
                      fontsize=16, fontweight='bold')
    
    # Set x-axis to show full 3 years
    ax_total.set_xlim(0, 3)
    
    # Increase tick label size
    ax_total.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax_total.grid(True, alpha=0.3)
    
    # Add legend
    ax_total.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save total CSF figure
    fig_total.savefig(plots_dir / f'{drug_name.lower()}_total_csf.png',
                    dpi=300, bbox_inches='tight')
    plt.close()

def plot_ab_bound_concentrations(sol, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot AB40Mb and AB42Mb concentrations in different compartments"""
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
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    
    # Plot using years on x-axis
    x_values = sol.ts / 24.0 / 365.0  # Convert hours to years
    
    # Define compartments and their colors
    compartments = {
        'Central': ('AB40Mb_Central', 'AB42Mb_Central', 'Vcent'),
        'Brain_Plasma': ('AB40Mb_Brain_Plasma', 'AB42Mb_Brain_Plasma', 'Vp_brain'),
        'BCSFB': ('AB40Mb_BCSFB_Bound', 'AB42Mb_BCSFB_Bound', 'V_BCSFB_brain'),
        'BBB': ('AB40Mb_BBB_Bound', 'AB42Mb_BBB_Bound', 'VBBB_brain')
    }
    
    colors = ['blue', 'green', 'purple', 'orange']
    
    # Plot AB40Mb concentrations (top row)
    for (label, (ab40_species, _, vol_key)), color in zip(compartments.items(), colors):
        axs[0, 0].plot(x_values, sol.ys[:, y_indexes[ab40_species]] / c[c_indexes[vol_key]], 
                      color=color, linewidth=2, label=label)
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('AB40 Bound (AB40Mb)', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)
    
    # Plot AB42Mb concentrations (top right)
    for (label, (_, ab42_species, vol_key)), color in zip(compartments.items(), colors):
        axs[0, 1].plot(x_values, sol.ys[:, y_indexes[ab42_species]] / c[c_indexes[vol_key]], 
                      color=color, linewidth=2, label=label)
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('AB42 Bound (AB42Mb)', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)
    
    # Plot CSF compartments for AB40Mb (bottom left)
    csf_compartments = {
        'LV': ('AB40Mb_LV', 'V_LV_brain'),
        'TFV': ('AB40Mb_TFV', 'V_TFV_brain'),
        'CM': ('AB40Mb_CM', 'V_CM_brain'),
        'SAS': ('AB40Mb_SAS', 'V_SAS_brain')
    }
    
    for (label, (species, vol_key)), color in zip(csf_compartments.items(), colors):
        axs[1, 0].plot(x_values, sol.ys[:, y_indexes[species]] / c[c_indexes[vol_key]], 
                      color=color, linewidth=2, label=label)
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('AB40 Bound in CSF Compartments', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)
    
    # Plot CSF compartments for AB42Mb (bottom right)
    csf_compartments = {
        'LV': ('AB42Mb_LV', 'V_LV_brain'),
        'TFV': ('AB42Mb_TFV', 'V_TFV_brain'),
        'CM': ('AB42Mb_CM', 'V_CM_brain'),
        'SAS': ('AB42Mb_SAS', 'V_SAS_brain')
    }
    
    for (label, (species, vol_key)), color in zip(csf_compartments.items(), colors):
        axs[1, 1].plot(x_values, sol.ys[:, y_indexes[species]] / c[c_indexes[vol_key]], 
                      color=color, linewidth=2, label=label)
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('AB42 Bound in CSF Compartments', fontsize=14)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=10)
    
    # Set x-axis limits and add markers for all subplots
    for ax in axs.flat:
        _setup_year_axis(ax, x_values)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add dosing markers
        for dose_time in dosing_times:
            ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.2)
        
        # Add end of dosing line
        ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'{drug_name} Bound Amyloid Concentrations: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_bound_amyloid_concentrations.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def load_no_dose_data(drug_type="gantenerumab"):
    """Load the saved data from run_no_dose_combined_master_model.py
    
    Args:
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        
    Returns:
        The final state from the no-dose simulation as a numpy array
    """
    # Load the saved data
    data_path = Path(f"generated/simulation_results_{drug_type}.csv")
    if not data_path.exists():
        print(f"Error: No saved data found at {data_path}")
        print("Please run run_no_dose_combined_master_model.py first")
        sys.exit(1)
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Get the final state
    final_state = jnp.array(df.iloc[-1].values[1:])  # Skip the time column
    
    return final_state

def main():
    """Main function to run the combined master model simulation"""
    parser = argparse.ArgumentParser(description="Run combined master model simulation")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                        default="gantenerumab", help="Drug type to simulate")
    parser.add_argument("--initial_age", type=float, default=0.0,
                        help="Initial age in years for the steady state simulation (default: 0.0)")
    args = parser.parse_args()
    
    # Print summary of simulation settings
    print(f"\n=== COMBINED MASTER MODEL SIMULATION ===")
    print(f"Drug: {args.drug.upper()}")
    print(f"Initial age for steady state: {args.initial_age} years")
    print("=" * 40)
    
    # Define the steady state simulation duration (in hours)
    steady_state_duration = 1000 #20*365*24  # 70 years in hours
    
    # Run simulation until steady state with specified initial age
    print("Running to steady state (no drug dose)...")
    sol_ss, c_ss, _ = run_simulation(
        steady_state_duration,  # Run for 70 years
        drug_type=args.drug, 
        steady_state=True,
        age=args.initial_age  # Use the specified initial age for steady state
    )

    # Generate plots for steady state
    print("\nGenerating steady state plots...")
    steady_state_dir = Path("generated/figures/steady_state")
    steady_state_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the final state from steady state simulation
    y0_ss = sol_ss.ys[-1].copy()
    
    # Modify initial conditions as specified
    module_name = "combined_master_jax"
    import importlib
    jax_module = importlib.import_module(module_name)
    y_indexes = jax_module.y_indexes
    
    # Set Microglia_Hi_Fract_0 to 0.02
    if 'Microglia_Hi_Fract' in y_indexes:
        y0_ss = y0_ss.at[y_indexes['Microglia_Hi_Fract']].set(0.02)
    
    # Set PVS plaque values equal to final plaque values
    if 'AB40_Plaque_unbound' in y_indexes and 'AB40_Plaque_unbound_PVS' in y_indexes:
        final_ab40_plaque = sol_ss.ys[-1, y_indexes['AB40_Plaque_unbound']]
        y0_ss = y0_ss.at[y_indexes['AB40_Plaque_unbound_PVS']].set(final_ab40_plaque)
    
    if 'AB42_Plaque_unbound' in y_indexes and 'AB42_Plaque_unbound_PVS' in y_indexes:
        final_ab42_plaque = sol_ss.ys[-1, y_indexes['AB42_Plaque_unbound']]
        y0_ss = y0_ss.at[y_indexes['AB42_Plaque_unbound_PVS']].set(final_ab42_plaque)
    
    # Calculate the total age in years (initial age + steady state duration)
    total_age = args.initial_age + (steady_state_duration / (24 * 365))  # Convert hours to years
    
    # Run simulation with drug dosing for 3 years (with dosing stopping at 1.5 years)
    print("\nRunning with drug dose from steady state...")
    print(f"Starting with age {total_age:.1f} years (initial {args.initial_age} + {steady_state_duration/(24*365):.1f} years from steady state)")
    sol, c, dosing_times = run_simulation(
        3*365*24,  # Run for 3 years
        drug_type=args.drug, 
        steady_state=False, 
        y0_ss=y0_ss,
        age=total_age  # Pass the total age (initial + steady state duration)
    )
    
    # Save the full solution to CSV
    print("\nSaving final solution data...")
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    jax_module = importlib.import_module(module_name)
    y_indexes = jax_module.y_indexes
    
    # Create a DataFrame with time and all species values
    data = {"time": sol.ts}
    for species_name, idx in y_indexes.items():
        data[species_name] = sol.ys[:, idx]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    output_dir = Path("generated/simulation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_file = output_dir / f"drug_simulation_multi_dose_{args.drug.lower()}.csv"
    df.to_csv(output_file, index=False)
    print(f"Full solution saved to {output_file}")
    
    # Create drug simulation directory
    drug_sim_dir = Path("generated/figures/drug_simulation_multi_dose")
    drug_sim_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for drug simulation
    plot_Ab_t_dynamics(sol, c, dosing_times, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_amyloid_dynamics(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_microglia_dynamics(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_individual_oligomers(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_fibrils_and_plaques(sol, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_bbb_bcsfb_subplots(sol, c, dosing_times, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_csf_subplots(sol, c, dosing_times, drug_type=args.drug, plots_dir=drug_sim_dir)
    plot_ab_bound_concentrations(sol, c, dosing_times, drug_type=args.drug, plots_dir=drug_sim_dir)
    
    print("\nAll plots saved in generated/figures/drug_simulation_multi_dose")

if __name__ == "__main__":
    main() 