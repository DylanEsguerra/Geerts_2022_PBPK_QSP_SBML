"""
Script to plot Ab_t and CSF dynamics from saved solution data.
This script reads the saved solution data from run_combined_master_model_multi_dose.py
and recreates the Ab_t and CSF plots.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import argparse
import importlib

# Add the project root to Python path
root_dir = Path(__file__).parents[1]  # Go up 1 level to reach models directory
sys.path.append(str(root_dir))

# Add generated/jax directory to Python path
jax_dir = Path("generated/jax").absolute()
if str(jax_dir) not in sys.path:
    sys.path.append(str(jax_dir))

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

def plot_Ab_t_dynamics(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
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
    isf_bound = np.zeros_like(df['time'].values)
    isf_bound += df['AB40_monomer_antibody_bound'].values
    isf_bound += df['AB42_monomer_antibody_bound'].values
    
    # Sum bound oligomers in ISF
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Oligomer{i:02d}_Antibody_bound"
            if species_name in df.columns:
                isf_bound += df[species_name].values
    
    # Sum bound fibrils in ISF
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Fibril{i:02d}_Antibody_bound"
            if species_name in df.columns:
                isf_bound += df[species_name].values
    
    # Add bound plaques in ISF
    for species_name in ["AB40_Plaque_Antibody_bound", "AB42_Plaque_Antibody_bound"]:
        if species_name in df.columns:
            isf_bound += df[species_name].values
    
    # Calculate total bound amyloid in PVS
    pvs_bound = np.zeros_like(df['time'].values)
    for ab_type in ['AB40', 'AB42']:
        # Add bound monomers
        species_name = f'{ab_type}_Monomer_PVS_bound'
        if species_name in df.columns:
            pvs_bound += df[species_name].values
        
        # Add bound oligomers
        for i in range(2, 17):
            species_name = f'{ab_type}_Oligomer{i:02d}_PVS_bound'
            if species_name in df.columns:
                pvs_bound += df[species_name].values
        
        # Add bound fibrils
        for i in range(17, 25):
            species_name = f'{ab_type}_Fibril{i:02d}_PVS_bound'
            if species_name in df.columns:
                pvs_bound += df[species_name].values
        
        # Add bound plaques
        species_name = f'{ab_type}_Plaque_bound_PVS'
        if species_name in df.columns:
            pvs_bound += df[species_name].values
    
    # Calculate total antibody concentrations
    isf_total = df['Ab_t'].values + isf_bound
    pvs_total = df['C_Antibody_unbound_PVS'].values + pvs_bound
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Plot ISF unbound antibody (top left)
    axs[0, 0].plot(x_values, df['Ab_t'].values / c[c_indexes['VIS_brain']], 
                   label='Unbound', linewidth=2, color='blue')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('Brain ISF Unbound Antibody', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)
    
    # Plot PVS unbound antibody (top right)
    axs[0, 1].plot(x_values, df['C_Antibody_unbound_PVS'].values / c[c_indexes['V_PVS']], 
                   label='Unbound', linewidth=2, color='green')
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('PVS Unbound Antibody', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)
    
    # Plot ISF total antibody (bottom left)
    axs[1, 0].plot(x_values, isf_total / c[c_indexes['VIS_brain']],
                   label='Total', linewidth=2, color='red')
    axs[1, 0].plot(x_values, df['Ab_t'].values / c[c_indexes['VIS_brain']],
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
    axs[1, 1].plot(x_values, df['C_Antibody_unbound_PVS'].values / c[c_indexes['V_PVS']],
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
    #plt.show()

def plot_csf_subplots(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot CSF compartment concentrations in subplots, including total CSF concentration"""
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
    
    # Load experimental data
    exp_data_path = Path("generated/simulation_results/Fig_4_CSF_Geerts.csv")
    print(f"Looking for experimental data at: {exp_data_path.absolute()}")
    if exp_data_path.exists():
        try:
            exp_data = pd.read_csv(exp_data_path)
            print("Found experimental data with columns:", exp_data.columns.tolist())
            exp_years = exp_data['Year'].values
            exp_conc = exp_data['Concentration'].values
            print(f"Loaded {len(exp_years)} experimental data points")
        except Exception as e:
            print(f"Error loading experimental data: {str(e)}")
            exp_years = None
            exp_conc = None
    else:
        print(f"Warning: Experimental data file not found at {exp_data_path}")
        exp_years = None
        exp_conc = None
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x3 subplots (added one for total CSF)
    fig, axs = plt.subplots(2, 3, figsize=(20, 12), sharex=True)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Calculate total bound amyloid in CSF compartments
    bound_amyloid = np.zeros_like(df['time'].values)
    for compartment in ['LV', 'TFV', 'CM', 'SAS']:
        for ab_type in ['AB40', 'AB42']:
            species_name = f'{ab_type}Mb_{compartment}'
            if species_name in df.columns:
                bound_amyloid += df[species_name].values
    
    # Calculate total antibody mass in all CSF compartments
    total_antibody_mass = (
        df['PK_LV_brain'].values +
        df['PK_TFV_brain'].values +
        df['PK_CM_brain'].values +
        df['PK_SAS_brain'].values
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
    axs[0, 0].plot(x_values, df['PK_LV_brain'].values / c[c_indexes['V_LV_brain']], 
                   color='blue', linewidth=2, label='LV')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('Lateral Ventricles (LV)', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot TFV concentration
    axs[0, 1].plot(x_values, df['PK_TFV_brain'].values / c[c_indexes['V_TFV_brain']], 
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
    axs[1, 0].plot(x_values, df['PK_CM_brain'].values / c[c_indexes['V_CM_brain']], 
                   color='purple', linewidth=2, label='CM')
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('Cisterna Magna (CM)', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot SAS concentration with experimental data
    axs[1, 1].plot(x_values, df['PK_SAS_brain'].values / c[c_indexes['V_SAS_brain']], 
                   color='orange', linewidth=2, label='SAS')
    if exp_years is not None and exp_conc is not None:
        axs[1, 1].scatter(exp_years, exp_conc, 
                         color='red', s=50, label='Published Results', zorder=5)
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('Subarachnoid Space (SAS)', fontsize=14)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=10)
    
    # Plot BCSFB concentrations
    axs[1, 2].plot(x_values, df['PK_BCSFB_bound_brain'].values / c[c_indexes['V_BCSFB_brain']], 
                   color='brown', linewidth=2, label='BCSFB Bound')
    axs[1, 2].plot(x_values, df['PK_BCSFB_unbound_brain'].values / c[c_indexes['V_BCSFB_brain']] , 
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
    fig_total = plt.figure(figsize=(12, 12))
    gs = fig_total.add_gridspec(2, 1, height_ratios=[1, 1])
    ax1 = fig_total.add_subplot(gs[0])
    ax2 = fig_total.add_subplot(gs[1], sharex=ax1)
    
    # First panel: Total CSF and experimental data
    ax1.plot(x_values, total_csf, 
             color='blue', linewidth=3, label='Total CSF (Incl. Bound)')
    if exp_years is not None and exp_conc is not None:
        ax1.scatter(exp_years, exp_conc, 
                   color='red', s=50, label='Published Results', zorder=5)
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax1.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax1.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
                label='End of Dosing (1.5 years)')
    
    # Set axis labels and title for first panel
    ax1.set_ylabel('Concentration (nM)', fontsize=14)
    ax1.set_title('Total CSF Concentration', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Second panel: Free and bound components
    ax2.plot(x_values, total_antibody, 
             color='red', linewidth=2, label='Free Antibody')
    ax2.plot(x_values, total_bound, 
             color='green', linewidth=2, label='Bound Amyloid')
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax2.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax2.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
                label='End of Dosing (1.5 years)')
    
    # Set axis labels and title for second panel
    ax2.set_xlabel('Time (years)', fontsize=14)
    ax2.set_ylabel('Concentration (nM)', fontsize=14)
    ax2.set_title('CSF Antibody Components', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Set x-axis to show full 3 years
    ax1.set_xlim(0, 3)
    
    # Increase tick label size for both panels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add overall title
    plt.suptitle(f'{drug_name} CSF Antibody Dynamics: {dose_info}',
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save total CSF figure
    fig_total.savefig(plots_dir / f'{drug_name.lower()}_total_csf.png',
                    dpi=300, bbox_inches='tight')
    plt.close()

def plot_total_isf_mab(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot total ISF mAb concentration (ISF + PVS total concentrations)"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Calculate total bound amyloid mass in ISF
    isf_bound_mass = np.zeros_like(df['time'].values)
    isf_bound_mass += df['AB40_monomer_antibody_bound'].values
    isf_bound_mass += df['AB42_monomer_antibody_bound'].values
    
    # Sum bound oligomers in ISF
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Oligomer{i:02d}_Antibody_bound"
            if species_name in df.columns:
                isf_bound_mass += df[species_name].values
    
    # Sum bound fibrils in ISF
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Fibril{i:02d}_Antibody_bound"
            if species_name in df.columns:
                isf_bound_mass += df[species_name].values
    
    # Add bound plaques in ISF
    for species_name in ["AB40_Plaque_Antibody_bound", "AB42_Plaque_Antibody_bound"]:
        if species_name in df.columns:
            isf_bound_mass += df[species_name].values
    
    # Calculate total bound amyloid mass in PVS
    pvs_bound_mass = np.zeros_like(df['time'].values)
    for ab_type in ['AB40', 'AB42']:
        # Add bound monomers
        species_name = f'{ab_type}_Monomer_PVS_bound'
        if species_name in df.columns:
            pvs_bound_mass += df[species_name].values
        
        # Add bound oligomers
        for i in range(2, 17):
            species_name = f'{ab_type}_Oligomer{i:02d}_PVS_bound'
            if species_name in df.columns:
                pvs_bound_mass += df[species_name].values
        
        # Add bound fibrils
        for i in range(17, 25):
            species_name = f'{ab_type}_Fibril{i:02d}_PVS_bound'
            if species_name in df.columns:
                pvs_bound_mass += df[species_name].values
        
        # Add bound plaques
        species_name = f'{ab_type}_Plaque_bound_PVS'
        if species_name in df.columns:
            pvs_bound_mass += df[species_name].values
    
    # Calculate total antibody masses
    isf_total_mass = df['Ab_t'].values + isf_bound_mass
    pvs_total_mass = df['C_Antibody_unbound_PVS'].values + pvs_bound_mass
    
    # Calculate total volume
    total_volume = c[c_indexes['VIS_brain']] + c[c_indexes['V_PVS']]
    
    # Calculate concentrations (mass/volume)
    isf_total = isf_total_mass / c[c_indexes['VIS_brain']]
    pvs_total = pvs_total_mass / c[c_indexes['V_PVS']]
    total_isf_mab = (isf_total_mass + pvs_total_mass) / total_volume
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Plot total ISF mAb concentration
    ax.plot(x_values, total_isf_mab, 
            color='red', linewidth=3, label='Total ISF mAb')
    ax.plot(x_values, isf_total, 
            color='blue', linewidth=2, label='ISF Total')
    ax.plot(x_values, pvs_total, 
            color='green', linewidth=2, label='PVS Total')
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
               label='End of Dosing (1.5 years)')
    
    # Set axis labels and title
    ax.set_xlabel('Time (years)', fontsize=14)
    ax.set_ylabel('Concentration (nM)', fontsize=14)
    ax.set_title(f'{drug_name} Total ISF mAb Concentration: {dose_info}',
                 fontsize=16, fontweight='bold')
    
    # Set x-axis to show full 3 years
    ax.set_xlim(0, 3)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_total_isf_mab.png',
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_total_sas_antibody(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot total SAS antibody concentration (unbound + bound amyloid)"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Load experimental data
    exp_data_path = Path("generated/simulation_results/Fig_4_CSF_Geerts.csv")
    print(f"Looking for experimental data at: {exp_data_path.absolute()}")
    if exp_data_path.exists():
        try:
            exp_data = pd.read_csv(exp_data_path)
            print("Found experimental data with columns:", exp_data.columns.tolist())
            exp_years = exp_data['Year'].values
            exp_conc = exp_data['Concentration'].values
            print(f"Loaded {len(exp_years)} experimental data points")
        except Exception as e:
            print(f"Error loading experimental data: {str(e)}")
            exp_years = None
            exp_conc = None
    else:
        print(f"Warning: Experimental data file not found at {exp_data_path}")
        exp_years = None
        exp_conc = None
    
    # Calculate total bound amyloid mass in SAS
    sas_bound_mass = np.zeros_like(df['time'].values)
    for ab_type in ['AB40', 'AB42']:
        species_name = f'{ab_type}Mb_SAS'
        if species_name in df.columns:
            sas_bound_mass += df[species_name].values
    
    # Calculate total SAS antibody mass
    sas_unbound_mass = df['PK_SAS_brain'].values
    sas_total_mass = sas_unbound_mass + sas_bound_mass
    
    # Calculate concentrations (mass/volume)
    sas_unbound = sas_unbound_mass / c[c_indexes['V_SAS_brain']]
    sas_bound = sas_bound_mass / c[c_indexes['V_SAS_brain']]
    sas_total = sas_total_mass / c[c_indexes['V_SAS_brain']]
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # First panel: Total SAS and experimental data
    ax1.plot(x_values, sas_total, 
            color='blue', linewidth=3, label='Total SAS Antibody')
    if exp_years is not None and exp_conc is not None:
        ax1.scatter(exp_years, exp_conc, 
                  color='red', s=50, label='Published Results', zorder=5)
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax1.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax1.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
               label='End of Dosing (1.5 years)')
    
    # Set axis labels and title for first panel
    ax1.set_ylabel('Concentration (nM)', fontsize=14)
    ax1.set_title('Total SAS Antibody Concentration', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Second panel: Bound and unbound components
    ax2.plot(x_values, sas_unbound, 
            color='red', linewidth=2, label='Unbound Antibody')
    ax2.plot(x_values, sas_bound, 
            color='green', linewidth=2, label='Bound Amyloid')
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax2.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax2.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
               label='End of Dosing (1.5 years)')
    
    # Set axis labels and title for second panel
    ax2.set_xlabel('Time (years)', fontsize=14)
    ax2.set_ylabel('Concentration (nM)', fontsize=14)
    ax2.set_title('SAS Antibody Components', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Set x-axis to show full 3 years
    ax1.set_xlim(0, 3)
    
    # Increase tick label size for both panels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add overall title
    plt.suptitle(f'{drug_name} SAS Antibody Dynamics: {dose_info}',
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_total_sas_antibody.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_fcrn_free(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot FcRn free BBB and FcRn free BCSFB concentrations over time"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Calculate concentrations
    fcrn_free_bbb = df['FcRn_free_BBB'].values / c[c_indexes['VBBB_brain']]
    fcrn_free_bcsfb = df['FcRn_free_BCSFB'].values / c[c_indexes['V_BCSFB_brain']]
    
    # Plot FcRn free concentrations
    ax.plot(x_values, fcrn_free_bbb, 
            color='blue', linewidth=2, label='FcRn free BBB')
    ax.plot(x_values, fcrn_free_bcsfb, 
            color='green', linewidth=2, label='FcRn free BCSFB')
    
    # Add dosing markers
    for dose_time in dosing_times:
        ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add end of dosing line
    ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
               label='End of Dosing (1.5 years)')
    
    # Set axis labels and title
    ax.set_xlabel('Time (years)', fontsize=14)
    ax.set_ylabel('Concentration (nM)', fontsize=14)
    ax.set_title(f'{drug_name} FcRn Free Concentrations: {dose_info}',
                 fontsize=16, fontweight='bold')
    
    # Set x-axis to show full 3 years
    ax.set_xlim(0, 3)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=12, framealpha=1, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure (commented out as requested)
    fig.savefig(plots_dir / f'{drug_name.lower()}_fcrn_free.png',
                dpi=300, bbox_inches='tight')
    #plt.show()

def plot_isf_components(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Create a multi-paneled plot showing all components of ISF mAb concentration"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Calculate bound amyloid components in ISF
    isf_bound_monomers = np.zeros_like(df['time'].values)
    isf_bound_monomers += df['AB40_monomer_antibody_bound'].values
    isf_bound_monomers += df['AB42_monomer_antibody_bound'].values
    
    # Sum bound oligomers in ISF
    isf_bound_oligomers = np.zeros_like(df['time'].values)
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Oligomer{i:02d}_Antibody_bound"
            if species_name in df.columns:
                isf_bound_oligomers += df[species_name].values
    
    # Sum bound fibrils in ISF
    isf_bound_fibrils = np.zeros_like(df['time'].values)
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            species_name = f"{prefix}_Fibril{i:02d}_Antibody_bound"
            if species_name in df.columns:
                isf_bound_fibrils += df[species_name].values
    
    # Add bound plaques in ISF
    isf_bound_plaques = np.zeros_like(df['time'].values)
    for species_name in ["AB40_Plaque_Antibody_bound", "AB42_Plaque_Antibody_bound"]:
        if species_name in df.columns:
            isf_bound_plaques += df[species_name].values
    
    # Calculate concentrations (mass/volume)
    isf_unbound = df['Ab_t'].values / c[c_indexes['VIS_brain']]
    isf_bound_monomers_conc = isf_bound_monomers / c[c_indexes['VIS_brain']]
    isf_bound_oligomers_conc = isf_bound_oligomers / c[c_indexes['VIS_brain']]
    isf_bound_fibrils_conc = isf_bound_fibrils / c[c_indexes['VIS_brain']]
    isf_bound_plaques_conc = isf_bound_plaques / c[c_indexes['VIS_brain']]
    
    # Plot unbound antibody (top left)
    axs[0, 0].plot(x_values, isf_unbound, 
                   color='blue', linewidth=2, label='Unbound')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('Unbound Antibody', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)
    
    # Plot bound monomers and oligomers (top right)
    axs[0, 1].plot(x_values, isf_bound_monomers_conc, 
                   color='green', linewidth=2, label='Bound Monomers')
    axs[0, 1].plot(x_values, isf_bound_oligomers_conc, 
                   color='orange', linewidth=2, label='Bound Oligomers')
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('Bound Monomers and Oligomers', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)
    
    # Plot bound fibrils and plaques (bottom left)
    axs[1, 0].plot(x_values, isf_bound_fibrils_conc, 
                   color='red', linewidth=2, label='Bound Fibrils')
    axs[1, 0].plot(x_values, isf_bound_plaques_conc, 
                   color='purple', linewidth=2, label='Bound Plaques')
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('Bound Fibrils and Plaques', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)
    
    # Plot total ISF mAb (bottom right)
    total_isf = (isf_unbound + isf_bound_monomers_conc + 
                 isf_bound_oligomers_conc + isf_bound_fibrils_conc + 
                 isf_bound_plaques_conc)
    axs[1, 1].plot(x_values, total_isf, 
                   color='black', linewidth=3, label='Total ISF mAb')
    axs[1, 1].plot(x_values, isf_unbound, 
                   color='blue', linewidth=2, label='Unbound')
    axs[1, 1].plot(x_values, isf_bound_monomers_conc + isf_bound_oligomers_conc + 
                   isf_bound_fibrils_conc + isf_bound_plaques_conc,
                   color='red', linewidth=2, label='Total Bound')
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('Total ISF mAb Components', fontsize=14)
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
    plt.suptitle(f'{drug_name} ISF mAb Components: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_isf_components.png',
                dpi=300, bbox_inches='tight')
    #plt.show()

def plot_pvs_components(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Create a multi-paneled plot showing all components of PVS mAb concentration"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Determine which drug is being used
    is_lecanemab = drug_type.lower() == "lecanemab"
    drug_name = "Lecanemab" if is_lecanemab else "Gantenerumab"
    dose_info = "10 mg/kg IV q2w" if is_lecanemab else "1200 mg SC q4w"
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Calculate bound amyloid components in PVS
    pvs_bound_monomers = np.zeros_like(df['time'].values)
    for ab_type in ['AB40', 'AB42']:
        species_name = f'{ab_type}_Monomer_PVS_bound'
        if species_name in df.columns:
            pvs_bound_monomers += df[species_name].values
    
    # Sum bound oligomers in PVS
    pvs_bound_oligomers = np.zeros_like(df['time'].values)
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            species_name = f'{prefix}_Oligomer{i:02d}_PVS_bound'
            if species_name in df.columns:
                pvs_bound_oligomers += df[species_name].values
    
    # Sum bound fibrils in PVS
    pvs_bound_fibrils = np.zeros_like(df['time'].values)
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            species_name = f'{prefix}_Fibril{i:02d}_PVS_bound'
            if species_name in df.columns:
                pvs_bound_fibrils += df[species_name].values
    
    # Add bound plaques in PVS
    pvs_bound_plaques = np.zeros_like(df['time'].values)
    for ab_type in ['AB40', 'AB42']:
        species_name = f'{ab_type}_Plaque_bound_PVS'
        if species_name in df.columns:
            pvs_bound_plaques += df[species_name].values
    
    # Calculate concentrations (mass/volume)
    pvs_unbound = df['C_Antibody_unbound_PVS'].values / c[c_indexes['V_PVS']]
    pvs_bound_monomers_conc = pvs_bound_monomers / c[c_indexes['V_PVS']]
    pvs_bound_oligomers_conc = pvs_bound_oligomers / c[c_indexes['V_PVS']]
    pvs_bound_fibrils_conc = pvs_bound_fibrils / c[c_indexes['V_PVS']]
    pvs_bound_plaques_conc = pvs_bound_plaques / c[c_indexes['V_PVS']]
    
    # Plot unbound antibody (top left)
    axs[0, 0].plot(x_values, pvs_unbound, 
                   color='blue', linewidth=2, label='Unbound')
    axs[0, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 0].set_title('Unbound Antibody', fontsize=14)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)
    
    # Plot bound monomers and oligomers (top right)
    axs[0, 1].plot(x_values, pvs_bound_monomers_conc, 
                   color='green', linewidth=2, label='Bound Monomers')
    axs[0, 1].plot(x_values, pvs_bound_oligomers_conc, 
                   color='orange', linewidth=2, label='Bound Oligomers')
    axs[0, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[0, 1].set_title('Bound Monomers and Oligomers', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)
    
    # Plot bound fibrils and plaques (bottom left)
    axs[1, 0].plot(x_values, pvs_bound_fibrils_conc, 
                   color='red', linewidth=2, label='Bound Fibrils')
    axs[1, 0].plot(x_values, pvs_bound_plaques_conc, 
                   color='purple', linewidth=2, label='Bound Plaques')
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 0].set_title('Bound Fibrils and Plaques', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)
    
    # Plot total PVS mAb (bottom right)
    total_pvs = (pvs_unbound + pvs_bound_monomers_conc + 
                 pvs_bound_oligomers_conc + pvs_bound_fibrils_conc + 
                 pvs_bound_plaques_conc)
    axs[1, 1].plot(x_values, total_pvs, 
                   color='black', linewidth=3, label='Total PVS mAb')
    axs[1, 1].plot(x_values, pvs_unbound, 
                   color='blue', linewidth=2, label='Unbound')
    axs[1, 1].plot(x_values, pvs_bound_monomers_conc + pvs_bound_oligomers_conc + 
                   pvs_bound_fibrils_conc + pvs_bound_plaques_conc,
                   color='red', linewidth=2, label='Total Bound')
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].set_ylabel('Concentration (nM)', fontsize=12)
    axs[1, 1].set_title('Total PVS mAb Components', fontsize=14)
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
    plt.suptitle(f'{drug_name} PVS mAb Components: {dose_info}', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    fig.savefig(plots_dir / f'{drug_name.lower()}_pvs_components.png',
                dpi=300, bbox_inches='tight')
    #plt.show()

def plot_amyloid_dynamics(df, c, dosing_times, drug_type="gantenerumab", plots_dir=None):
    """Plot total AB40 and AB42 for oligomers, fibrils, and plaques in separate subplots with concentrations"""
    if plots_dir is None:
        plots_dir = Path("generated/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Import the generated model to get species indices
    module_name = "combined_master_jax"
    import importlib
    try:
        jax_module = importlib.import_module(module_name)
        c_indexes = jax_module.c_indexes
    except ImportError:
        print(f"Error: Could not import module {module_name}")
        return
    
    # Create figure with 3x2 subplots (AB40 left column, AB42 right column)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18), sharex=True)
    
    # Initialize arrays for total concentrations
    total_ab40_oligomers = np.zeros_like(df['time'].values)
    total_ab42_oligomers = np.zeros_like(df['time'].values)
    total_ab40_fibrils = np.zeros_like(df['time'].values)
    total_ab42_fibrils = np.zeros_like(df['time'].values)
    total_ab40_plaques = np.zeros_like(df['time'].values)
    total_ab42_plaques = np.zeros_like(df['time'].values)
    
    # Sum all AB40 oligomers (both free and bound)
    ab40_oligomer_count = 0
    for i in range(2, 17):
        species_name = f'AB40_Oligomer{i:02d}'
        if species_name in df.columns:
            total_ab40_oligomers += df[species_name].values
            ab40_oligomer_count += 1
        # Also add antibody-bound species if they exist
        bound_species_name = f'AB40_Oligomer{i:02d}_Antibody_bound'
        if bound_species_name in df.columns:
            total_ab40_oligomers += df[bound_species_name].values
    
    # Sum all AB42 oligomers (both free and bound)
    ab42_oligomer_count = 0
    for i in range(2, 17):
        species_name = f'AB42_Oligomer{i:02d}'
        if species_name in df.columns:
            total_ab42_oligomers += df[species_name].values
            ab42_oligomer_count += 1
        # Also add antibody-bound species if they exist
        bound_species_name = f'AB42_Oligomer{i:02d}_Antibody_bound'
        if bound_species_name in df.columns:
            total_ab42_oligomers += df[bound_species_name].values
    
    # Sum all AB40 fibrils (both free and bound)
    ab40_fibril_count = 0
    for i in range(17, 25):
        species_name = f'AB40_Fibril{i:02d}'
        if species_name in df.columns:
            total_ab40_fibrils += df[species_name].values
            ab40_fibril_count += 1
        # Also add antibody-bound species if they exist
        bound_species_name = f'AB40_Fibril{i:02d}_Antibody_bound'
        if bound_species_name in df.columns:
            total_ab40_fibrils += df[bound_species_name].values
    
    # Sum all AB42 fibrils (both free and bound)
    ab42_fibril_count = 0
    for i in range(17, 25):
        species_name = f'AB42_Fibril{i:02d}'
        if species_name in df.columns:
            total_ab42_fibrils += df[species_name].values
            ab42_fibril_count += 1
        # Also add antibody-bound species if they exist
        bound_species_name = f'AB42_Fibril{i:02d}_Antibody_bound'
        if bound_species_name in df.columns:
            total_ab42_fibrils += df[bound_species_name].values
    
    # Sum AB40 plaques (free and bound)
    if "AB40_Plaque_unbound" in df.columns:
        total_ab40_plaques += df["AB40_Plaque_unbound"].values
    if "AB40_Plaque_Antibody_bound" in df.columns:
        total_ab40_plaques += df["AB40_Plaque_Antibody_bound"].values
    
    # Sum AB42 plaques (free and bound)
    if "AB42_Plaque_unbound" in df.columns:
        total_ab42_plaques += df["AB42_Plaque_unbound"].values
    if "AB42_Plaque_Antibody_bound" in df.columns:
        total_ab42_plaques += df["AB42_Plaque_Antibody_bound"].values
    
    # Convert to concentrations by dividing by ISF brain volume
    vis_brain_volume = c[c_indexes['VIS_brain']]
    total_ab40_oligomers_conc = total_ab40_oligomers / vis_brain_volume
    total_ab42_oligomers_conc = total_ab42_oligomers / vis_brain_volume
    total_ab40_fibrils_conc = total_ab40_fibrils / vis_brain_volume
    total_ab42_fibrils_conc = total_ab42_fibrils / vis_brain_volume
    total_ab40_plaques_conc = total_ab40_plaques / vis_brain_volume
    total_ab42_plaques_conc = total_ab42_plaques / vis_brain_volume
    
    # Plot using years on x-axis
    x_values = df['time'].values / 24.0 / 365.0  # Convert hours to years
    
    # Plot AB40 oligomers (top left)
    ax1.plot(x_values, total_ab40_oligomers_conc, 
             label=f'Total (n={ab40_oligomer_count})', 
             linewidth=2, color='blue')
    ax1.set_ylabel('Concentration (nM)', fontsize=12)
    ax1.set_title('AB40 Oligomers', fontsize=14)
    
    # Plot AB42 oligomers (top right)
    ax2.plot(x_values, total_ab42_oligomers_conc, 
             label=f'Total (n={ab42_oligomer_count})', 
             linewidth=2, color='red')
    ax2.set_title('AB42 Oligomers', fontsize=14)
    
    # Plot AB40 fibrils (middle left)
    ax3.plot(x_values, total_ab40_fibrils_conc,
             label=f'Total (n={ab40_fibril_count})',
             linewidth=2, color='blue')
    ax3.set_ylabel('Concentration (nM)', fontsize=12)
    ax3.set_title('AB40 Fibrils', fontsize=14)
    
    # Plot AB42 fibrils (middle right)
    ax4.plot(x_values, total_ab42_fibrils_conc,
             label=f'Total (n={ab42_fibril_count})',
             linewidth=2, color='red')
    ax4.set_title('AB42 Fibrils', fontsize=14)
    
    # Plot AB40 plaques (bottom left)
    ax5.plot(x_values, total_ab40_plaques_conc,
             label='Total',
             linewidth=2, color='blue')
    ax5.set_ylabel('Concentration (nM)', fontsize=12)
    ax5.set_xlabel('Time (years)', fontsize=12)
    ax5.set_title('AB40 Plaques', fontsize=14)
    
    # Plot AB42 plaques (bottom right)
    ax6.plot(x_values, total_ab42_plaques_conc,
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
        
        # Add dosing markers
        for dose_time in dosing_times:
            ax.axvline(x=dose_time/24.0/365.0, color='gray', linestyle='--', alpha=0.3)
        
        # Add end of dosing line
        ax.axvline(x=1.5, color='red', linestyle='-', alpha=0.5,
                  label='End of Dosing (1.5 years)')
    
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

def main():
    """Main function to plot saved solution data"""
    parser = argparse.ArgumentParser(description="Plot saved solution data")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                        default="gantenerumab", help="Drug type to plot")
    args = parser.parse_args()
    
    # Print summary of plotting settings
    print(f"\n=== PLOTTING SAVED SOLUTION DATA ===")
    print(f"Drug: {args.drug.upper()}")
    print("=" * 40)
    
    # Load the saved solution data
    data_path = Path(f"generated/simulation_results/drug_simulation_multi_dose_{args.drug.lower()}.csv")
    if not data_path.exists():
        print(f"Error: No saved data found at {data_path}")
        print("Please run run_combined_master_model_multi_dose.py first")
        sys.exit(1)
    
    # Read the data
    df = pd.read_csv(data_path)
    
    # Import the generated model to get constants
    module_name = "combined_master_jax"
    import importlib
    jax_module = importlib.import_module(module_name)
    c = jax_module.c
    
    # Calculate dosing times based on drug type
    is_lecanemab = args.drug.lower() == "lecanemab"
    if is_lecanemab:
        # Lecanemab: every 2 weeks (336 hours) for 1.5 years
        dosing_interval = 336.0  # 2 weeks in hours
        max_dosing_time = 13140  # 1.5 years in hours
        num_doses = int(np.ceil(max_dosing_time / dosing_interval))
        dosing_times = np.arange(0, max_dosing_time, dosing_interval)
    else:
        # Gantenerumab: every 4 weeks (672 hours) for 1.5 years
        dosing_interval = 672.0  # 4 weeks in hours
        max_dosing_time = 13140  # 1.5 years in hours
        num_doses = int(np.ceil(max_dosing_time / dosing_interval))
        dosing_times = np.arange(0, max_dosing_time, dosing_interval)
    
    # Create plots directory
    plots_dir = Path("generated/figures/saved_solution")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    #print("\nGenerating Ab_t dynamics plot...")
    plot_Ab_t_dynamics(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    #print("Generating CSF subplots...")
    plot_csf_subplots(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    #print("Generating total ISF mAb plot...")
    plot_total_isf_mab(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    #print("Generating total SAS antibody plot...")
    plot_total_sas_antibody(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    #print("Generating FcRn free concentrations plot...")
    plot_fcrn_free(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    print("Generating ISF components plot...")
    plot_isf_components(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    print("Generating PVS components plot...")
    plot_pvs_components(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    print("Generating amyloid dynamics plot...")
    plot_amyloid_dynamics(df, c, dosing_times, drug_type=args.drug, plots_dir=plots_dir)
    
    print(f"\nAll plots saved in {plots_dir}")

if __name__ == "__main__":
    main() 