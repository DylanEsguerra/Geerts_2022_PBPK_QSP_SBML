"""
Drug Model 74-Year Simulation with SBML Mode Parameters

This script runs a 74-year simulation of the Drug model (combined_master_model_gantenerumab_DRUG.txt)
using the SBML mode parameter settings from compare_models_combined.py.

Key SBML Mode Parameters:
- CL_AB42_IDE = 100.2 (400 * 0.2505 = 100.2)
- exp_decline_rate_IDE_fortytwo = 1.15E-05*0.2525 (volume-scaled for SBML)

This script applies the same parameter updates as the SBML mode in compare_models_combined.py
but specifically for the Drug model and for a 74-year simulation period.
"""

import tellurium as te
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add parent directory to path to import visualization functions
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from visualize_tellurium_simulation import (
    get_ab42_ratios_and_concentrations_final_values,
    get_suvr_final_value,
    calculate_suvr
)

# Simple solution and model classes for visualization
class Solution:
    def __init__(self, ts, ys):
        self.ts = ts
        self.ys = ys

class SimpleModel:
    def __init__(self, species_names, initial_conditions):
        self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
        self.y0 = initial_conditions

def run_drug_model_simulation(rr, years, output_file, is_placebo=False):
    """Run simulation for Drug model with SBML mode parameters."""
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()
    start_time = 0
    end_time = years * 365 * 24
    num_points = int(end_time/100)
    rr.reset()
    
    # Apply SBML mode parameter updates (same as compare_models_combined.py)
    if is_placebo:
        print("Setting SBML mode parameters for PLACEBO simulation...")
    else:
        print("Setting SBML mode parameters for DRUG simulation...")
    
    rr.setValue('IDE_conc', 3)
    rr.setValue('k_APP_production', 75)
    rr.setValue('k_M_O2_fortytwo', 0.003564)
    rr.setValue('k_F24_O12_fortytwo', 100)
    rr.setValue('k_O2_O3_fortytwo', 0.01368)
    rr.setValue('k_O3_O4_fortytwo', 0.01)
    rr.setValue('k_O4_O5_fortytwo', 0.00273185)
    rr.setValue('k_O5_O6_fortytwo', 0.00273361)
    
    # SBML mode specific parameters
    rr.setValue('CL_AB42_IDE', 100.2)  # 400 * 0.2505 = 100.2 
    rr.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05*0.2525)  # Volume-scaled 
    
    # Set drug dose to 0 for placebo
    if is_placebo:
        rr.setValue('SC_DoseAmount', 0)
        print("PLACEBO: SC_DoseAmount set to 0")
    
    print(f"Running {years}-year {'PLACEBO' if is_placebo else 'DRUG'} simulation...")
    result = rr.simulate(start_time, end_time, num_points, selections=selections)

    df = pd.DataFrame(result, columns=selections)
    df.to_csv(output_file, index=False)
    print(f"{'PLACEBO' if is_placebo else 'DRUG'} simulation results saved to: {output_file}")
    return output_file

def create_drug_model_plots(sol_drug, model_drug, sol_placebo, model_placebo, plots_dir):
    """Create comprehensive plots for Drug model showing last 4 years of data with placebo comparison."""
    print("Creating Drug model visualization plots with placebo comparison...")
    
    # Convert time to years for both simulations
    years_drug = sol_drug.ts / (24 * 365)
    years_placebo = sol_placebo.ts / (24 * 365)
    
    # Find the last 4 years of data
    last_4_years_mask_drug = years_drug >= (years_drug[-1] - 4)
    last_4_years_mask_placebo = years_placebo >= (years_placebo[-1] - 4)
    years_filtered_drug = years_drug[last_4_years_mask_drug]
    years_filtered_placebo = years_placebo[last_4_years_mask_placebo]
    
    # Volume scaling factor for ISF compartment
    volume_scale_factor_isf = 0.2505
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Oligomer Loads (top left)
    y_indexes_drug = model_drug.y_indexes
    y_indexes_placebo = model_placebo.y_indexes
    
    # Calculate weighted oligomer loads for AB42 (O2-O16) - DRUG
    ab42_oligomer_load_drug = np.zeros(len(sol_drug.ts))
    for i in range(2, 17):
        species_name = f'AB42_Oligomer{i:02d}'
        if species_name in y_indexes_drug:
            ab42_oligomer_load_drug += (i-1) * sol_drug.ys[:, y_indexes_drug[species_name]]
    
    ab42_oligomer_load_filtered_drug = ab42_oligomer_load_drug[last_4_years_mask_drug] / volume_scale_factor_isf
    
    # Calculate weighted oligomer loads for AB42 (O2-O16) - PLACEBO
    ab42_oligomer_load_placebo = np.zeros(len(sol_placebo.ts))
    for i in range(2, 17):
        species_name = f'AB42_Oligomer{i:02d}'
        if species_name in y_indexes_placebo:
            ab42_oligomer_load_placebo += (i-1) * sol_placebo.ys[:, y_indexes_placebo[species_name]]
    
    ab42_oligomer_load_filtered_placebo = ab42_oligomer_load_placebo[last_4_years_mask_placebo] / volume_scale_factor_isf
    
    ax1.plot(years_filtered_drug, ab42_oligomer_load_filtered_drug, linewidth=3, color='green', label='Drug')
    ax1.plot(years_filtered_placebo, ab42_oligomer_load_filtered_placebo, linewidth=3, color='black', label='Placebo')
    ax1.set_ylabel('Oligomer Load (nM)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time (years)', fontsize=16, fontweight='bold')
    ax1.set_title('AB42 Oligomer Load (Last 4 Years)', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # 2. Protofibril Loads (top right)
    # Calculate weighted fibril loads for AB42 (F17-F24) - DRUG
    ab42_fibril_load_drug = np.zeros(len(sol_drug.ts))
    for i in range(17, 25):
        species_name = f'AB42_Fibril{i:02d}'
        if species_name in y_indexes_drug:
            ab42_fibril_load_drug += (i-1) * sol_drug.ys[:, y_indexes_drug[species_name]]
    
    ab42_fibril_load_filtered_drug = ab42_fibril_load_drug[last_4_years_mask_drug] / volume_scale_factor_isf
    
    # Calculate weighted fibril loads for AB42 (F17-F24) - PLACEBO
    ab42_fibril_load_placebo = np.zeros(len(sol_placebo.ts))
    for i in range(17, 25):
        species_name = f'AB42_Fibril{i:02d}'
        if species_name in y_indexes_placebo:
            ab42_fibril_load_placebo += (i-1) * sol_placebo.ys[:, y_indexes_placebo[species_name]]
    
    ab42_fibril_load_filtered_placebo = ab42_fibril_load_placebo[last_4_years_mask_placebo] / volume_scale_factor_isf
    
    ax2.plot(years_filtered_drug, ab42_fibril_load_filtered_drug, linewidth=3, color='green', label='Drug')
    ax2.plot(years_filtered_placebo, ab42_fibril_load_filtered_placebo, linewidth=3, color='black', label='Placebo')
    ax2.set_ylabel('Protofibril Load (nM)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time (years)', fontsize=16, fontweight='bold')
    ax2.set_title('AB42 Protofibril Load (Last 4 Years)', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    # 3. Plaque Loads (bottom left)
    ab42_plaque_drug = sol_drug.ys[:, y_indexes_drug['AB42_Plaque_unbound']] / volume_scale_factor_isf
    ab42_plaque_filtered_drug = ab42_plaque_drug[last_4_years_mask_drug]
    
    ab42_plaque_placebo = sol_placebo.ys[:, y_indexes_placebo['AB42_Plaque_unbound']] / volume_scale_factor_isf
    ab42_plaque_filtered_placebo = ab42_plaque_placebo[last_4_years_mask_placebo]
    
    ax3.plot(years_filtered_drug, ab42_plaque_filtered_drug, linewidth=3, color='green', label='Drug')
    ax3.plot(years_filtered_placebo, ab42_plaque_filtered_placebo, linewidth=3, color='black', label='Placebo')
    ax3.set_ylabel('Plaque Load (nM)', fontsize=16, fontweight='bold')
    ax3.set_xlabel('Time (years)', fontsize=16, fontweight='bold')
    ax3.set_title('AB42 Plaque Load (Last 4 Years)', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    # 4. SUVR Plot (bottom right)
    suvr_drug = calculate_suvr(sol_drug, model_drug)
    suvr_filtered_drug = suvr_drug[last_4_years_mask_drug]
    
    suvr_placebo = calculate_suvr(sol_placebo, model_placebo)
    suvr_filtered_placebo = suvr_placebo[last_4_years_mask_placebo]
    
    ax4.plot(years_filtered_drug, suvr_filtered_drug, linewidth=3, color='green', label='Drug')
    ax4.plot(years_filtered_placebo, suvr_filtered_placebo, linewidth=3, color='black', label='Placebo')
    ax4.set_ylabel('SUVR', fontsize=16, fontweight='bold')
    ax4.set_xlabel('Time (years)', fontsize=16, fontweight='bold')
    ax4.set_title('SUVR Progression (Last 4 Years)', fontsize=18, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    fig.savefig(plots_dir / 'gantenerumab_drug_model_last_4_years.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"Drug model plots with placebo comparison saved to {plots_dir}")


def create_mab_csf_plot(sol, model, plots_dir):
    """Create plot for total mAb concentration in CSF (SAS) from years 70-73."""
    print("Creating mAb total CSF (SAS) concentration plot...")
    
    # Convert time to years
    years = sol.ts / (24 * 365)
    
    # Find years 70-73
    mask_70_73 = (years >= 70) & (years <= 73)
    years_filtered = years[mask_70_73]
    
    # Volume scaling factor for SAS compartment
    volume_scale_factor_sas = 0.09875
    
    y_indexes = model.y_indexes
    
    # Calculate total mAb concentration in CSF (SAS)
    # PK_SAS_brain + AB40Mb_SAS + AB42Mb_SAS
    pk_sas = sol.ys[:, y_indexes['PK_SAS_brain']] / volume_scale_factor_sas
    ab40mb_sas = sol.ys[:, y_indexes['AB40Mb_SAS']] / volume_scale_factor_sas
    ab42mb_sas = sol.ys[:, y_indexes['AB42Mb_SAS']] / volume_scale_factor_sas
    
    total_mab_sas = pk_sas + ab40mb_sas + ab42mb_sas
    
    # Filter for years 70-73
    total_mab_sas_filtered = total_mab_sas[mask_70_73]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the total mAb concentration
    ax.plot(years_filtered, total_mab_sas_filtered, linewidth=3, color='blue', label='Total mAb CSF (SAS)')
    
    # Add dotted line at 1.5 years after dosing ends (assuming dosing ends at year 70)
    dosing_end_year = 70
    dotted_line_start = dosing_end_year + 1.5
    
    if dotted_line_start <= years_filtered[-1]:
        # Find the index where the dotted line should start
        dotted_start_idx = np.where(years_filtered >= dotted_line_start)[0]
        if len(dotted_start_idx) > 0:
            dotted_start_idx = dotted_start_idx[0]
            
            # Plot solid line before dosing ends
            ax.plot(years_filtered[:dotted_start_idx], total_mab_sas_filtered[:dotted_start_idx], 
                   linewidth=3, color='blue', label='During Dosing')
            
            # Plot dotted line after dosing ends
            ax.plot(years_filtered[dotted_start_idx:], total_mab_sas_filtered[dotted_start_idx:], 
                   linewidth=3, color='blue', linestyle='--', label='Post Dosing')
    
    ax.set_ylabel('Total mAb Concentration (nM)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (years)', fontsize=16, fontweight='bold')
    ax.set_title('mAb Total CSF (SAS) Concentration (Years 70-73)', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    fig.savefig(plots_dir / 'gantenerumab_mab_total_csf_sas.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"mAb CSF plot saved to {plots_dir}")



def main():
    """Main function to run the Drug model simulation."""
    years = 74.0
    print(f"Starting Drug model simulation for {years} years with SBML mode parameters")
    
    # Load Drug model
    drug_model_path = Path("../generated/sbml/combined_master_model_gantenerumab_DRUG.txt")
    print(f"Loading Drug model from: {drug_model_path}")
    
    with open(drug_model_path, "r") as f:
        antimony_str = f.read()
    
    # Create Tellurium model
    rr = te.loadAntimonyModel(antimony_str)
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)
    
    # Run DRUG simulation
    output_file_drug = f'drug_model_simulation_results_{years}yr_sbml.csv'
    run_drug_model_simulation(rr, years, output_file_drug, is_placebo=False)
    
    # Load DRUG results for analysis
    df_drug = pd.read_csv(output_file_drug)
    time_values_drug = df_drug['time'].values
    species_data_drug = df_drug.drop('time', axis=1).values
    sol_drug = Solution(time_values_drug, species_data_drug)
    model_drug = SimpleModel(df_drug.drop('time', axis=1).columns, species_data_drug[0])
    
    # Run PLACEBO simulation
    output_file_placebo = f'placebo_model_simulation_results_{years}yr_sbml.csv'
    run_drug_model_simulation(rr, years, output_file_placebo, is_placebo=True)
    
    # Load PLACEBO results for analysis
    df_placebo = pd.read_csv(output_file_placebo)
    time_values_placebo = df_placebo['time'].values
    species_data_placebo = df_placebo.drop('time', axis=1).values
    sol_placebo = Solution(time_values_placebo, species_data_placebo)
    model_placebo = SimpleModel(df_placebo.drop('time', axis=1).columns, species_data_placebo[0])
    
    # Create output directory for plots
    plots_dir = Path("simulation_plots/drug_model_74yr_sbml")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    
    # Create new visualization for Drug model - last 4 years with placebo comparison
    create_drug_model_plots(sol_drug, model_drug, sol_placebo, model_placebo, plots_dir)
    
    # Create mAb CSF plot (drug only)
    create_mab_csf_plot(sol_drug, model_drug, plots_dir)
    
    # Get final values for summary (using drug simulation)
    final_values = get_ab42_ratios_and_concentrations_final_values(sol_drug, model_drug)
    final_suvr = get_suvr_final_value(sol_drug, model_drug)
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Simulation duration: {years} years")
    print(f"Model: Drug model (combined_master_model_gantenerumab_DRUG.txt)")
    print(f"Mode: SBML parameters")
    print(f"Key parameters:")
    print(f"  - CL_AB42_IDE: 100.2")
    print(f"  - exp_decline_rate_IDE_fortytwo: {1.15E-05*0.2525:.2e}")
    print(f"Drug results file: {output_file_drug}")
    print(f"Placebo results file: {output_file_placebo}")
    print(f"Plots directory: {plots_dir}")
    print("\nFinal values:")
    print(f"  - AB42/AB40 ratio: {final_values['brain_plasma_ratio']:.4f}")
    print(f"  - AB42 concentration: {final_values['ab42_isf_pg_ml']:.4f} pg/mL")
    print(f"  - SUVR: {final_suvr:.4f}")
    print("="*60)
    
    print(f"\nSimulation complete!")
    print(f"Drug results saved to {output_file_drug}")
    print(f"Placebo results saved to {output_file_placebo}")
    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    main() 