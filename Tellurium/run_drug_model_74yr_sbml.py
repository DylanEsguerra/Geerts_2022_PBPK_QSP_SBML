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

# Import K_rates_extrapolate for parameter updates
from K_rates_extrapolate import calculate_k_rates

# Simple solution and model classes for visualization
class Solution:
    def __init__(self, ts, ys):
        self.ts = ts
        self.ys = ys

class SimpleModel:
    def __init__(self, species_names, initial_conditions):
        self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
        self.y0 = initial_conditions

def update_aggregation_rates(rr):
    """Update all aggregation rates using K_rates_extrapolate calculation."""
    print("Updating aggregation rates using K_rates_extrapolate...")
    
    # Get the base rates from the model
    k_O1_O2_fortytwo = rr['k_M_O2_fortytwo']  # AB42 monomer to dimer
    k_O2_O3_fortytwo = rr['k_O2_O3_fortytwo']  # AB42 dimer to trimer
    k_O2_M_fortytwo = rr['k_O2_M_fortytwo']  # AB42 dimer to monomer
    k_O3_O2_fortytwo = rr['k_O3_O2_fortytwo']  # AB42 trimer to dimer
    
    # Calculate all rates using K_rates_extrapolate
    rates = calculate_k_rates(
        kf0_fortytwo=k_O1_O2_fortytwo,
        kf1_fortytwo=k_O2_O3_fortytwo,
        kb0_fortytwo=k_O2_M_fortytwo,
        kb1_fortytwo=k_O3_O2_fortytwo,
        baseline_ab42_plaque_rate=0.0443230279057089
    )
    
    # Update the model with calculated rates \
    failed_params = []
    
    # Just iterate through all the calculated rates and set them directly
    for param_name, value in rates.items():
        # Only update AB42 rates (skip AB40 and plaque rates for now)
        if 'fortytwo' in param_name:
            try:
                rr.setValue(param_name, value)
                print(f"Updated {param_name} to {value}")
            except Exception as e:
                error_msg = f"Could not set {param_name}: {e}"
                print(error_msg)
                failed_params.append(error_msg)
    
    if failed_params:
        print("Failed to set some aggregation rate parameters:")
        for param in failed_params:
            print(f"- {param}")
    else:
        print("Successfully updated all aggregation rates using K_rates_extrapolate")
    
    return failed_params

def run_drug_model_simulation(rr, years, output_file, is_placebo=False):
    """Run simulation for Drug model with SBML mode parameters."""
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()
    start_time = 0
    end_time = years * 365 * 24
    num_points = int(end_time/200)
    rr.reset()
    
    # Apply SBML mode parameter updates (same as compare_models_combined.py)
    if is_placebo:
        print("Setting SBML mode parameters for PLACEBO simulation...")
    else:
        print("Setting SBML mode parameters for DRUG simulation...")
    
    rr.setValue('IDE_conc', 7.208542043204368)
    rr.setValue('Microglia_cell_count', 0.6158207628681558)
    rr.setValue('k_APP_production', 125.39958281853755)
    rr.setValue('k_F24_O12_fortytwo', 8.411308100233814)
    rr.setValue('k_M_O2_fortytwo', 0.0011969797316319238)
    rr.setValue('k_O2_O3_fortytwo', 0.0009995988649416513)
    rr.setValue('k_O2_M_fortytwo', 71.27022085388667)
    rr.setValue('k_O3_O2_fortytwo', 5.696488107534822e-07)
    
    # SBML mode specific parameters
    rr.setValue('CL_AB42_IDE', 100.2)  # 400 * 0.2505 = 100.2 
    rr.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05)  # Volume-scaled 
    
    # Update all aggregation rates using K_rates_extrapolate
    failed_rates = update_aggregation_rates(rr)
    if failed_rates:
        print("Warning: Some aggregation rates could not be updated")
    
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
    # Add vertical dotted line at dosing end (1.5 years after start)
    ax1.axvline(x=71.5, color='grey', linestyle='--', alpha=0.7, linewidth=2)
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
    # Add vertical dotted line at dosing end (1.5 years after start)
    ax2.axvline(x=71.5, color='grey', linestyle='--', alpha=0.7, linewidth=2)
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
    # Add vertical dotted line at dosing end (1.5 years after start)
    ax3.axvline(x=71.5, color='grey', linestyle='--', alpha=0.7, linewidth=2)
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
    # Add vertical dotted line at dosing end (1.5 years after start)
    ax4.axvline(x=71.5, color='grey', linestyle='--', alpha=0.7, linewidth=2)
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


def create_mab_plot(sol, model, plots_dir):
    """Create plot for total mAb concentration in total CSF and ISF from years 70-73."""
    print("Creating mAb total CSF and ISF concentration plot...")
    
    # Convert time to years
    years = sol.ts / (24 * 365)
    
    # Find years 70-73
    mask_70_73 = (years >= 70) & (years <= 73)
    years_filtered = years[mask_70_73]
    
    # CSF compartment volumes (in liters)
    volume_lv = 0.0225
    volume_tfv = 0.0225
    volume_cm = 0.0075
    volume_sas = 0.09875
    total_csf_volume = volume_lv + volume_tfv + volume_cm + volume_sas
    
    # ISF volume (from model)
    volume_isf = 0.15  # liters
    
    y_indexes = model.y_indexes
    
    # ===== CSF CALCULATION =====
    # Calculate total mAb mass in all CSF compartments
    # Free antibody (PK_) in all CSF compartments
    pk_lv = sol.ys[:, y_indexes['PK_LV_brain']]
    pk_tfv = sol.ys[:, y_indexes['PK_TFV_brain']]
    pk_cm = sol.ys[:, y_indexes['PK_CM_brain']]
    pk_sas = sol.ys[:, y_indexes['PK_SAS_brain']]
    
    # Bound antibody (AB40Mb_ and AB42Mb_) in all CSF compartments
    ab40mb_lv = sol.ys[:, y_indexes['AB40Mb_LV']]
    ab40mb_tfv = sol.ys[:, y_indexes['AB40Mb_TFV']]
    ab40mb_cm = sol.ys[:, y_indexes['AB40Mb_CM']]
    ab40mb_sas = sol.ys[:, y_indexes['AB40Mb_SAS']]
    
    ab42mb_lv = sol.ys[:, y_indexes['AB42Mb_LV']]
    ab42mb_tfv = sol.ys[:, y_indexes['AB42Mb_TFV']]
    ab42mb_cm = sol.ys[:, y_indexes['AB42Mb_CM']]
    ab42mb_sas = sol.ys[:, y_indexes['AB42Mb_SAS']]
    
    # Calculate total mAb mass in each CSF compartment
    total_mab_lv = pk_lv + ab40mb_lv + ab42mb_lv
    total_mab_tfv = pk_tfv + ab40mb_tfv + ab42mb_tfv
    total_mab_cm = pk_cm + ab40mb_cm + ab42mb_cm
    total_mab_sas = pk_sas + ab40mb_sas + ab42mb_sas
    
    # Calculate total mAb mass across all CSF compartments
    total_mab_csf_mass = total_mab_lv + total_mab_tfv + total_mab_cm + total_mab_sas
    
    # Calculate total CSF concentration (total mass / total volume)
    total_mab_csf_concentration = total_mab_csf_mass / total_csf_volume
    
    # ===== ISF CALCULATION =====
    # Free antibody in ISF
    ab_t = sol.ys[:, y_indexes['Ab_t']]
    
    # Antibody bound to oligomers (O2-O16)
    oligomer_antibody_bound = np.zeros(len(sol.ts))
    for size in range(2, 17):
        # AB40 oligomers
        try:
            ab40_oligomer_bound = sol.ys[:, y_indexes[f'AB40_Oligomer{size:02d}_Antibody_bound']]
            oligomer_antibody_bound += ab40_oligomer_bound
        except KeyError:
            pass
        
        # AB42 oligomers
        try:
            ab42_oligomer_bound = sol.ys[:, y_indexes[f'AB42_Oligomer{size:02d}_Antibody_bound']]
            oligomer_antibody_bound += ab42_oligomer_bound
        except KeyError:
            pass
    
    # Antibody bound to fibrils (F17-F24)
    fibril_antibody_bound = np.zeros(len(sol.ts))
    for size in range(17, 25):
        # AB40 fibrils
        try:
            ab40_fibril_bound = sol.ys[:, y_indexes[f'AB40_Fibril{size}_Antibody_bound']]
            fibril_antibody_bound += ab40_fibril_bound
        except KeyError:
            pass
        
        # AB42 fibrils
        try:
            ab42_fibril_bound = sol.ys[:, y_indexes[f'AB42_Fibril{size}_Antibody_bound']]
            fibril_antibody_bound += ab42_fibril_bound
        except KeyError:
            pass
    
    # Antibody bound to plaques
    try:
        ab40_plaque_bound = sol.ys[:, y_indexes['AB40_Plaque_Antibody_bound']]
        ab42_plaque_bound = sol.ys[:, y_indexes['AB42_Plaque_Antibody_bound']]
        plaque_antibody_bound = ab40_plaque_bound + ab42_plaque_bound
    except KeyError:
        plaque_antibody_bound = np.zeros(len(sol.ts))
    
    # Calculate total ISF mAb mass
    total_mab_isf_mass = ab_t + oligomer_antibody_bound + fibril_antibody_bound + plaque_antibody_bound
    
    # Calculate total ISF concentration (total mass / ISF volume)
    total_mab_isf_concentration = total_mab_isf_mass / volume_isf
    
    # Filter for years 70-73
    total_mab_csf_filtered = total_mab_csf_concentration[mask_70_73]
    total_mab_isf_filtered = total_mab_isf_concentration[mask_70_73]
    
    # Create the plot with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Panel 1: Total CSF mAb
    ax1.plot(years_filtered, total_mab_csf_filtered, linewidth=3, color='blue', label='Total mAb CSF')
    ax1.axvline(x=71.5, color='grey', linestyle='--', alpha=0.7, linewidth=2)
    ax1.set_ylabel('Total mAb Concentration (nM)', fontsize=16, fontweight='bold')
    ax1.set_title('mAb Total CSF Concentration (Years 70-73)', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Panel 2: Total ISF mAb
    ax2.plot(years_filtered, total_mab_isf_filtered, linewidth=3, color='red', label='Total mAb ISF')
    ax2.axvline(x=71.5, color='grey', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_ylabel('Total mAb Concentration (nM)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time (years)', fontsize=16, fontweight='bold')
    ax2.set_title('mAb Total ISF Concentration (Years 70-73)', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    fig.savefig(plots_dir / 'gantenerumab_total_csf.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"mAb total CSF and ISF plot saved to {plots_dir}")



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
    
    # Create mAb plot (drug only)
    create_mab_plot(sol_drug, model_drug, plots_dir)
    
    # Get final values for summary (using drug simulation)
    final_values = get_ab42_ratios_and_concentrations_final_values(sol_drug, model_drug)
    final_suvr = get_suvr_final_value(sol_drug, model_drug)
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Simulation duration: {years} years")
    print(f"Model: Drug model (combined_master_model_gantenerumab_DRUG.txt)")
    print(f"Mode: SBML parameters with K_rates_extrapolate")
    print(f"Key parameters:")
    print(f"  - CL_AB42_IDE: 100.2")
    print(f"  - exp_decline_rate_IDE_fortytwo: {1.15E-05*0.2525:.2e}")
    print(f"  - Aggregation rates: Updated using K_rates_extrapolate")
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