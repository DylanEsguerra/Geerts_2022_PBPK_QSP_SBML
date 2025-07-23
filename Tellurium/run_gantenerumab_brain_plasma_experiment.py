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

from visualize_tellurium_simulation import calculate_suvr
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
    k_O1_O2_fortytwo = rr['k_M_O2_fortytwo']
    k_O2_O3_fortytwo = rr['k_O2_O3_fortytwo']
    k_O2_M_fortytwo = rr['k_O2_M_fortytwo']
    k_O3_O2_fortytwo = rr['k_O3_O2_fortytwo']
    rates = calculate_k_rates(
        kf0_fortytwo=k_O1_O2_fortytwo,
        kf1_fortytwo=k_O2_O3_fortytwo,
        kb0_fortytwo=k_O2_M_fortytwo,
        kb1_fortytwo=k_O3_O2_fortytwo,
        baseline_ab42_plaque_rate=0.0443230279057089
    )
    failed_params = []
    for param_name, value in rates.items():
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

def run_gantenerumab_experiment():
    # Simulation parameters
    years_pre = 70.0
    days_post = 85.0
    years_post = days_post / 365.0
    total_years = years_pre + years_post
    print(f"Running simulation for {years_pre} years pre-dose + {days_post} days post-dose ({total_years:.3f} years total)")

    # Load Drug model
    drug_model_path = Path("../generated/sbml/Gant_1_Dose.txt")
    print(f"Loading Drug model from: {drug_model_path}")
    with open(drug_model_path, "r") as f:
        antimony_str = f.read()
    rr = te.loadAntimonyModel(antimony_str)
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-6
    rr.integrator.relative_tolerance = 1e-6
    rr.integrator.setValue('stiff', True)
    rr.reset()

    # Set parameters (as in run_drug_model_74yr_sbml.py)
    '''
    rr.setValue('IDE_conc', 7.208542043204368)
    rr.setValue('Microglia_cell_count', 0.6158207628681558)
    rr.setValue('k_APP_production', 125.39958281853755)
    rr.setValue('k_F24_O12_fortytwo', 8.411308100233814)
    rr.setValue('k_M_O2_fortytwo', 0.0011969797316319238)
    rr.setValue('k_O2_O3_fortytwo', 0.0009995988649416513)
    rr.setValue('k_O2_M_fortytwo', 71.27022085388667)
    rr.setValue('k_O3_O2_fortytwo', 5.696488107534822e-07)
    rr.setValue('SC_DoseAmount', 2050.6)
    # SBML mode specific parameters
    rr.setValue('CL_AB42_IDE', 100.2)  # 400 * 0.2505 = 100.2 
    rr.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05)  # Volume-scaled 
    update_aggregation_rates(rr)
     '''
    rr.setValue('fta0', 1/1270)
    rr.setValue('fta1', 1/30.7)
    rr.setValue('fta2', 1/2.51)
    rr.setValue('fta3', 1/0.69)
    

    # Simulate
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()
    start_time = 0
    end_time = total_years * 365 * 24
    num_points = int((end_time - start_time) / 10)  # every 10 hours
    rr.reset()
    print(f"Simulating from {start_time} to {end_time} hours ({num_points} points)...")
    result = rr.simulate(start_time, end_time, num_points, selections=selections)
    df = pd.DataFrame(result, columns=selections)

    # Save results
    out_dir = Path("simulation_plots/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gantenerumab_brain_plasma_experiment.csv"
    df.to_csv(out_csv, index=False)
    print(f"Simulation results saved to: {out_csv}")

    # Extract time window for plotting (last 85 days)
    time_hours = df['time'].values
    time_days = (time_hours - years_pre * 365 * 24) / 24.0
    mask = (time_days >= 0) & (time_days <= days_post)

    # --- VOLUMES (set as constants if not available as global parameters) ---
    V_brain_plasma = 0.0319  # L
    V_central = 19.69       # L

    # Extract relevant columns (masses)
    y_indexes = {name: idx for idx, name in enumerate(df.columns)}
    pk_p_brain = df.iloc[mask, y_indexes['PK_p_brain']] 
    ab40mb_brain = df.iloc[mask, y_indexes['AB40Mb_Brain_Plasma']] 
    ab42mb_brain = df.iloc[mask, y_indexes['AB42Mb_Brain_Plasma']] 
    pk_sum = ab40mb_brain + ab42mb_brain + pk_p_brain
    plot_time = time_days[mask]

    pk_central = df.iloc[mask, y_indexes['PK_central']] 
    ab42mb_central = df.iloc[mask, y_indexes['AB42Mb_Central']] if 'AB42Mb_Central' in y_indexes else 0
    ab42mu_central = df.iloc[mask, y_indexes['AB42Mu_Central']] if 'AB42Mu_Central' in y_indexes else 0
    pk_central_sum = pk_central + ab42mb_central + ab42mu_central

    # Load empirical data
    empirical = pd.read_csv('Geerts_Gant_Data.csv')
    empirical_time = empirical['Time'].values
    empirical_conc = empirical['Concentration'].values

    # --- 4-PANEL PLOT ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top left: PK_p_brain (concentration) vs empirical data
    axs[0, 0].semilogy(plot_time, pk_p_brain/V_brain_plasma, color='black', linewidth=3, label='PK_p_brain (model)')
    axs[0, 0].scatter(empirical_time, empirical_conc, color='red', s=100, marker='o', edgecolors='red', linewidths=2, facecolors='none', label='Empirical Data')
    axs[0, 0].set_xlabel('Time after dose (days)')
    axs[0, 0].set_ylabel('Concentration (nM)')
    axs[0, 0].set_title('PK_p_brain (Brain Plasma) vs Empirical')
    axs[0, 0].set_ylim(1, 1000)
    axs[0, 0].set_xlim(0, days_post)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=10)

    # 2. Top right: (AB40Mb_Brain_Plasma + AB42Mb_Brain_Plasma)/V_brain_plasma and PK_p_brain/V_brain_plasma vs empirical
    axs[0, 1].semilogy(plot_time, (ab42mb_brain + ab40mb_brain)/V_brain_plasma, color='blue', linewidth=3, label='AB40Mb+AB42Mb Brain Plasma (model)')
    #axs[0, 1].scatter(empirical_time, empirical_conc, color='red', s=100, marker='o', edgecolors='red', linewidths=2, facecolors='none', label='Empirical Data')
    axs[0, 1].set_xlabel('Time after dose (days)')
    axs[0, 1].set_ylabel('Concentration (nM)')
    axs[0, 1].set_title('AB40Mb + AB42Mb(Brain Plasma)')
    #axs[0, 1].set_ylim(1, 1000)
    axs[0, 1].set_xlim(0, days_post)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=10)

    # 3. Bottom left: PK_central (concentration) vs empirical data
    axs[1, 0].semilogy(plot_time, pk_central/V_central, color='purple', linewidth=3, label='PK_central (model)')
    axs[1, 0].scatter(empirical_time, empirical_conc, color='red', s=100, marker='o', edgecolors='red', linewidths=2, facecolors='none', label='Empirical Data')
    axs[1, 0].set_xlabel('Time after dose (days)')
    axs[1, 0].set_ylabel('Concentration (nM)')
    axs[1, 0].set_title('PK_central (Central) vs Empirical')
    axs[1, 0].set_ylim(1, 1000)
    axs[1, 0].set_xlim(0, days_post)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(fontsize=10)

    # 4. Bottom right: (PK_central + AB42Mb_Central + AB42Mu_Central)/V_central vs empirical data
    axs[1, 1].semilogy(plot_time, (ab42mb_central + ab42mu_central)/V_central, color='green', linewidth=3, label='PK_central+AB42Mb+AB42Mu (model)')
    #axs[1, 1].scatter(empirical_time, empirical_conc, color='red', s=100, marker='o', edgecolors='red', linewidths=2, facecolors='none', label='Empirical Data')
    axs[1, 1].set_xlabel('Time after dose (days)')
    axs[1, 1].set_ylabel('Concentration (nM)')
    axs[1, 1].set_title('Central: AB42Mb+AB42Mu')
    #axs[1, 1].set_ylim(1, 1000)
    axs[1, 1].set_xlim(0, days_post)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / 'gantenerumab_4panel_vs_empirical.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    run_gantenerumab_experiment() 