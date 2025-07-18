import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
from visualize_tellurium_simulation import (
    load_tellurium_data, 
    create_solution_object, 
    create_plots,
    plot_individual_oligomers,
    plot_fibrils_and_plaques,
    plot_ab42_ratios_and_concentrations,
    plot_suvr,
    get_ab42_ratios_and_concentrations_final_values,
    get_suvr_final_value,
    calculate_suvr
)
import re

# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from K_rates_extrapolate import calculate_k_rates

def run_simulation(rr, years, output_file):
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()
    start_time = 0
    end_time = years * 365 * 24
    num_points = 300
    result = rr.simulate(start_time, end_time, num_points, selections=selections)
    df = pd.DataFrame(result, columns=selections)
    df.to_csv(output_file, index=False)
    return output_file

def set_optimized_values(rr):
    """
    Set optimized parameter values that are common to both APOE4 and non-APOE4 models.
    Returns a list of any parameters that couldn't be set.
    """
    failed_params = []

    #'''
    rr.setValue('IDE_conc', 3)
    #rr.setValue('IDE_conc', 0.005)
    rr.setValue('k_APP_production', 75)
    rr.setValue('k_M_O2_fortytwo',0.003564)
    rr.setValue('k_F24_O12_fortytwo',100)
    rr.setValue('k_O2_O3_fortytwo',0.01368)
    rr.setValue('k_O3_O4_fortytwo',0.01)
    rr.setValue('k_O4_O5_fortytwo',0.00273185)
    rr.setValue('k_O5_O6_fortytwo',0.00273361)
    # CL_AB42_IDE is now set in the main script based on mode
    #rr.setValue('CL_AB42_IDE', 400) # 400 * 0.2505 = 100.2
    rr.setValue('CL_AB42_IDE', 100.2) # 400 * 0.2505 = 100.2

    rr.setValue('exp_decline_rate_IDE_fortytwo',1.15E-05*0.2525)
    #'''


    # Calculate and set aggregation rates using K_rates_extrapolate with Don's specific values
    # No need to convert from h⁻¹ to s⁻¹ for the rate calculation function
    # Using Don's specific values as the base rates
    kf0_fortytwo = rr.getValue('k_M_O2_fortytwo') 
    kf1_fortytwo = rr.getValue('k_O2_O3_fortytwo')
    kb0_fortytwo = rr.getValue('k_O2_M_fortytwo') 
    kb1_fortytwo = rr.getValue('k_O3_O2_fortytwo') 
    
    try:
        rates = calculate_k_rates(
        kf0_fortytwo=kf0_fortytwo,
        kf1_fortytwo=kf1_fortytwo,
        kb0_fortytwo=kb0_fortytwo,
        kb1_fortytwo=kb1_fortytwo,
        baseline_ab42_plaque_rate=0.06,
    )
        
        # Update all AB42 rates, including plaque rates
        for key, value in rates.items():
            if 'Plaque_fortytwo' in key: # trying only don plot for initial values test 
                try:
                    setattr(rr, key, value)
                    print(f"Set {key} to {value}")
                except Exception as e:
                    error_msg = f"Could not set {key}: {e}"
                    print(error_msg)
                    failed_params.append(error_msg)
    except Exception as e:
        error_msg = f"Could not calculate aggregation rates: {e}"
        print(error_msg)
        failed_params.append(error_msg)

    # use these to override the rates     

    rr.setValue('k_O3_O4_fortytwo',0.01)
    rr.setValue('k_O4_O5_fortytwo',0.00273185)
    rr.setValue('k_O5_O6_fortytwo',0.00273361)
    
    return failed_params

def set_apoe4_params(rr):
    """Set APOE4-specific microglia parameters after setting optimized values."""
    # Set optimized values first
    failed_params = set_optimized_values(rr)
    
    # Set APOE4-specific microglia parameters
    apoe4_params = {
        'Microglia_EC50_forty': 20,
        'Microglia_EC50_fortytwo': 300,
        'Microglia_Vmax_forty': 0.0001,
        'Microglia_Vmax_fortytwo': 0.0001
    }
    
    for param_name, value in apoe4_params.items():
        try:
            rr.setValue(param_name, value)
        except Exception as e:
            error_msg = f"Could not set APOE4 parameter {param_name}: {e}"
            print(error_msg)
            failed_params.append(error_msg)
    
    if failed_params:
        print(f"APOE4 simulation: {len(failed_params)} parameters failed to set")
    else:
        print("APOE4 simulation: All parameters set successfully")

def set_nonapoe4_params(rr):
    """Set non-APOE4-specific microglia parameters after setting optimized values."""
    # Set optimized values first
    failed_params = set_optimized_values(rr)
    
    # Set non-APOE4-specific microglia parameters
    nonapoe4_params = {
        'Microglia_EC50_forty': 8,
        'Microglia_EC50_fortytwo': 120,
        'Microglia_Vmax_forty': 0.00015,
        'Microglia_Vmax_fortytwo': 0.00015
    }
    
    for param_name, value in nonapoe4_params.items():
        try:
            rr.setValue(param_name, value)
        except Exception as e:
            error_msg = f"Could not set non-APOE4 parameter {param_name}: {e}"
            print(error_msg)
            failed_params.append(error_msg)
    
    if failed_params:
        print(f"Non-APOE4 simulation: {len(failed_params)} parameters failed to set")
    else:
        print("Non-APOE4 simulation: All parameters set successfully")

def plot_six_panel_ab42_comparison(
    sol_apoe4, model_apoe4, sol_nonapoe4, model_nonapoe4, drug_type="gantenerumab", plots_dir=None
):
    """Create a six-panel plot comparing APOE4 and non-APOE4 simulations."""
    if plots_dir is None:
        plots_dir = Path("simulation_plots/tellurium_steady_state_apoe_compare")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Colors
    red = '#d62728'  # APOE4
    blue = '#1f77b4' # non-APOE4

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 10))

    # --- Process APOE4 Data ---
    x_values_a = sol_apoe4.ts / 24.0 / 365.0
    start_idx_a = np.where(x_values_a >= (max(x_values_a) - 80))[0][0]
    x_filtered_a = x_values_a[start_idx_a:]
    yidx_a = model_apoe4.y_indexes
    volume_scale_factor_isf = 0.2505

    # --- Process non-APOE4 Data ---
    x_values_n = sol_nonapoe4.ts / 24.0 / 365.0
    start_idx_n = np.where(x_values_n >= (max(x_values_n) - 80))[0][0]
    x_filtered_n = x_values_n[start_idx_n:]
    yidx_n = model_nonapoe4.y_indexes

    # Panel 1: Oligomers (Top Left)
    oligomer_loads_a = np.array([np.sum([int(name.split('Oligomer')[1]) * sol_apoe4.ys[t, idx] for name, idx in yidx_a.items() if re.match(r'AB42_Oligomer\d+$', name)]) for t in range(len(sol_apoe4.ts))]) / volume_scale_factor_isf
    oligomer_loads_n = np.array([np.sum([int(name.split('Oligomer')[1]) * sol_nonapoe4.ys[t, idx] for name, idx in yidx_n.items() if re.match(r'AB42_Oligomer\d+$', name)]) for t in range(len(sol_nonapoe4.ts))]) / volume_scale_factor_isf
    ax1.plot(x_filtered_a, oligomer_loads_a[start_idx_a:], linewidth=3, color=red, label='APOE4 Oligomers')
    ax1.plot(x_filtered_n, oligomer_loads_n[start_idx_n:], linewidth=3, color=blue, label='non-APOE4 Oligomers')
    ax1.set_title('Weighted Oligomers', fontsize=16, fontweight='bold')

    # Panel 2: Protofibrils (Top Right)
    fibril_loads_a = np.array([np.sum([int(name.split('Fibril')[1]) * sol_apoe4.ys[t, idx] for name, idx in yidx_a.items() if re.match(r'AB42_Fibril\d+$', name)]) for t in range(len(sol_apoe4.ts))]) / volume_scale_factor_isf
    fibril_loads_n = np.array([np.sum([int(name.split('Fibril')[1]) * sol_nonapoe4.ys[t, idx] for name, idx in yidx_n.items() if re.match(r'AB42_Fibril\d+$', name)]) for t in range(len(sol_nonapoe4.ts))]) / volume_scale_factor_isf
    ax2.plot(x_filtered_a, fibril_loads_a[start_idx_a:], linewidth=3, color=red, label='APOE4 Protofibrils')
    ax2.plot(x_filtered_n, fibril_loads_n[start_idx_n:], linewidth=3, color=blue, label='non-APOE4 Protofibrils')
    ax2.set_title('Weighted Protofibrils', fontsize=16, fontweight='bold')

    # Panel 3: SUVR (Middle Left)
    suvr_a = calculate_suvr(sol_apoe4, model_apoe4)[start_idx_a:]
    suvr_n = calculate_suvr(sol_nonapoe4, model_nonapoe4)[start_idx_n:]
    ax3.plot(x_filtered_a, suvr_a, linewidth=3, color=red, label='APOE4 SUVR')
    ax3.plot(x_filtered_n, suvr_n, linewidth=3, color=blue, label='non-APOE4 SUVR')
    ax3.set_title('SUVR', fontsize=16, fontweight='bold')

    # Panel 4: AB42 Monomer (Middle Right)
    monomer_a = sol_apoe4.ys[start_idx_a:, yidx_a['AB42_Monomer']] / volume_scale_factor_isf
    monomer_n = sol_nonapoe4.ys[start_idx_n:, yidx_n['AB42_Monomer']] / volume_scale_factor_isf
    ax4.plot(x_filtered_a, monomer_a, linewidth=3, color=red, label='APOE4 Monomer')
    ax4.plot(x_filtered_n, monomer_n, linewidth=3, color=blue, label='non-APOE4 Monomer')
    ax4.set_title('AB42 Monomer (ISF)', fontsize=16, fontweight='bold')

    # Panel 5: CL_AB42_IDE (Bottom Left)
    cl_ide_a = sol_apoe4.ys[start_idx_a:, yidx_a['CL_AB42_IDE']] / volume_scale_factor_isf
    cl_ide_n = sol_nonapoe4.ys[start_idx_n:, yidx_n['CL_AB42_IDE']] / volume_scale_factor_isf
    ax5.plot(x_filtered_a, cl_ide_a, linewidth=3, color=red, label='APOE4 CL_AB42_IDE')
    ax5.plot(x_filtered_n, cl_ide_n, linewidth=3, color=blue, label='non-APOE4 CL_AB42_IDE')
    ax5.set_title('AB42 IDE Clearance', fontsize=16, fontweight='bold')

    # Panel 6: AB42 Plaque (Bottom Right)
    plaque_a = sol_apoe4.ys[start_idx_a:, yidx_a['AB42_Plaque_unbound']] / volume_scale_factor_isf
    plaque_n = sol_nonapoe4.ys[start_idx_n:, yidx_n['AB42_Plaque_unbound']] / volume_scale_factor_isf
    ax6.plot(x_filtered_a, plaque_a, linewidth=3, color=red, label='APOE4 Plaque')
    ax6.plot(x_filtered_n, plaque_n, linewidth=3, color=blue, label='non-APOE4 Plaque')
    ax6.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold')


    # Set specific y-axis labels for each panel
    ax1.set_ylabel('Oligomer Load (nM)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Protofibril Load (nM)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('SUVR', fontsize=14, fontweight='bold')
    ax4.set_ylabel('AB42 Monomer (nM)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('IDE Clearance (1/h)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Plaque Load (nM)', fontsize=14, fontweight='bold')
    
    # Set x-axis labels and other formatting for all panels
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    fig.savefig(plots_dir / f'{drug_type.lower()}_six_panel_ab42_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tellurium simulation for APOE4 and non-APOE4 comparison.")
    parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
    args = parser.parse_args()

    sbml_path = Path("../generated/sbml/combined_master_model_gantenerumab.xml")
    #antimony_path = Path("../generated/sbml/Geerts_2023_1.txt")
    with open(sbml_path, "r") as f:
        sbml_str = f.read()
        #antimony_str = f.read()

    # --- Run APOE4 Simulation ---
    #rr = te.loadSBMLModel(sbml_str)
    rr = te.loadSBMLModel(sbml_str)
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)
    set_apoe4_params(rr)
    output_file_apoe4 = f'default_simulation_results_{args.years}yr_all_vars_apoe4.csv'
    run_simulation(rr, args.years, output_file_apoe4)
    df_apoe4 = pd.read_csv(output_file_apoe4)

    # --- Run non-APOE4 Simulation ---
    rr.reset() # Reset for the new simulation
    set_nonapoe4_params(rr)
    output_file_nonapoe4 = f'default_simulation_results_{args.years}yr_all_vars_nonapoe4.csv'
    run_simulation(rr, args.years, output_file_nonapoe4)
    df_nonapoe4 = pd.read_csv(output_file_nonapoe4)

    # --- Create Solution and Model Objects ---
    class SimpleModel:
        def __init__(self, species_names, initial_conditions):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
            self.y0 = initial_conditions

    # APOE4 objects
    time_apoe4 = df_apoe4['time'].values
    species_apoe4 = df_apoe4.drop('time', axis=1).values
    model_apoe4 = SimpleModel(df_apoe4.drop('time', axis=1).columns, species_apoe4[0])
    sol_apoe4 = create_solution_object(time_apoe4, species_apoe4)

    # non-APOE4 objects
    time_nonapoe4 = df_nonapoe4['time'].values
    species_nonapoe4 = df_nonapoe4.drop('time', axis=1).values
    model_nonapoe4 = SimpleModel(df_nonapoe4.drop('time', axis=1).columns, species_nonapoe4[0])
    sol_nonapoe4 = create_solution_object(time_nonapoe4, species_nonapoe4)

    # --- Generate Plots ---
    plots_dir = Path("simulation_plots/tellurium_steady_state_apoe_compare")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_six_panel_ab42_comparison(
        sol_apoe4, model_apoe4, sol_nonapoe4, model_nonapoe4,
        drug_type=args.drug, plots_dir=plots_dir
    )

    print(f"APOE4 vs non-APOE4 simulation and comparison plots complete. Plots saved to {plots_dir}") 