"""
Model Comparison Script: Dylan_LibSBML vs Don_Antimony_PBPK

This script compares two different implementations of the Geerts model:

1. Dylan_LibSBML Model (Combined Master Model):
   - Generated using libSBML and sbmltoodejax
   - IDE clearance implemented as a time-varying parameter (CL_AB42_IDE)
   - Uses a rate rule for parameter evolution of CL_AB42_IDE
   - Requires modification to include additional volume multiplication in clearance terms in order to match Don's model

2. Don_Antimony_PBPK Model:
   - Written directly in Antimony syntax
   - IDE clearance implemented as a species (IDE_activity_ISF)
   - Uses a reaction for species evolution of IDE_activity_ISF
   - Different volume handling in clearance terms

KEY DIFFERENCES:
- IDE clearance: Parameter vs Species implementation
- Volume handling: Don's model appears to multiply the clearance by the volume of the ISF compartment
- Model generation: libSBML vs direct Antimony

COMPARISON RESULTS:
- Antimony models (txt files): Models match when CL_AB42_IDE = 400
- SBML models (xml files): Models match when CL_AB42_IDE = 100.2
    - Note: The SBML version is the original version from Dylan's model. 
    - The Antimony version is the modified version that matches Don's model.

USAGE:
    python compare_models_combined.py [--years YEARS] [--mode {antimony,sbml}]
    --mode antimony: Compare using Antimony text files (CL_AB42_IDE = 400)
    --mode sbml: Compare using SBML XML files (CL_AB42_IDE = 100.2)
"""

import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
import re
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

# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


def run_simulation_dylan_model(rr, years, output_file):
    """Run simulation for Dylan_LibSBML model (combined master model)."""
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()
    start_time = 0
    end_time = years * 365 * 24
    num_points = 300
    
    
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
    # CL_AB42_IDE and exp_decline_rate_IDE_fortytwo are now set in the main script based on mode
    #rr.setValue('CL_AB42_IDE', 400) # 400 * 0.2505 = 100.2 ISE_activity_ISF
    #rr.setValue('CL_AB42_IDE', 100.2) # 400 * 0.2505 = 100.2
    # rr.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05) # For Antimony
    # rr.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05*0.2525) # For SBML
    #'''
    
    
    result = rr.simulate(start_time, end_time, num_points, selections=selections)

    df = pd.DataFrame(result, columns=selections)
    df.to_csv(output_file, index=False)
    return output_file

def run_simulation_don_model(rr, years, output_file):
    """Run simulation for Don_Antimony_PBPK model."""
    start_time = 0
    end_time = years * 365 * 24
    num_points = 300
    
    # Include all AB42 oligomer species
    selections = ['time', '[AB42_O1_ISF]', '[AB42_O25_ISF]', '[IDE_activity_ISF]']
    for i in range(2, 25):  # O2 through O24
        selections.append(f'[AB42_O{i}_ISF]')
    #rr.setValue('IDE_conc_ISF', 3 / 0.2505)
    result = rr.simulate(start_time, end_time, num_points, selections)
    
    # Convert to DataFrame for compatibility
    df = pd.DataFrame(result, columns=result.colnames)
    df.to_csv(output_file, index=False)
    return result

def suvr_formula(oligo, proto, plaque, C1=2.5, C2=400000, C3=1.3, Hill=3.5):
    """Calculate SUVR using the provided formula."""
    numerator = oligo + proto + C3 * 24.0 * plaque
    denominator = numerator**Hill + C2**Hill
    
    # Handle scalar values properly
    if hasattr(denominator, '__len__'):
        # If it's an array
        if np.any(denominator == 0):
            denominator = np.where(denominator == 0, 1.0, denominator)
    else:
        # If it's a scalar
        if denominator == 0:
            denominator = 1.0
            
    suvr = 1.0 + C1 * (numerator**Hill) / denominator
    return suvr

def calculate_oligomer_weighted_sum_don(result):
    """Calculate oligomer weighted sum (O2-O16) for Don_Antimony_PBPK model."""
    oligomer_weighted_sum = result['[AB42_O2_ISF]'] * 1  # O2 weighted by (2-1)=1
    for i in range(3, 17):  # O3 through O16
        oligomer_weighted_sum += result[f'[AB42_O{i}_ISF]'] * (i-1)
    return oligomer_weighted_sum

def calculate_proto_weighted_sum_don(result):
    """Calculate protofibril weighted sum (O17-O24) for Don_Antimony_PBPK model."""
    proto_weighted_sum = result['[AB42_O17_ISF]'] * 16  # O17 weighted by (17-1)=16
    for i in range(18, 25):  # O18 through O24
        proto_weighted_sum += result[f'[AB42_O{i}_ISF]'] * (i-1)
    return proto_weighted_sum

def calculate_suvr_don(result):
    """Calculate SUVR for Don_Antimony_PBPK model."""
    # Calculate oligomers (O2-O16) using new weighting
    oligo = calculate_oligomer_weighted_sum_don(result)
    
    # Calculate protofibrils (O17-O24) using new weighting
    proto = calculate_proto_weighted_sum_don(result)
    
    # Calculate plaques (O25) - no weighting needed
    plaque = result['[AB42_O25_ISF]']
    
    # Use the SUVR formula (no volume scaling)
    suvr = suvr_formula(oligo, proto, plaque)
    return suvr

def plot_six_panel_model_comparison(
    sol_dylan, model_dylan, result_don, drug_type="gantenerumab", plots_dir=None, mode="sbml"
):
    """Create a six-panel plot comparing Dylan_LibSBML and Don_Antimony_PBPK model simulations."""
    if plots_dir is None:
        plots_dir = Path("simulation_plots/model_comparison")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Colors
    red = '#d62728'    # Dylan_LibSBML model
    blue = '#1f77b4'   # Don_Antimony_PBPK model

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 10))

    # --- Process Dylan_LibSBML Model Data ---
    x_values_dylan = sol_dylan.ts / 24.0 / 365.0
    start_idx_dylan = np.where(x_values_dylan >= (max(x_values_dylan) - 80))[0][0]
    x_filtered_dylan = x_values_dylan[start_idx_dylan:]
    yidx_dylan = model_dylan.y_indexes
    volume_scale_factor_isf = 0.2505

    # --- Process Don_Antimony_PBPK Model Data ---
    time_years_don = result_don['time'] / 24.0 / 365.0
    start_idx_don = np.where(time_years_don >= (max(time_years_don) - 80))[0][0]
    x_filtered_don = time_years_don[start_idx_don:]

    # Panel 1: Oligomers (Top Left)
    # Dylan_LibSBML model: weighted oligomers
    oligomer_loads_dylan = np.array([np.sum([(int(name.split('Oligomer')[1])-1) * sol_dylan.ys[t, idx] for name, idx in yidx_dylan.items() if re.match(r'AB42_Oligomer\d+$', name)]) for t in range(len(sol_dylan.ts))]) / volume_scale_factor_isf
    
    # Don_Antimony_PBPK model: weighted oligomers (O2-O16)
    oligomer_weighted_sum_don = calculate_oligomer_weighted_sum_don(result_don)
    
    ax1.plot(x_filtered_dylan, oligomer_loads_dylan[start_idx_dylan:], linewidth=3, color=red, label="Dylan_LibSBML Model Oligomers")
    ax1.plot(x_filtered_don, oligomer_weighted_sum_don[start_idx_don:], linewidth=3, color=blue, label='Don_Antimony_PBPK Model Oligomers')
    ax1.set_title('Weighted Oligomers', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Oligomer Load (nM)', fontsize=14, fontweight='bold')

    # Panel 2: Protofibrils (Top Right)
    # Dylan_LibSBML model: weighted fibrils
    fibril_loads_dylan = np.array([np.sum([(int(name.split('Fibril')[1])-1) * sol_dylan.ys[t, idx] for name, idx in yidx_dylan.items() if re.match(r'AB42_Fibril\d+$', name)]) for t in range(len(sol_dylan.ts))]) / volume_scale_factor_isf
    
    # Don_Antimony_PBPK model: weighted protofibrils (O17-O24)
    protofibril_weighted_sum_don = calculate_proto_weighted_sum_don(result_don)
    
    ax2.plot(x_filtered_dylan, fibril_loads_dylan[start_idx_dylan:], linewidth=3, color=red, label="Dylan_LibSBML Model Protofibrils")
    ax2.plot(x_filtered_don, protofibril_weighted_sum_don[start_idx_don:], linewidth=3, color=blue, label='Don_Antimony_PBPK Model Protofibrils')
    ax2.set_title('Weighted Protofibrils', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Protofibril Load (nM)', fontsize=14, fontweight='bold')

    # Panel 3: SUVR (Middle Left)
    # Dylan_LibSBML model: using existing calculate_suvr function
    suvr_dylan = calculate_suvr(sol_dylan, model_dylan)[start_idx_dylan:]
    
    # Don_Antimony_PBPK model: using new SUVR calculation
    suvr_don = calculate_suvr_don(result_don)[start_idx_don:]
    
    ax3.plot(x_filtered_dylan, suvr_dylan, linewidth=3, color=red, label="Dylan_LibSBML Model SUVR")
    ax3.plot(x_filtered_don, suvr_don, linewidth=3, color=blue, label='Don_Antimony_PBPK Model SUVR')
    ax3.set_title('SUVR', fontsize=16, fontweight='bold')
    ax3.set_ylabel('SUVR', fontsize=14, fontweight='bold')

    # Panel 4: AB42 Monomer (Middle Right)
    # Dylan_LibSBML model: AB42_Monomer
    monomer_dylan = sol_dylan.ys[start_idx_dylan:, yidx_dylan['AB42_Monomer']] / volume_scale_factor_isf
    
    # Don_Antimony_PBPK model: AB42_O1_ISF
    if '[AB42_O1_ISF]' in result_don.colnames:
        monomer_don = result_don['[AB42_O1_ISF]'][start_idx_don:]
        ax4.plot(x_filtered_dylan, monomer_dylan, linewidth=3, color=red, label="Dylan_LibSBML Model Monomer")
        ax4.plot(x_filtered_don, monomer_don, linewidth=3, color=blue, label='Don_Antimony_PBPK Model Monomer')
        ax4.set_title('AB42 Monomer (ISF)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('AB42 Monomer (nM)', fontsize=14, fontweight='bold')
    else:
        ax4.plot(x_filtered_dylan, monomer_dylan, linewidth=3, color=red, label="Dylan_LibSBML Model Monomer")
        ax4.set_title('AB42 Monomer (ISF)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('AB42 Monomer (nM)', fontsize=14, fontweight='bold')

    # Panel 5: IDE Activity/Clearance (Bottom Left)
    # Dylan_LibSBML model: CL_AB42_IDE
    if 'CL_AB42_IDE' in yidx_dylan:
        cl_ide_dylan = sol_dylan.ys[start_idx_dylan:, yidx_dylan['CL_AB42_IDE']]/volume_scale_factor_isf
        ax5.plot(x_filtered_dylan, cl_ide_dylan, linewidth=3, color=red, label="Dylan_LibSBML Model IDE Clearance")
        ax5.set_ylabel('IDE Clearance (1/h)', fontsize=14, fontweight='bold')
    else:
        ax5.set_ylabel('IDE Activity', fontsize=14, fontweight='bold')
    
    # Don_Antimony_PBPK model: IDE_activity_ISF
    if '[IDE_activity_ISF]' in result_don.colnames:
        ide_data_don = result_don['[IDE_activity_ISF]'][start_idx_don:]
        ax5.plot(x_filtered_don, ide_data_don, linewidth=3, color=blue, label='Don_Antimony_PBPK Model IDE Activity')
        ax5.set_title('IDE Activity/Clearance', fontsize=16, fontweight='bold')
    else:
        ax5.set_title('IDE Activity/Clearance', fontsize=16, fontweight='bold')

    # Panel 6: AB42 Plaque (Bottom Right)
    # Dylan_LibSBML model: AB42_Plaque_unbound
    if 'AB42_Plaque_unbound' in yidx_dylan:
        plaque_dylan = sol_dylan.ys[start_idx_dylan:, yidx_dylan['AB42_Plaque_unbound']] / volume_scale_factor_isf
        ax6.plot(x_filtered_dylan, plaque_dylan, linewidth=3, color=red, label="Dylan_LibSBML Model Plaque")
        ax6.set_ylabel('Plaque Load (nM)', fontsize=14, fontweight='bold')
    else:
        ax6.set_ylabel('Plaque Load (nM)', fontsize=14, fontweight='bold')
    
    # Don_Antimony_PBPK model: AB42_O25_ISF
    if '[AB42_O25_ISF]' in result_don.colnames:
        plaque_data_don = result_don['[AB42_O25_ISF]'][start_idx_don:]
        if np.any(plaque_data_don > 0):  # Check if we have plaque data
            ax6.plot(x_filtered_don, plaque_data_don, linewidth=3, color=blue, label='Don_Antimony_PBPK Model Plaque')
            ax6.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold')
        else:
            ax6.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold')
    else:
        ax6.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold')

    # Set x-axis labels and other formatting for all panels
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    fig.savefig(plots_dir / f'{drug_type.lower()}_model_comparison_six_panel_{mode}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tellurium simulation comparing Dylan_LibSBML and Don_Antimony_PBPK models.")
    parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
    parser.add_argument("--mode", type=str, choices=["antimony", "sbml"], default="sbml", 
                      help="Comparison mode: 'antimony' for txt files (CL_AB42_IDE=400), 'sbml' for xml files (CL_AB42_IDE=100.2)")
    args = parser.parse_args()

    print(f"Running comparison in {args.mode.upper()} mode")
    if args.mode == "antimony":
        print("Using Antimony text files - CL_AB42_IDE will be set to 400")
    else:
        print("Using SBML XML files - CL_AB42_IDE will be set to 100.2")

    # Load Dylan_LibSBML model (combined master model)
    if args.mode == "antimony":
        antimony_path_dylan = Path("../generated/sbml/combined_master_model_MTK_Microglia2_Antimony.txt")
        with open(antimony_path_dylan, "r") as f:
            antimony_str_dylan = f.read()
        sbml_str_dylan = None
    else:
        sbml_path_dylan = Path("../generated/sbml/combined_master_model_gantenerumab.xml")
        with open(sbml_path_dylan, "r") as f:
            sbml_str_dylan = f.read()
        antimony_str_dylan = None

    # Load Don_Antimony_PBPK model
    if args.mode == "antimony":
        #antimony_path_don = Path("../generated/sbml/Antimony_PBPK_model.txt")
        antimony_path_don = Path("Antimony_Geerts_model_opt_flexible6a.txt")
        with open(antimony_path_don, "r") as f:
            antimony_str_don = f.read()
        sbml_str_don = None
    else:
        sbml_path_don = Path("../generated/sbml/Antimony_PBPK_model.xml")
        with open(sbml_path_don, "r") as f:
            sbml_str_don = f.read()
        antimony_str_don = None

    # --- Run Dylan_LibSBML Model Simulation ---
    print("Running Dylan_LibSBML model simulation...")
    if args.mode == "antimony":
        rr_dylan = te.loadAntimonyModel(antimony_str_dylan)
    else:
        rr_dylan = te.loadSBMLModel(sbml_str_dylan)
    rr_dylan.setIntegrator('cvode')
    rr_dylan.integrator.absolute_tolerance = 1e-8
    rr_dylan.integrator.relative_tolerance = 1e-8
    rr_dylan.integrator.setValue('stiff', True)
    
    # Set appropriate CL_AB42_IDE value based on mode
    if args.mode == "antimony":
        rr_dylan.setValue('CL_AB42_IDE', 400)  # For Antimony comparison
    else:
        rr_dylan.setValue('CL_AB42_IDE', 100.2)  # For SBML comparison
    
    # Set appropriate exp_decline_rate_IDE_fortytwo value based on mode
    # For SBML: use volume-scaled value (1.15E-05*0.2525) because volume multiplication was removed from the decay
    # For Antimony: use unscaled value (1.15E-05) because volume term is still in the rate itself
    if args.mode == "antimony":
        rr_dylan.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05)  # For Antimony comparison
    else:
        rr_dylan.setValue('exp_decline_rate_IDE_fortytwo', 1.15E-05*0.2525)  # For SBML comparison
    
    output_file_dylan = f'dylan_libsbml_model_simulation_results_{args.years}yr_{args.mode}.csv'
    run_simulation_dylan_model(rr_dylan, args.years, output_file_dylan)
    df_dylan = pd.read_csv(output_file_dylan)

    # --- Run Don_Antimony_PBPK Model Simulation ---
    print("Running Don_Antimony_PBPK model simulation...")
    if args.mode == "antimony":
        rr_don = te.loadAntimonyModel(antimony_str_don)
    else:
        rr_don = te.loadSBMLModel(sbml_str_don)
    rr_don.setIntegrator('cvode')
    rr_don.integrator.absolute_tolerance = 1e-8
    rr_don.integrator.relative_tolerance = 1e-8
    rr_don.integrator.setValue('stiff', True)
    
    output_file_don = f'don_antimony_pbpk_model_simulation_results_{args.years}yr_{args.mode}.csv'
    result_don = run_simulation_don_model(rr_don, args.years, output_file_don)

    # --- Create Solution and Model Objects for Dylan_LibSBML Model ---
    class SimpleModel:
        def __init__(self, species_names, initial_conditions):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
            self.y0 = initial_conditions

    # Dylan_LibSBML model objects
    time_dylan = df_dylan['time'].values
    species_dylan = df_dylan.drop('time', axis=1).values
    model_dylan = SimpleModel(df_dylan.drop('time', axis=1).columns, species_dylan[0])
    sol_dylan = create_solution_object(time_dylan, species_dylan)

    # --- Generate Comparison Plots ---
    plots_dir = Path("simulation_plots/model_comparison")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_six_panel_model_comparison(
        sol_dylan, model_dylan, result_don,
        drug_type=args.drug, plots_dir=plots_dir, mode=args.mode
    )

    print(f"Model comparison simulation and plots complete ({args.mode.upper()} mode).")
    print(f"Plots saved to {plots_dir}")
    print(f"Dylan_LibSBML model results: {output_file_dylan}")
    print(f"Don_Antimony_PBPK model results: {output_file_don}")
    if args.mode == "antimony":
        print("Note: CL_AB42_IDE was set to 400 for Antimony comparison")
        print("Note: exp_decline_rate_IDE_fortytwo was set to 1.15E-05 (unscaled) for Antimony comparison")
    else:
        print("Note: CL_AB42_IDE was set to 100.2 for SBML comparison")
        print("Note: exp_decline_rate_IDE_fortytwo was set to 1.15E-05*0.2525 (volume-scaled) for SBML comparison") 