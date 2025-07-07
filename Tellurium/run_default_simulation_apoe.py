import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from visualize_tellurium_simulation import (
    create_solution_object
)

def run_simulation(rr, years, output_file):
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()
    start_time = 0
    end_time = years * 365 * 24
    num_points = 300
    result = rr.simulate(start_time, end_time, num_points, selections=selections)
    df = pd.DataFrame(result, columns=selections)
    df.to_csv(output_file, index=False)
    return output_file

def set_apoe4_params(rr):
    # Set APOE4 values for EC50 and Vmax
    rr.setValue('Microglia_EC50_forty', 20)
    rr.setValue('Microglia_EC50_fortytwo', 300)
    rr.setValue('Microglia_Vmax_forty', 0.0001)
    rr.setValue('Microglia_Vmax_fortytwo', 0.0001)

def set_nonapoe4_params(rr):
    # Set Non-APOE4 values for EC50 and Vmax
    rr.setValue('Microglia_EC50_forty', 8)
    rr.setValue('Microglia_EC50_fortytwo', 120)
    rr.setValue('Microglia_Vmax_forty', 0.00015)
    rr.setValue('Microglia_Vmax_fortytwo', 0.00015)

def plot_ab42_ratios_and_concentrations_overlay(
    sol_nonapoe4, model_nonapoe4, sol_apoe4, model_apoe4, drug_type="gantenerumab", plots_dir=None
):
    import matplotlib.pyplot as plt
    if plots_dir is None:
        plots_dir = Path("simulation_plots/tellurium_steady_state_apoe_compare")
    plots_dir.mkdir(parents=True, exist_ok=True)

    from visualize_tellurium_simulation import calculate_suvr

    # Create 2x2 subplot layout: SUVR, ratio, ISF, CSF
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Time axis
    x_values = sol_nonapoe4.ts / 24.0 / 365.0
    start_idx = np.where(x_values >= (max(x_values) - 80))[0][0]
    x_filtered = x_values[start_idx:]

    # Non-APOE4
    yidx_n = model_nonapoe4.y_indexes
    nM_to_pg = 4514.0
    volume_scale_factor_csf = 0.09875
    volume_scale_factor_isf = 0.2505

    # APOE4
    yidx_a = model_apoe4.y_indexes

    # 1. SUVR
    suvr_n = calculate_suvr(sol_nonapoe4, model_nonapoe4)[start_idx:]
    suvr_a = calculate_suvr(sol_apoe4, model_apoe4)[start_idx:]
    ax1.plot(x_filtered, suvr_n, linewidth=3, color='blue', label='SUVR Non-APOE4')
    ax1.plot(x_filtered, suvr_a, linewidth=3, color='red', label='SUVR APOE4')
    ax1.set_ylabel('SUVR', fontsize=26)
    ax1.set_xlabel('Time (years)', fontsize=26)
    ax1.set_title('SUVR Progression', fontsize=30)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=22)

    # 2. Brain plasma ratios
    ab42_bp_n = sol_nonapoe4.ys[:, yidx_n['AB42Mu_Brain_Plasma']]
    ab40_bp_n = sol_nonapoe4.ys[:, yidx_n['AB40Mu_Brain_Plasma']]
    ab42_bp_a = sol_apoe4.ys[:, yidx_a['AB42Mu_Brain_Plasma']]
    ab40_bp_a = sol_apoe4.ys[:, yidx_a['AB40Mu_Brain_Plasma']]
    ratio_n = np.where(ab40_bp_n > 0, ab42_bp_n / ab40_bp_n, 0)[start_idx:]
    ratio_a = np.where(ab40_bp_a > 0, ab42_bp_a / ab40_bp_a, 0)[start_idx:]
    ax2.plot(x_filtered, ratio_n, linewidth=2, color='blue', label='Non-APOE4')
    ax2.plot(x_filtered, ratio_a, linewidth=2, color='red', label='APOE4')
    ax2.set_ylabel('Ratio', fontsize=26)
    ax2.set_xlabel('Time (years)', fontsize=26)
    ax2.set_title('Brain Plasma AB42/AB40 Ratio', fontsize=30)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=22)

    # 3. ISF AB42
    ab42_isf_n = sol_nonapoe4.ys[:, yidx_n['AB42_Monomer']] / volume_scale_factor_isf * nM_to_pg
    ab42_isf_a = sol_apoe4.ys[:, yidx_a['AB42_Monomer']] / volume_scale_factor_isf * nM_to_pg
    ax3.semilogy(x_filtered, ab42_isf_n[start_idx:], linewidth=2, color='blue', label='Non-APOE4')
    ax3.semilogy(x_filtered, ab42_isf_a[start_idx:], linewidth=2, color='red', label='APOE4')
    ax3.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax3.set_xlabel('Time (years)', fontsize=26)
    ax3.set_title('Total ISF AB42', fontsize=30)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=22)

    # 4. CSF SAS AB42
    ab42_sas_n = sol_nonapoe4.ys[:, yidx_n['AB42Mu_SAS']] / volume_scale_factor_csf * nM_to_pg
    ab42_sas_a = sol_apoe4.ys[:, yidx_a['AB42Mu_SAS']] / volume_scale_factor_csf * nM_to_pg
    ax4.plot(x_filtered, ab42_sas_n[start_idx:], linewidth=2, color='blue', label='Non-APOE4')
    ax4.plot(x_filtered, ab42_sas_a[start_idx:], linewidth=2, color='red', label='APOE4')
    ax4.set_ylabel('Concentration (pg/mL)', fontsize=26)
    ax4.set_xlabel('Time (years)', fontsize=26)
    ax4.set_title('CSF SAS AB42', fontsize=30)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=22)

    plt.tight_layout()
    fig.savefig(plots_dir / f'{drug_type.lower()}_ab42_ratios_and_concentrations_apoe_compare.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tellurium simulation for Non-APOE4 and APOE4.")
    parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
    args = parser.parse_args()

    xml_path = Path("../generated/sbml/combined_master_model.xml")
    with open(xml_path, "r") as f:
        sbml_str = f.read()

    # Run Non-APOE4
    rr = te.loadSBMLModel(sbml_str)
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)
    set_nonapoe4_params(rr)
    output_file_nonapoe4 = f'default_simulation_results_{args.years}yr_all_vars_nonapoe4.csv'
    run_simulation(rr, args.years, output_file_nonapoe4)

    # Run APOE4
    rr.reset()
    set_apoe4_params(rr)
    output_file_apoe4 = f'default_simulation_results_{args.years}yr_all_vars_apoe4.csv'
    run_simulation(rr, args.years, output_file_apoe4)

    # Load results
    df_nonapoe4 = pd.read_csv(output_file_nonapoe4)
    df_apoe4 = pd.read_csv(output_file_apoe4)

    # Create solution objects directly from the CSV data
    time_nonapoe4 = df_nonapoe4['time'].values
    species_nonapoe4 = df_nonapoe4.drop('time', axis=1).values
    time_apoe4 = df_apoe4['time'].values
    species_apoe4 = df_apoe4.drop('time', axis=1).values

    # Create model objects with species indexes
    class SimpleModel:
        def __init__(self, species_names, initial_conditions):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
            self.y0 = initial_conditions

    model_nonapoe4 = SimpleModel(df_nonapoe4.drop('time', axis=1).columns, species_nonapoe4[0])
    model_apoe4 = SimpleModel(df_apoe4.drop('time', axis=1).columns, species_apoe4[0])

    sol_nonapoe4 = create_solution_object(time_nonapoe4, species_nonapoe4)
    sol_apoe4 = create_solution_object(time_apoe4, species_apoe4)

    plots_dir = Path("simulation_plots/tellurium_steady_state_apoe_compare")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_ab42_ratios_and_concentrations_overlay(
        sol_nonapoe4, model_nonapoe4, sol_apoe4, model_apoe4, drug_type=args.drug, plots_dir=plots_dir
    )

    print(f"APOE4 and Non-APOE4 simulation and overlay plots complete. Plots saved to {plots_dir}") 