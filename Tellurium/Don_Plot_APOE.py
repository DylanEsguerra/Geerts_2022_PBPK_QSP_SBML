import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
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

def plot_four_panel_ab42_apoe4_only(
    sol_apoe4, model_apoe4, drug_type="gantenerumab", plots_dir=None
):
    """Create a four-panel plot for APOE4 only, showing AB42 species with specified colors and styles."""
    import matplotlib.pyplot as plt
    if plots_dir is None:
        plots_dir = Path("simulation_plots/tellurium_steady_state_apoe_compare")
    plots_dir.mkdir(parents=True, exist_ok=True)

    from visualize_tellurium_simulation import calculate_suvr

    # Colors
    blue = '#1f77b4'
    yellow = '#ffb300'

    # Create 2x2 subplot layout, slightly smaller
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Time axis
    x_values = sol_apoe4.ts / 24.0 / 365.0
    start_idx = np.where(x_values >= (max(x_values) - 80))[0][0]
    x_filtered = x_values[start_idx:]

    # APOE4 model indexes
    yidx_a = model_apoe4.y_indexes
    volume_scale_factor_isf = 0.2505

    # 1. Panel 1: AB42 Oligomers (weighted and unweighted loads)
    ab42_oligomer_pattern = re.compile(r'AB42_Oligomer\d+$')
    oligomer_loads_a = {'weighted': [], 'unweighted': []}
    for t in range(len(sol_apoe4.ts)):
        weighted_sum_a = 0.0
        unweighted_sum_a = 0.0
        for name, idx in yidx_a.items():
            if ab42_oligomer_pattern.match(name):
                size = int(name.split('Oligomer')[1])
                concentration = sol_apoe4.ys[t, idx]
                weighted_sum_a += size * concentration
                unweighted_sum_a += concentration
        oligomer_loads_a['weighted'].append(weighted_sum_a / volume_scale_factor_isf)
        oligomer_loads_a['unweighted'].append(unweighted_sum_a / volume_scale_factor_isf)
    # Plot oligomer loads
    ax1.plot(x_filtered, oligomer_loads_a['unweighted'][start_idx:],
             linewidth=3, color=blue, linestyle='-', label='Oligomers')
    ax1.plot(x_filtered, oligomer_loads_a['weighted'][start_idx:],
             linewidth=3, color=yellow, linestyle='-', label='Oligomers weighted')
    ax1.set_ylabel('Concentration', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Time', fontsize=18, fontweight='bold')
    ax1.set_title('Oligomers', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # 2. Panel 2: AB42 Fibrils (weighted and unweighted loads)
    ab42_fibril_pattern = re.compile(r'AB42_Fibril\d+$')
    fibril_loads_a = {'weighted': [], 'unweighted': []}
    for t in range(len(sol_apoe4.ts)):
        weighted_sum_a = 0.0
        unweighted_sum_a = 0.0
        for name, idx in yidx_a.items():
            if ab42_fibril_pattern.match(name):
                size = int(name.split('Fibril')[1])
                concentration = sol_apoe4.ys[t, idx]
                weighted_sum_a += size * concentration
                unweighted_sum_a += concentration
        fibril_loads_a['weighted'].append(weighted_sum_a / volume_scale_factor_isf)
        fibril_loads_a['unweighted'].append(unweighted_sum_a / volume_scale_factor_isf)
    # Plot fibril loads
    ax2.plot(x_filtered, fibril_loads_a['unweighted'][start_idx:],
             linewidth=3, color=blue, linestyle='-', label='Proto')
    ax2.plot(x_filtered, fibril_loads_a['weighted'][start_idx:],
             linewidth=3, color=yellow, linestyle='-', label='Proto weighted')
    ax2.set_ylabel('Concentration', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=18, fontweight='bold')
    ax2.set_title('Proto', fontsize=20, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    # 3. Panel 3: SUVR
    suvr_a = calculate_suvr(sol_apoe4, model_apoe4)[start_idx:]
    ax3.plot(x_filtered, suvr_a, linewidth=3, color=blue, label='SUVR')
    ax3.set_ylabel('Concentration', fontsize=18, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=18, fontweight='bold')
    ax3.set_title('SUVR', fontsize=20, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)

    # 4. Panel 4: AB42 Monomers in ISF and Plaque
    ab42_isf_a = sol_apoe4.ys[:, yidx_a['AB42_Monomer']] / volume_scale_factor_isf
    ab42_plaque_a = sol_apoe4.ys[:, yidx_a['AB42_Plaque_unbound']] / volume_scale_factor_isf
    ax4.plot(x_filtered, ab42_isf_a[start_idx:], linewidth=3, color=blue, linestyle='-', label='AB42 Monomer (ISF)')
    ax4.plot(x_filtered, ab42_plaque_a[start_idx:], linewidth=3, color=yellow, linestyle='-', label='AB42 Plaque')
    ax4.set_ylabel('Concentration', fontsize=18, fontweight='bold')
    ax4.set_xlabel('Time', fontsize=18, fontweight='bold')
    ax4.set_title('AB42_O1_ISF and AB42_O25_ISF', fontsize=20, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    fig.savefig(plots_dir / f'{drug_type.lower()}_four_panel_ab42_apoe4_only.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tellurium simulation for APOE4 only.")
    parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
    args = parser.parse_args()

    xml_path = Path("../generated/sbml/combined_master_model.xml")
    with open(xml_path, "r") as f:
        sbml_str = f.read()

    # Run APOE4 only
    rr = te.loadSBMLModel(sbml_str)
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)
    set_apoe4_params(rr)
    output_file_apoe4 = f'default_simulation_results_{args.years}yr_all_vars_apoe4.csv'
    run_simulation(rr, args.years, output_file_apoe4)

    # Load results
    df_apoe4 = pd.read_csv(output_file_apoe4)

    # Create solution objects directly from the CSV data
    time_apoe4 = df_apoe4['time'].values
    species_apoe4 = df_apoe4.drop('time', axis=1).values

    # Create model objects with species indexes
    class SimpleModel:
        def __init__(self, species_names, initial_conditions):
            self.y_indexes = {name: idx for idx, name in enumerate(species_names)}
            self.y0 = initial_conditions

    model_apoe4 = SimpleModel(df_apoe4.drop('time', axis=1).columns, species_apoe4[0])

    sol_apoe4 = create_solution_object(time_apoe4, species_apoe4)

    plots_dir = Path("simulation_plots/tellurium_steady_state_apoe_compare")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_four_panel_ab42_apoe4_only(
        sol_apoe4, model_apoe4, drug_type=args.drug, plots_dir=plots_dir
    )

    print(f"APOE4 simulation and four-panel AB42 plots complete. Plots saved to {plots_dir}") 