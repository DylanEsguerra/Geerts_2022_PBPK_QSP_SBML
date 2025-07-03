import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Import plotting functions from visualize_tellurium_simulation
from visualize_tellurium_simulation import (
    load_tellurium_data, 
    create_solution_object, 
    create_plots,
    plot_individual_oligomers,
    plot_fibrils_and_plaques,
    plot_ab42_ratios_and_concentrations,
    get_ab42_ratios_and_concentrations_final_values
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run Tellurium simulation with SC event at age > 70")
parser.add_argument("--years", type=float, default=74.0, help="Number of years to simulate")
parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing CSV file, skip simulation")
args = parser.parse_args()

output_file = f'default_simulation_results_{args.years}yr_all_vars.csv'

if not args.plot_only:
    # Load the SBML/Antimony model
    xml_path = Path("../generated/sbml/combined_master_model_antimony.txt")
    with open(xml_path, "r") as f:
        antimony_str = f.read()
    rr = te.loadAntimonyModel(antimony_str)
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-6
    rr.integrator.relative_tolerance = 1e-6
    rr.integrator.setValue('stiff', True)

    # Get all floating species, global parameters, and reactions
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()

    # Simulation settings
    start_time = 0
    end_time = args.years * 365 * 24  # 74 years in hours
    num_points = 300

    # Run simulation
    result = rr.simulate(start_time, end_time, num_points, selections=selections)
    df = pd.DataFrame(result, columns=selections)
    df.to_csv(output_file, index=False)
    print(f'Simulation complete. Results saved to {output_file}')

# Load simulation results
print("Generating plots...")
df = pd.read_csv(output_file)
time_years = df['time'] / (24 * 365)

# Helper: get column if present, else zeros
species = df.columns
get = lambda name: df[name] if name in species else np.zeros_like(time_years)

# 1. Total antibody in brain plasma
pk_p_brain = get('PK_p_brain')
ab40mb_brain_plasma = get('AB40Mb_Brain_Plasma')
ab42mb_brain_plasma = get('AB42Mb_Brain_Plasma')
ab42mu_brain_plasma = get('AB42Mu_Brain_Plasma')
ab40mu_brain_plasma = get('AB40Mu_Brain_Plasma')

total_antibody = pk_p_brain + ab40mb_brain_plasma + ab42mb_brain_plasma

plt.figure(figsize=(10,6))
plt.plot(time_years, total_antibody, label='Total Antibody (PK_p_brain + AB40Mb + AB42Mb)', color='purple')
plt.xlabel('Time (years)')
plt.ylabel('Concentration (nM)')
plt.title('Total Antibody in Brain Plasma')
plt.legend()
plt.tight_layout()
Path('simulation_plots/tellurium_steady_state').mkdir(parents=True, exist_ok=True)
plt.savefig('simulation_plots/tellurium_steady_state/total_antibody_brain_plasma.png', dpi=300)
plt.show()
plt.close()

# 2. Total Abeta40 in brain plasma
abeta40 = ab40mb_brain_plasma + ab40mu_brain_plasma
plt.figure(figsize=(10,6))
plt.plot(time_years, abeta40, label='Total Abeta40 (AB40Mb + AB40Mu)', color='blue')
plt.xlabel('Time (years)')
plt.ylabel('Concentration (nM)')
plt.title('Total Abeta40 in Brain Plasma')
plt.legend()
plt.tight_layout()
plt.savefig('simulation_plots/tellurium_steady_state/total_abeta40_brain_plasma.png', dpi=300)
plt.show()
plt.close()

# 3. Total Abeta42 in brain plasma
ab42mb_brain_plasma = get('AB42Mb_Brain_Plasma')
ab42mu_brain_plasma = get('AB42Mu_Brain_Plasma')
abeta42 = ab42mb_brain_plasma + ab42mu_brain_plasma
plt.figure(figsize=(10,6))
plt.plot(time_years, abeta42, label='Total Abeta42 (AB42Mb + AB42Mu)', color='red')
plt.xlabel('Time (years)')
plt.ylabel('Concentration (nM)')
plt.title('Total Abeta42 in Brain Plasma')
plt.legend()
plt.tight_layout()
plt.savefig('simulation_plots/tellurium_steady_state/total_abeta42_brain_plasma.png', dpi=300)
plt.show()
plt.close()

print("All requested plots generated and saved.")

# Generate plots using visualize_tellurium_simulation functions
print("Generating plots...")

try:
    # Load the simulation data
    time_points, species_data, model = load_tellurium_data(years=args.years)
    sol = create_solution_object(time_points, species_data)
    
    # Create output directory for plots
    plots_dir = Path("simulation_plots/tellurium_steady_state")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Get and print final values
    final_values = get_ab42_ratios_and_concentrations_final_values(sol, model)
    if final_values:
        print("\nFinal AB42/AB40 Ratios and Concentrations:")
        print(f"Brain Plasma AB42/AB40 Ratio: {final_values['brain_plasma_ratio']:.4f}")
        print(f"Total ISF AB42: {final_values['ab42_isf_pg_ml']:.2f} pg/mL")
        print(f"CSF SAS AB42: {final_values['ab42_sas_pg_ml']:.2f} pg/mL")
        print(f"Total ISF AB40: {final_values['ab40_isf_pg_ml']:.2f} pg/mL")
        print(f"CSF SAS AB40: {final_values['ab40_sas_pg_ml']:.2f} pg/mL")
    
    
except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc() 