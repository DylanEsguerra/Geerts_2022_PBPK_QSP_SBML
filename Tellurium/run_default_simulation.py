import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

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
parser = argparse.ArgumentParser(description="Run default Tellurium simulation")
parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
parser.add_argument("--plot-only", action="store_true", help="Only generate plots from existing CSV file, skip simulation")
args = parser.parse_args()

# Define output file path
output_file = f'default_simulation_results_{args.years}yr_all_vars.csv'

if not args.plot_only:
    # Load the SBML model
    xml_path = Path("../generated/sbml/combined_master_model.xml")
    with open(xml_path, "r") as f:
        sbml_str = f.read()

    # Load the model
    rr = te.loadSBMLModel(sbml_str)
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)

    

    # Get all floating species, global parameters, and reactions
    selections = ['time'] + rr.getFloatingSpeciesIds() + rr.getGlobalParameterIds() + rr.getReactionIds()

    # Simulation settings
    start_time = 0
    end_time = args.years * 365 * 24  # Convert years to hours
    num_points = 300

    # Run simulation
    result = rr.simulate(start_time, end_time, num_points, selections=selections)

    # Save to CSV
    df = pd.DataFrame(result, columns=selections)
    df.to_csv(output_file, index=False)

    print(f'Simulation complete. Results saved to {output_file}')

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
    
    # Generate all plots
    create_plots(sol, model)
    plot_individual_oligomers(sol, model, drug_type=args.drug, plots_dir=plots_dir)
    plot_fibrils_and_plaques(sol, model, drug_type=args.drug, plots_dir=plots_dir)
    plot_ab42_ratios_and_concentrations(sol, model, drug_type=args.drug, plots_dir=plots_dir)
    
    print(f"\nPlots saved to {plots_dir}")
    print("All plots generated successfully!")
    
except Exception as e:
    print(f"Error generating plots: {e}")
    import traceback
    traceback.print_exc() 