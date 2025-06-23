import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run default Tellurium simulation")
parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
args = parser.parse_args()

# Load the SBML model
xml_path = Path("../generated/sbml/combined_master_model.xml")
with open(xml_path, "r") as f:
    sbml_str = f.read()

# Load the model
rr = te.loadSBMLModel(sbml_str)
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-10
rr.integrator.relative_tolerance = 1e-10
rr.integrator.setValue('stiff', True)

rr.CL_AB42_IDE = 50
rr.AB42_IDE_Kcat_exp = 50

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
output_file = f'default_simulation_results_{args.years}yr_all_vars.csv'
df.to_csv(output_file, index=False)

print(f'Simulation complete. Results saved to {output_file}') 