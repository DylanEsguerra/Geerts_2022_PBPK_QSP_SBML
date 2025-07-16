import tellurium as te
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from K_rates_extrapolate_Don import calculate_k_rates

def set_optimized_values(rr):
    """
    Set optimized parameter values that are common to both APOE4 and non-APOE4 models.
    Returns a list of any parameters that couldn't be set.
    """
    failed_params = []
    
    # Define the optimized parameter values
    optimized_params = {
        'IDE_conc_ISF': 7.208542043204368,
        'Microglia': 0.6158207628681558,
        'k_APP_production': 125.39958281853755,
        'k_O24_O12_AB42_ISF': 8.411308100233814,
        'k_O1_O2_AB42_ISF': 0.0011969797316319238,
        'k_O2_O3_AB42_ISF': 0.0009995988649416513,
        'k_O2_O1_AB42_ISF': 71.27022085388667,
        'k_O3_O2_AB42_ISF': 5.696488107534822e-07,
        'Baseline_AB42_O_P': 0.0443230279057089,
        }
    
    # Set the basic optimized parameters
    for param_name, value in optimized_params.items():
        try:
            rr.setValue(param_name, value)
            print(f"Set {param_name} to {value}")
        except Exception as e:
            error_msg = f"Could not set {param_name}: {e}"
            print(error_msg)
            failed_params.append(error_msg)
    
    rates = calculate_k_rates(rr['k_O1_O2_AB42_ISF'], rr['k_O2_O3_AB42_ISF'], rr['k_O2_O1_AB42_ISF'], rr['k_O3_O2_AB42_ISF'])
    oligomer_sizes = list(range(4, 25))
    for i, size in enumerate(oligomer_sizes):
        try: 
            rr.setValue(f'k_O{size-1}_O{size}_AB42_ISF', rates[f'k_O{size-1}_O{size}_AB42_ISF'])
            rr.setValue(f'k_O{size}_O{size-1}_AB42_ISF', rates[f'k_O{size}_O{size-1}_AB42_ISF'])
        except Exception as e:
            error_msg = f"Could not set {f'k_O{size-1}_O{size}_AB42_ISF'}: {e}"
            print(error_msg)
            failed_params.append(error_msg)
    
    return failed_params

def run_simulation(rr, years, output_file):
    """Run simulation and save results to CSV."""
    # Run simulation with specific species selections to match new approach
    start_time = 0
    end_time = years * 365 * 24
    num_points = 300
    
    # Include all AB42 oligomer species like in the reference code
    selections = ['time', '[AB42_O1_ISF]', '[AB42_O25_ISF]', '[IDE_activity_ISF]']
    for i in range(2, 25):  # O2 through O24
        selections.append(f'[AB42_O{i}_ISF]')
    
    result = rr.simulate(start_time, end_time, num_points, selections)
    
    # Convert to DataFrame for compatibility
    df = pd.DataFrame(result, columns=result.colnames)
    df.to_csv(output_file, index=False)
    return result

def suvr_formula(oligo, proto, plaque, C1=2.5, C2=400000, C3=1.3, Hill=3.5):
    """
    Calculate SUVR using the provided formula.
    
    Parameters:
    oligo, proto, plaque: input oligomer values (arrays or scalars)
    C1, C2, C3, Hill: constants from the formula
    
    Returns:
    suvr: predicted SUVR value(s)
    """
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

def calculate_oligomer_weighted_sum(result):
    """Calculate oligomer weighted sum (O2-O17) using new method."""
    oligomer_weighted_sum = result['[AB42_O2_ISF]'] * 1  # O2 weighted by (2-1)=1
    for i in range(3, 18):  # O3 through O17
        oligomer_weighted_sum += result[f'[AB42_O{i}_ISF]'] * (i-1)
    return oligomer_weighted_sum

def calculate_proto_weighted_sum(result):
    """Calculate protofibril weighted sum (O18-O24) using new method."""
    proto_weighted_sum = result['[AB42_O18_ISF]'] * 17  # O18 weighted by (18-1)=17
    for i in range(19, 25):  # O19 through O24
        proto_weighted_sum += result[f'[AB42_O{i}_ISF]'] * (i-1)
    return proto_weighted_sum

def calculate_suvr_antimony(result):
    """Calculate SUVR for Antimony model using the new formula and approach."""
    # Calculate oligomers (O2-O17) using new weighting
    oligo = calculate_oligomer_weighted_sum(result)
    
    # Calculate protofibrils (O18-O24) using new weighting
    proto = calculate_proto_weighted_sum(result)
    
    # Calculate plaques (O25) - no weighting needed
    plaque = result['[AB42_O25_ISF]']
    
    # Use the SUVR formula (no volume scaling)
    suvr = suvr_formula(oligo, proto, plaque)
    return suvr

def plot_six_panel_antimony(result, drug_type="gantenerumab", plots_dir=None):
    """Create a six-panel plot for the Antimony model simulation using new approach."""
    if plots_dir is None:
        plots_dir = Path("simulation_plots/antimony_results")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Colors
    blue = '#1f77b4'

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 10))

    # Get time in years
    time_years = result['time'] / 24.0 / 365.0
    start_idx = np.where(time_years >= (max(time_years) - 80))[0][0]
    x_filtered = time_years[start_idx:]

    # Panel 1: Oligomers (Top Left) - weighted oligomers (O2-O17)
    oligomer_weighted_sum = calculate_oligomer_weighted_sum(result)
    ax1.plot(x_filtered, oligomer_weighted_sum[start_idx:], linewidth=3, color=blue, label='AB42 Oligomers')
    ax1.set_title('Weighted Oligomers', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Oligomer Load (nM)', fontsize=14, fontweight='bold')

    # Panel 2: Protofibrils (Top Right) - weighted protofibrils (O18-O24)
    protofibril_weighted_sum = calculate_proto_weighted_sum(result)
    ax2.plot(x_filtered, protofibril_weighted_sum[start_idx:], linewidth=3, color=blue, label='AB42 Protofibrils')
    ax2.set_title('Weighted Protofibrils', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Protofibril Load (nM)', fontsize=14, fontweight='bold')

    # Panel 3: SUVR (Middle Left) - using new formula
    suvr = calculate_suvr_antimony(result)[start_idx:]
    ax3.plot(x_filtered, suvr, linewidth=3, color=blue, label='SUVR')
    ax3.set_title('SUVR', fontsize=16, fontweight='bold')
    ax3.set_ylabel('SUVR', fontsize=14, fontweight='bold')

    # Panel 4: AB42 Monomer (Middle Right) - no volume scaling
    if '[AB42_O1_ISF]' in result.colnames:
        monomer = result['[AB42_O1_ISF]'][start_idx:]
        ax4.plot(x_filtered, monomer, linewidth=3, color=blue, label='AB42 Monomer')
        ax4.set_title('AB42 Monomer (ISF)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('AB42 Monomer (nM)', fontsize=14, fontweight='bold')
    else:
        ax4.set_title('AB42 Monomer (ISF) - Not Found', fontsize=16, fontweight='bold')
        ax4.set_ylabel('AB42 Monomer (nM)', fontsize=14, fontweight='bold')

    # Panel 5: IDE Activity (Bottom Left)
    ide_found = False
    if '[IDE_activity_ISF]' in result.colnames:
        ide_data = result['[IDE_activity_ISF]'][start_idx:]
        ax5.plot(x_filtered, ide_data, linewidth=3, color=blue, label='IDE Activity')
        ax5.set_title('IDE Activity', fontsize=16, fontweight='bold')
        ax5.set_ylabel('IDE Activity', fontsize=14, fontweight='bold')
        ide_found = True
    
    if not ide_found:
        ax5.plot(x_filtered, np.ones_like(x_filtered), linewidth=3, color=blue, label='IDE Activity (placeholder)')
        ax5.set_title('IDE Activity (Not Found)', fontsize=16, fontweight='bold')
        ax5.set_ylabel('IDE Activity', fontsize=14, fontweight='bold')

    # Panel 6: AB42 Plaque (Bottom Right) - plaques (O25, no volume scaling)
    if '[AB42_O25_ISF]' in result.colnames:
        plaque_data = result['[AB42_O25_ISF]'][start_idx:]
        if np.any(plaque_data > 0):  # Check if we have plaque data
            ax6.plot(x_filtered, plaque_data, linewidth=3, color=blue, label='AB42 Plaque')
            ax6.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold')
            ax6.set_ylabel('Plaque Load (nM)', fontsize=14, fontweight='bold')
        else:
            ax6.set_title('AB42 Plaque Load (No Data)', fontsize=16, fontweight='bold')
            ax6.set_ylabel('Plaque Load (nM)', fontsize=14, fontweight='bold')
    else:
        ax6.set_title('AB42 Plaque Load (Not Found)', fontsize=16, fontweight='bold')
        ax6.set_ylabel('Plaque Load (nM)', fontsize=14, fontweight='bold')

    # Set x-axis labels and other formatting for all panels
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    fig.savefig(plots_dir / f'{drug_type.lower()}_antimony_six_panel.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tellurium simulation for Antimony model.")
    parser.add_argument("--years", type=float, default=100.0, help="Number of years to simulate")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
    args = parser.parse_args()

    # Load the Antimony modelxml
    antimony_path = Path("../generated/sbml/Antimony_Geerts_model_opt_flexible5b.txt")

    
    # Read the Antimony model file
    with open(antimony_path, "r") as f:
        antimony_str = f.read()

    # Load Antimony model into Tellurium
    rr = te.loadAntimonyModel(antimony_str)
    #rr = te.loadSBMLModel(antimony_str)
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)

    # Set Optimized Parameters 
    failed_params = set_optimized_values(rr)
    if failed_params:
        print("Failed to set some parameters:")
        for param in failed_params:
            print(f"- {param}")
        sys.exit(1)
    
    # Run simulation (no parameter changes)
    output_file = f'antimony_simulation_results_{args.years}yr.csv'
    result = run_simulation(rr, args.years, output_file)

    # Generate Plots using the new approach
    plots_dir = Path("simulation_plots/antimony_results")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_six_panel_antimony(result, drug_type=args.drug, plots_dir=plots_dir)

    print(f"Antimony model simulation and plots complete. Plots saved to {plots_dir}") 