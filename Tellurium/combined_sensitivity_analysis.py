import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm
import multiprocessing
from itertools import product
from functools import partial

# Add parent directory to path to import K_rates_extrapolate if needed ( not here because no low order agg rates )
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Create directory for sensitivity analysis figures if it doesn't exist
sensitivity_figures_dir = os.path.join('simulation_plots', 'sensitivity_analysis')
os.makedirs(sensitivity_figures_dir, exist_ok=True)

# Define simulation selections globally so the worker function can access it
simulation_selections = ['time', '[AB42_Monomer]',
                         '[AB42_Oligomer02]', '[AB42_Oligomer03]', '[AB42_Oligomer04]',
                         '[AB42_Oligomer05]', '[AB42_Oligomer06]', '[AB42_Oligomer07]',
                         '[AB42_Oligomer08]', '[AB42_Oligomer09]', '[AB42_Oligomer10]',
                         '[AB42_Oligomer11]', '[AB42_Oligomer12]', '[AB42_Oligomer13]',
                         '[AB42_Oligomer14]', '[AB42_Oligomer15]', '[AB42_Oligomer16]',
                         '[AB42_Fibril17]', '[AB42_Fibril18]', '[AB42_Fibril19]',
                         '[AB42_Fibril20]', '[AB42_Fibril21]', '[AB42_Fibril22]',
                         '[AB42_Fibril23]', '[AB42_Fibril24]',
                         '[AB42_Plaque_unbound]']

def calculate_suvr_and_weighted_sum(result, c1=2.52, c2=1.3, c3=3.5, c4=400000, volume_scale_factor_isf=0.2505):
    """
    Calculate SUVR and weighted amyloid species sum from simulation result data.
    """
    n_timepoints = len(result)
    suvr = np.zeros(n_timepoints)
    weighted_amyloid_sum = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        ab42_oligo = np.sum([size * result[t, i] for i, size in enumerate(range(2, 17), 2)])
        ab42_proto = np.sum([size * result[t, i] for i, size in enumerate(range(17, 25), 17)])
        ab42_plaque = result[t, 25]
        
        suvr_weighted_sum = (ab42_oligo + ab42_proto + c2 * 25 * ab42_plaque) / volume_scale_factor_isf
        weighted_amyloid_sum[t] = ab42_oligo + ab42_proto

        numerator = c1 * (suvr_weighted_sum ** c3)
        denominator = (suvr_weighted_sum ** c3) + (c4 ** c3)
        
        suvr[t] = 1.0 + (numerator / denominator) if denominator > 0 else 1.0
            
    return suvr, weighted_amyloid_sum

def run_single_simulation(params, sbml_model_str):
    """Worker function to run a single simulation."""
    kcat, k_rate = params
    try:
        rr = te.loadSBMLModel(sbml_model_str)
        rr.setIntegrator('cvode')
        rr.integrator.absolute_tolerance = 1e-6
        rr.integrator.relative_tolerance = 1e-6
        rr.integrator.setValue('stiff', True)
        
        rr.AB42_IDE_Kcat_exp = kcat
        rr.CL_AB42_IDE = kcat
        rr.k_F24_O12_fortytwo = k_rate
        
        t1 = 100 * 365 * 24
        result = rr.simulate(0, t1, 500, selections=simulation_selections)
        
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Simulation resulted in NaN or Inf")
            
        suvr, weighted_sum = calculate_suvr_and_weighted_sum(result)
        
        print(f"Success: k_cat={kcat:.2f}, k_rate={k_rate:.2f}")
        return (kcat, k_rate, result[-1, 1], result[-1, 25], weighted_sum[-1], suvr[-1])

    except Exception as e:
        print(f"Failed: k_cat={kcat:.2f}, k_rate={k_rate:.2f} -> {e}")
        return (kcat, k_rate, np.nan, np.nan, np.nan, np.nan)

def main():
    """Main function to run the parallel sensitivity analysis."""
    # Define sensitivity ranges
    kcat_values = np.linspace(100, 1000, 10)
    k_rate_values = np.linspace(10, 100, 10)

    # Load the SBML model string once
    xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'Geerts_2023_1.xml')
    with open(xml_path, "r") as f:
        sbml_str = f.read()

    # Prepare 2D grids for results
    final_monomer = np.zeros((len(kcat_values), len(k_rate_values)))
    final_plaque = np.zeros((len(kcat_values), len(k_rate_values)))
    final_weighted_sum = np.zeros((len(kcat_values), len(k_rate_values)))
    final_suvr = np.zeros((len(kcat_values), len(k_rate_values)))

    # Create a list of all parameter combinations
    param_combinations = list(product(kcat_values, k_rate_values))
    
    # Use multiprocessing to run simulations in parallel
    num_cores = 10 # 12 is the max for my computer
    print(f"Starting parallel simulations on {num_cores} cores...")

    worker_func = partial(run_single_simulation, sbml_model_str=sbml_str)

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(worker_func, param_combinations)

    print("All simulations complete. Processing and plotting results...")

    # Create maps for quick indexing
    kcat_map = {val: i for i, val in enumerate(kcat_values)}
    k_rate_map = {val: j for j, val in enumerate(k_rate_values)}

    # Populate result grids
    for result_tuple in results:
        kcat, k_rate, monomer, plaque, weighted_sum, suvr = result_tuple
        i = kcat_map[kcat]
        j = k_rate_map[k_rate]
        final_monomer[i, j] = monomer
        final_plaque[i, j] = plaque
        final_weighted_sum[i, j] = weighted_sum
        final_suvr[i, j] = suvr
        
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle('Combined Sensitivity Analysis (100-year endpoint)', fontsize=20, fontweight='bold')

    def plot_heatmap(ax, data, title, norm=None):
        im = ax.imshow(data, cmap='viridis', aspect='auto', origin='lower',
                       extent=[k_rate_values.min(), k_rate_values.max(), kcat_values.min(), kcat_values.max()],
                       norm=norm)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('k_F24_O12_fortytwo (1/h)', fontsize=12)
        ax.set_ylabel('AB42_IDE_Kcat_exp', fontsize=12)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Concentration (nM) or SUVR', fontsize=12)

    plot_heatmap(axes[0, 0], final_monomer, 'Final ISF Monomer', norm=LogNorm())
    plot_heatmap(axes[0, 1], final_plaque, 'Final Plaque Load', norm=LogNorm())
    plot_heatmap(axes[1, 0], final_weighted_sum, 'Final Weighted Oligomer/Fibril Sum', norm=LogNorm())
    plot_heatmap(axes[1, 1], final_suvr, 'Final SUVR')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(sensitivity_figures_dir, 'combined_kcat_krate_sensitivity.png')
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Combined sensitivity analysis complete. Plot saved to {save_path}")

if __name__ == "__main__":
    main() 