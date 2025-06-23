import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from K_rates_extrapolate import calculate_k_rates

# Create directory for sensitivity analysis figures if it doesn't exist
sensitivity_figures_dir = os.path.join('simulation_plots', 'sensitivity_analysis')
os.makedirs(sensitivity_figures_dir, exist_ok=True)

# Define sensitivity ranges for baseline plaque formation rates
# These are the minimum values as requested, with ranges going up from there
baseline_ab40_rate_values = np.linspace(5e-06, 5e-04, 20)     # AB40 baseline plaque rate range
baseline_ab42_rate_values = np.linspace(5e-05, 5e-03, 20)    # AB42 baseline plaque rate range

# Define simulation selections for AB40 species
simulation_selections_ab40 = ['time', '[AB40_Monomer]', 
                             '[AB40_Oligomer02]', '[AB40_Oligomer03]', '[AB40_Oligomer04]',
                             '[AB40_Oligomer05]', '[AB40_Oligomer06]', '[AB40_Oligomer07]',
                             '[AB40_Oligomer08]', '[AB40_Oligomer09]', '[AB40_Oligomer10]',
                             '[AB40_Oligomer11]', '[AB40_Oligomer12]', '[AB40_Oligomer13]',
                             '[AB40_Oligomer14]', '[AB40_Oligomer15]', '[AB40_Oligomer16]',
                             '[AB40_Fibril17]', '[AB40_Fibril18]', '[AB40_Fibril19]',
                             '[AB40_Fibril20]', '[AB40_Fibril21]', '[AB40_Fibril22]',
                             '[AB40_Fibril23]', '[AB40_Fibril24]',
                             '[AB40_Plaque_unbound]']

# Define simulation selections for AB42 species
simulation_selections_ab42 = ['time', '[AB42_Monomer]', 
                             '[AB42_Oligomer02]', '[AB42_Oligomer03]', '[AB42_Oligomer04]',
                             '[AB42_Oligomer05]', '[AB42_Oligomer06]', '[AB42_Oligomer07]',
                             '[AB42_Oligomer08]', '[AB42_Oligomer09]', '[AB42_Oligomer10]',
                             '[AB42_Oligomer11]', '[AB42_Oligomer12]', '[AB42_Oligomer13]',
                             '[AB42_Oligomer14]', '[AB42_Oligomer15]', '[AB42_Oligomer16]',
                             '[AB42_Fibril17]', '[AB42_Fibril18]', '[AB42_Fibril19]',
                             '[AB42_Fibril20]', '[AB42_Fibril21]', '[AB42_Fibril22]',
                             '[AB42_Fibril23]', '[AB42_Fibril24]',
                             '[AB42_Plaque_unbound]']

# Load the SBML model
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

# More robust integrator settings
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-8  # Less strict tolerance
rr.integrator.relative_tolerance = 1e-8  # Less strict tolerance
rr.integrator.setValue('stiff', True)

# Main sensitivity analysis for baseline_ab40_plaque_rate
# original value is 0.000005 (5e-06) - using range defined at top of script
colors = plt.cm.viridis(np.linspace(0, 1, len(baseline_ab40_rate_values)))
t1 = 100*365*24

# Set improved font parameters for sensitivity analysis plots
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
})

# Create figure with colorbar
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.1], height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
cax = fig.add_subplot(gs[:, 2])  # Colorbar axis

# Store simulation results for reuse in plotting function
ab40_simulation_results = {}
successful_sims = 0
for i, baseline_rate in enumerate(baseline_ab40_rate_values):
    try:
        rr.reset()
        # Calculate all rates for this baseline rate
        rates = calculate_k_rates(baseline_ab40_plaque_rate=baseline_rate)
        
        # Update all rates in the model, including plaque rates
        for key, value in rates.items():
            try:
                setattr(rr, key, value)
            except Exception as e:
                print(f"Could not set {key}: {e}")
        
        # Run simulation with more conservative output points
        result = rr.simulate(0, t1, 200, selections=simulation_selections_ab40)
        
        # Check if result is valid (no NaNs or extreme values)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for baseline_ab40_rate={baseline_rate:.1e}")
            continue
            
        # Store simulation result and rates for plotting function
        ab40_simulation_results[baseline_rate] = {
            'result': result,
            'rates': rates
        }
            
        # Calculate total loads
        time = result[:,0]/365/24  # Convert to years
        monomer = result[:,1]
        
        # Sum all oligomers (sizes 2-16)
        oligomer_load = np.sum(result[:,2:17], axis=1)
        
        # Sum all fibrils (sizes 17-24)
        fibril_load = np.sum(result[:,17:25], axis=1)
        
        # Get plaque load
        plaque_load = result[:,25]
        
        ax1.semilogy(time, monomer, color=colors[i], linewidth=2.5, alpha=0.8)
        ax2.semilogy(time, oligomer_load, color=colors[i], linewidth=2.5, alpha=0.8)
        ax3.semilogy(time, fibril_load, color=colors[i], linewidth=2.5, alpha=0.8)
        ax4.semilogy(time, plaque_load, color=colors[i], linewidth=2.5, alpha=0.8)
        successful_sims += 1
        
    except Exception as e:
        print(f"Simulation failed for baseline_ab40_rate={baseline_rate:.1e}: {e}")

print(f"Successful simulations for AB40 baseline rate: {successful_sims}/{len(baseline_ab40_rate_values)}")

ax1.set_title('AB40 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB40 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB40 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB40 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar with logarithmic normalization
from matplotlib.colors import LogNorm
norm = LogNorm(vmin=baseline_ab40_rate_values.min(), vmax=baseline_ab40_rate_values.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('AB40 Baseline Rate (L/(nM·h))', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add global title for AB40 baseline rate sensitivity
fig.suptitle('AB40 Aggregation Sensitivity vs AB40 Baseline Plaque Rate\nOriginal value: 5e-06', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'baseline_ab40_plaque_rate_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Sensitivity analysis for baseline_ab42_plaque_rate ---
# original value is 0.00005 (5e-05) - using range defined at top of script
colors2 = plt.cm.plasma(np.linspace(0, 1, len(baseline_ab42_rate_values)))

# Create figure with colorbar for AB42
fig2 = plt.figure(figsize=(18, 12))
gs2 = fig2.add_gridspec(2, 3, width_ratios=[1, 1, 0.1], height_ratios=[1, 1])

ax1 = fig2.add_subplot(gs2[0, 0])
ax2 = fig2.add_subplot(gs2[0, 1])
ax3 = fig2.add_subplot(gs2[1, 0])
ax4 = fig2.add_subplot(gs2[1, 1])
cax2 = fig2.add_subplot(gs2[:, 2])  # Colorbar axis

# Store simulation results for reuse in plotting function
ab42_simulation_results = {}
successful_sims2 = 0
for i, baseline_rate in enumerate(baseline_ab42_rate_values):
    try:
        rr.reset()
        # More robust integrator reset
        rr.setIntegrator('cvode')
        rr.integrator.absolute_tolerance = 1e-8
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.setValue('stiff', True)
        
        # Calculate all rates for this baseline rate
        rates = calculate_k_rates(baseline_ab42_plaque_rate=baseline_rate)
        
        # Update all rates in the model, including plaque rates
        for key, value in rates.items():
            try:
                setattr(rr, key, value)
            except Exception as e:
                print(f"Could not set {key}: {e}")
        
        # Run simulation
        result = rr.simulate(0, t1, 200, selections=simulation_selections_ab42)
        
        # Validate result
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for baseline_ab42_rate={baseline_rate:.1e}")
            continue
            
        # Store simulation result and rates for plotting function
        ab42_simulation_results[baseline_rate] = {
            'result': result,
            'rates': rates
        }
            
        # Calculate total loads
        time = result[:,0]/365/24  # Convert to years
        monomer = result[:,1]
        
        # Sum all oligomers (sizes 2-16)
        oligomer_load = np.sum(result[:,2:17], axis=1)
        
        # Sum all fibrils (sizes 17-24)
        fibril_load = np.sum(result[:,17:25], axis=1)
        
        # Get plaque load
        plaque_load = result[:,25]
        
        ax1.semilogy(time, monomer, color=colors2[i], linewidth=2.5, alpha=0.8)
        ax2.semilogy(time, oligomer_load, color=colors2[i], linewidth=2.5, alpha=0.8)
        ax3.semilogy(time, fibril_load, color=colors2[i], linewidth=2.5, alpha=0.8)
        ax4.semilogy(time, plaque_load, color=colors2[i], linewidth=2.5, alpha=0.8)
        successful_sims2 += 1
        
    except Exception as e:
        print(f"Simulation failed for baseline_ab42_rate={baseline_rate:.1e}: {e}")

print(f"Successful simulations for AB42 baseline rate: {successful_sims2}/{len(baseline_ab42_rate_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar for AB42 with logarithmic normalization
norm2 = LogNorm(vmin=baseline_ab42_rate_values.min(), vmax=baseline_ab42_rate_values.max())
sm2 = cm.ScalarMappable(cmap='plasma', norm=norm2)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=cax2)
cbar2.set_label('AB42 Baseline Rate (L/(nM·h))', fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=12)

# Add global title for AB42 baseline rate sensitivity
fig2.suptitle('AB42 Aggregation Sensitivity vs AB42 Baseline Plaque Rate\nOriginal value: 5e-05', 
              fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'baseline_ab42_plaque_rate_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show()

# Reset font parameters to original values
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'lines.linewidth': 3,
    'axes.linewidth': 2,
}) 