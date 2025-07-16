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

# Define sensitivity range for AB42_IDE_Kcat_exp
# AB42_IDE_Kcat_exp: original value is 50
AB42_IDE_Kcat_exp_values = np.linspace(50, 1000, 100)

# Define simulation selections once
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

# Load the SBML model
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'Geerts_2023_1.xml') # using APOE4 model
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

# More robust integrator settings
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-6  # More conservative tolerance
rr.integrator.relative_tolerance = 1e-6  # More conservative tolerance
rr.integrator.setValue('stiff', True)

# Main sensitivity analysis for AB42_IDE_Kcat_exp (AB42 IDE catalytic efficiency)
# original value is 50 - using range defined at top of script
colors = plt.cm.viridis(np.linspace(0, 1, len(AB42_IDE_Kcat_exp_values)))
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

# Store simulation results for analysis
AB42_IDE_Kcat_exp_simulation_results = {}
successful_sims = 0
for i, kcat_exp in enumerate(AB42_IDE_Kcat_exp_values):
    try:
        rr.reset()
        
        # Set the main parameter
        rr.AB42_IDE_Kcat_exp = kcat_exp
        rr.CL_AB42_IDE = kcat_exp # initial condition of rate rule
        
        # Run simulation with more conservative output points
        result = rr.simulate(0, t1, 10000, selections=simulation_selections)
        
        # Check if result is valid (no NaNs or extreme values)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for AB42_IDE_Kcat_exp={kcat_exp:.1f}")
            continue
            
        # Store simulation result for analysis
        AB42_IDE_Kcat_exp_simulation_results[kcat_exp] = {
            'result': result
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
        print(f"Simulation failed for AB42_IDE_Kcat_exp={kcat_exp:.1f}: {e}")

print(f"Successful simulations for AB42_IDE_Kcat_exp: {successful_sims}/{len(AB42_IDE_Kcat_exp_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar
norm = Normalize(vmin=AB42_IDE_Kcat_exp_values.min(), vmax=AB42_IDE_Kcat_exp_values.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('AB42_IDE_Kcat_exp Values', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add global title for AB42_IDE_Kcat_exp sensitivity
fig.suptitle('AB42 Clearance Sensitivity vs AB42_IDE_Kcat_exp\nOriginal value: 50', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'AB42_IDE_Kcat_exp_sensitivity.png'), dpi=300, bbox_inches='tight')
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

print("Sensitivity analysis for AB42_IDE_Kcat_exp completed!") 
