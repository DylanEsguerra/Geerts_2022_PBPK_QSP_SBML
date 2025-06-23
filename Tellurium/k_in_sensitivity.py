import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create directory for sensitivity analysis figures if it doesn't exist
sensitivity_figures_dir = os.path.join('simulation_plots', 'sensitivity_analysis')
os.makedirs(sensitivity_figures_dir, exist_ok=True)

# Load the SBML model
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()
# Load the model
rr = te.loadSBMLModel(sbml_str)
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-12
rr.integrator.relative_tolerance = 1e-8
# set integrator settings
rr.integrator.setValue('stiff', True)

# Define the range of kin_AB42 values to test
K_in_42_Rates = np.linspace(0.010, 0.50, 20)  
colors = plt.cm.viridis(np.linspace(0, 1, len(K_in_42_Rates)))
t1 = 100*365*24  # Try a shorter time first

# Create figure with colorbar for first analysis
fig1 = plt.figure(figsize=(16, 4))
gs1 = fig1.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1])

ax1 = fig1.add_subplot(gs1[0, 0])
ax2 = fig1.add_subplot(gs1[0, 1])
ax3 = fig1.add_subplot(gs1[0, 2])
cax1 = fig1.add_subplot(gs1[0, 3])  # Colorbar axis

for i, k_in_42 in enumerate(K_in_42_Rates):
    rr.reset()
    rr.k_in_AB42 = k_in_42 
    try:
        # Get all AB42 species for calculating loads
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', 
                                                    '[AB42_Oligomer02]', '[AB42_Oligomer03]', '[AB42_Oligomer04]',
                                                    '[AB42_Oligomer05]', '[AB42_Oligomer06]', '[AB42_Oligomer07]',
                                                    '[AB42_Oligomer08]', '[AB42_Oligomer09]', '[AB42_Oligomer10]',
                                                    '[AB42_Oligomer11]', '[AB42_Oligomer12]', '[AB42_Oligomer13]',
                                                    '[AB42_Oligomer14]', '[AB42_Oligomer15]', '[AB42_Oligomer16]',
                                                    '[AB42_Fibril17]', '[AB42_Fibril18]', '[AB42_Fibril19]',
                                                    '[AB42_Fibril20]', '[AB42_Fibril21]', '[AB42_Fibril22]',
                                                    '[AB42_Fibril23]', '[AB42_Fibril24]',
                                                    '[AB42_Plaque_unbound]'])
        
        # Calculate total oligomer load (sizes 2-16)
        oligomer_load = np.sum(result[:,2:17], axis=1)
        
        # Get plaque load
        plaque_load = result[:,25]
        
        ax1.semilogy(result[:,0]/365/24, result[:,1], color=colors[i], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax2.semilogy(result[:,0]/365/24, oligomer_load, color=colors[i], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax3.semilogy(result[:,0]/365/24, plaque_load, color=colors[i], linewidth=2.5, marker='o', markersize=3, markevery=30)
    except Exception as e:
        print(f"Simulation failed for Rate={k_in_42:.2e}: {e}")

ax1.set_xlabel('Time (years)', fontsize=13)
ax2.set_xlabel('Time (years)', fontsize=13)
ax3.set_xlabel('Time (years)', fontsize=13)
ax1.set_ylabel('Concentration (nM)', fontsize=13)
ax2.set_ylabel('Concentration (nM)', fontsize=13)
ax3.set_ylabel('Concentration (nM)', fontsize=13)
ax1.set_title('AB42 Monomer', fontsize=14)
ax2.set_title('Total AB42 Oligomer Load', fontsize=14)
ax3.set_title('AB42 Plaque Load', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax3.tick_params(axis='both', which='major', labelsize=11)

# Add colorbar for first analysis
norm1 = Normalize(vmin=K_in_42_Rates.min(), vmax=K_in_42_Rates.max())
sm1 = cm.ScalarMappable(cmap='viridis', norm=norm1)
sm1.set_array([])
cbar1 = plt.colorbar(sm1, cax=cax1)
cbar1.set_label('k_in_AB42', fontsize=13, fontweight='bold')
cbar1.ax.tick_params(labelsize=11)

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'K_in_AB42_rate_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show() 

# --- Sensitivity analysis for k_APP_production ---
k_APP_production_values = np.linspace(100, 1500, 20)
colors2 = plt.cm.plasma(np.linspace(0, 1, len(k_APP_production_values)))
t1 = 100*365*24  # same as above

# Create figure with colorbar for second analysis
fig2 = plt.figure(figsize=(16, 4))
gs2 = fig2.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.1])

ax1 = fig2.add_subplot(gs2[0, 0])
ax2 = fig2.add_subplot(gs2[0, 1])
ax3 = fig2.add_subplot(gs2[0, 2])
cax2 = fig2.add_subplot(gs2[0, 3])  # Colorbar axis

for i, k_app in enumerate(k_APP_production_values):
    rr.reset()
    rr.k_APP_production = k_app
    try:
        # Get all AB42 species for calculating loads
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', 
                                                    '[AB42_Oligomer02]', '[AB42_Oligomer03]', '[AB42_Oligomer04]',
                                                    '[AB42_Oligomer05]', '[AB42_Oligomer06]', '[AB42_Oligomer07]',
                                                    '[AB42_Oligomer08]', '[AB42_Oligomer09]', '[AB42_Oligomer10]',
                                                    '[AB42_Oligomer11]', '[AB42_Oligomer12]', '[AB42_Oligomer13]',
                                                    '[AB42_Oligomer14]', '[AB42_Oligomer15]', '[AB42_Oligomer16]',
                                                    '[AB42_Fibril17]', '[AB42_Fibril18]', '[AB42_Fibril19]',
                                                    '[AB42_Fibril20]', '[AB42_Fibril21]', '[AB42_Fibril22]',
                                                    '[AB42_Fibril23]', '[AB42_Fibril24]',
                                                    '[AB42_Plaque_unbound]'])
        
        # Calculate total oligomer load (sizes 2-16)
        oligomer_load = np.sum(result[:,2:17], axis=1)
        
        # Get plaque load
        plaque_load = result[:,25]
        
        ax1.semilogy(result[:,0]/365/24, result[:,1], color=colors2[i], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax2.semilogy(result[:,0]/365/24, oligomer_load, color=colors2[i], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax3.semilogy(result[:,0]/365/24, plaque_load, color=colors2[i], linewidth=2.5, marker='o', markersize=3, markevery=30)
    except Exception as e:
        print(f"Simulation failed for k_APP_production={k_app:.0f}: {e}")

ax1.set_xlabel('Time (years)', fontsize=13)
ax2.set_xlabel('Time (years)', fontsize=13)
ax3.set_xlabel('Time (years)', fontsize=13)
ax1.set_ylabel('Concentration (nM)', fontsize=13)
ax2.set_ylabel('Concentration (nM)', fontsize=13)
ax3.set_ylabel('Concentration (nM)', fontsize=13)
ax1.set_title('AB42 Monomer', fontsize=14)
ax2.set_title('Total AB42 Oligomer Load', fontsize=14)
ax3.set_title('AB42 Plaque Load', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax2.tick_params(axis='both', which='major', labelsize=11)
ax3.tick_params(axis='both', which='major', labelsize=11)

# Add colorbar for second analysis
norm2 = Normalize(vmin=k_APP_production_values.min(), vmax=k_APP_production_values.max())
sm2 = cm.ScalarMappable(cmap='plasma', norm=norm2)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=cax2)
cbar2.set_label('k_APP_production', fontsize=13, fontweight='bold')
cbar2.ax.tick_params(labelsize=11)

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'k_APP_production_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show() 
