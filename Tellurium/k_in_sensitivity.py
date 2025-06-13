import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Load the Antimony model from the text file
with open("Geerts_2023_Antimony-3.txt", "r") as f:
    antimony_str = f.read()

# Load the model
rr = te.loada(antimony_str)
#rr = te.loadSBMLModel(sbml_str)
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-12
rr.integrator.relative_tolerance = 1e-8
# set integrator settings
rr.integrator.setValue('stiff', True)

# Define the range of kin_AB42 values to test
K_in_42_Rates = np.linspace(0.010, 0.050, 10)  
colors = mpl.colormaps['tab10'].colors
t1 = 100*365*24  # Try a shorter time first
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for i, k_in_42 in enumerate(K_in_42_Rates):
    rr.reset()
    rr.k_in_AB42 = k_in_42 
    try:
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', '[AB42_Oligomer16]'])
        ax1.semilogy(result[:,0]/365/24, result[:,1], color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax2.semilogy(result[:,0]/365/24, result[:,2], color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        # Direct label at end of each line
        ax1.text(result[-1,0]/365/24, result[-1,1], f'{k_in_42:.2e}', color=colors[i % len(colors)], fontsize=8, va='center')
        ax2.text(result[-1,0]/365/24, result[-1,2], f'{k_in_42:.2e}', color=colors[i % len(colors)], fontsize=8, va='center')
    except Exception as e:
        print(f"Simulation failed for Rate={k_in_42:.2e}: {e}")
ax1.set_xlabel('Time (years)', fontsize=13)
ax2.set_xlabel('Time (years)', fontsize=13)
ax1.set_ylabel('Concentration (nM)', fontsize=13)
ax2.set_ylabel('Concentration (nM)', fontsize=13)
ax1.set_title('AB42 Monomer', fontsize=14)
ax2.set_title('AB42 Oligomer16', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax2.tick_params(axis='both', which='major', labelsize=11)
plt.tight_layout(pad=2.0)
plt.savefig('K_in_AB42_rate_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show() 

# --- Sensitivity analysis for k_APP_production ---
k_APP_production_values = np.linspace(100, 1500, 10)
colors = mpl.colormaps['tab10'].colors
t1 = 100*365*24  # same as above
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(8, 4))
for i, k_app in enumerate(k_APP_production_values):
    rr.reset()
    rr.k_APP_production = k_app
    try:
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', '[AB42_Oligomer16]'])
        ax3.semilogy(result[:,0]/365/24, result[:,1], color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax4.semilogy(result[:,0]/365/24, result[:,2], color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        # Direct label at end of each line
        ax3.text(result[-1,0]/365/24, result[-1,1], f'{k_app:.0f}', color=colors[i % len(colors)], fontsize=8, va='center')
        ax4.text(result[-1,0]/365/24, result[-1,2], f'{k_app:.0f}', color=colors[i % len(colors)], fontsize=8, va='center')
    except Exception as e:
        print(f"Simulation failed for k_APP_production={k_app:.0f}: {e}")
ax3.set_xlabel('Time (years)', fontsize=13)
ax4.set_xlabel('Time (years)', fontsize=13)
ax3.set_ylabel('Concentration (nM)', fontsize=13)
ax4.set_ylabel('Concentration (nM)', fontsize=13)
ax3.set_title('AB42 Monomer', fontsize=14)
ax4.set_title('AB42 Oligomer16', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=11)
ax4.tick_params(axis='both', which='major', labelsize=11)
plt.tight_layout(pad=2.0)
plt.savefig('k_APP_production_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show() 
