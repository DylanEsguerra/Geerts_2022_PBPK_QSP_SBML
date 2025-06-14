'''

'''

import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

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
rr.integrator.absolute_tolerance = 1e-6  # Even more relaxed tolerance
rr.integrator.relative_tolerance = 1e-6  # Even more relaxed tolerance
# set integrator settings
rr.integrator.setValue('stiff', True)


# Define the range of CL_AB42_IDE values to test
CL_Rates = np.linspace(1.e-08, 1.e-04, 10)  
colors = mpl.colormaps['tab10'].colors
t1 = 100*365*24  # Try a shorter time first
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# First sensitivity analysis
for i, CL_Rate in enumerate(CL_Rates):
    # Complete reset of the model
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)
    
    rr.exp_decline_rate_IDE_fortytwo = CL_Rate 
    try:
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', 'CL_AB42_IDE'])
        ax1.semilogy(result[:,0]/365/24, result[:,1], label=f'exp_decline_rate_IDE_fortytwo={CL_Rate:.2e}', color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax2.semilogy(result[:,0]/365/24, result[:,2], label=f'exp_decline_rate_IDE_fortytwo={CL_Rate:.2e}', color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)

    except Exception as e:
        print(f"Simulation failed for CL_Rate={CL_Rate:.2e}: {e}")

ax1.set_xlabel('Time (years)', fontsize=13)
ax2.set_xlabel('Time (years)', fontsize=13)
ax1.set_ylabel('Concentration (nM)', fontsize=13)
ax2.set_ylabel('Rate (1/hr)', fontsize=13)
ax1.set_title('AB42 Monomer', fontsize=14)
ax2.set_title('CL_AB42_IDE', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax2.tick_params(axis='both', which='major', labelsize=11)
#ax2.legend(title='exp_decline_rate_IDE_fortytwo', fontsize=9, title_fontsize=10, loc='lower left', bbox_to_anchor=(0.0, 1.0))
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'exp_decline_rate_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show() 

# Complete reset before second analysis
rr.reset()
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-8
rr.integrator.relative_tolerance = 1e-8
rr.integrator.setValue('stiff', True)

# Second sensitivity analysis
CL_Rates_0 = np.linspace(10, 1500, 10)  
colors = mpl.colormaps['tab10'].colors
t1 = 100*365*24
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

for i, CL_Rate_0 in enumerate(CL_Rates_0):
    # Complete reset for each simulation
    rr.reset()
    rr.setIntegrator('cvode')
    rr.integrator.absolute_tolerance = 1e-8
    rr.integrator.relative_tolerance = 1e-8
    rr.integrator.setValue('stiff', True)
    
    rr.CL_AB42_IDE = CL_Rate_0
    rr.AB42_IDE_Kcat_exp = CL_Rate_0
    try:
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', 'CL_AB42_IDE'])
        ax1.semilogy(result[:,0]/365/24, result[:,1], label=f'AB42_IDE_Kcat_exp={CL_Rate_0:.2e}', color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax2.semilogy(result[:,0]/365/24, result[:,2], label=f'AB42_IDE_Kcat_exp={CL_Rate_0:.2e}', color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)

    except Exception as e:
        print(f"Simulation failed for CL_Rate={CL_Rate_0:.2e}: {e}")

ax1.set_xlabel('Time (years)', fontsize=13)
ax2.set_xlabel('Time (years)', fontsize=13)
ax1.set_ylabel('Concentration (nM)', fontsize=13)
ax2.set_ylabel('Rate (1/hr)', fontsize=13)
ax1.set_title('AB42 Monomer', fontsize=14)
ax2.set_title('CL_AB42_IDE', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=11)
ax2.tick_params(axis='both', which='major', labelsize=11)
#ax2.legend(title='AB42_IDE_Kcat_exp', fontsize=9, title_fontsize=10, loc='lower left', bbox_to_anchor=(0.0, 1.0))
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'AB42_IDE_Kcat_exp_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show() 