'''

'''

import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the Antimony model from the text file
#with open("Geerts_2023_Antimony-3.txt", "r") as f:
with open("combined_master_model.xml", "r") as f:
    # For Antimony  
    #antimony_str = f.read()
    # For SBML
    sbml_str = f.read()

# Load the model
#rr = te.loada(antimony_str)
rr = te.loadSBMLModel(sbml_str)
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-12
rr.integrator.relative_tolerance = 1e-12
# set integrator settings
rr.integrator.setValue('stiff', True)

# Define the range of CL_AB42_IDE values to test
CL_Rates = np.linspace(1.e-08, 1.e-04, 10)  # Example: 0.1, 0.48, 0.86, 1.24, 1.62, 2.0
colors = mpl.colormaps['tab10'].colors
t1 = 100*365*24  # Try a shorter time first
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for i, CL_Rate in enumerate(CL_Rates):
    rr.reset()
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
plt.savefig('exp_decline_rate_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show() 


# Define the range of CL_AB42_IDE values to test
CL_Rates_0 = np.linspace(10, 150, 10)  # Example: 0.1, 0.48, 0.86, 1.24, 1.62, 2.0
colors = mpl.colormaps['tab10'].colors
t1 = 100*365*24  # Try a shorter time first
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
for i, CL_Rate_0 in enumerate(CL_Rates_0):
    rr.reset()
    rr.CL_AB42_IDE = CL_Rate_0 #(Initial condition of rate rule)
    rr.AB42_IDE_Kcat_exp = CL_Rate_0
    #rr.AB42_IDE_Kcat_lin = CL_Rate_0
    try:
        result = rr.simulate(0, t1, 300, selections=['time', '[AB42_Monomer]', 'CL_AB42_IDE'])
        ax1.semilogy(result[:,0]/365/24, result[:,1], label=f'AB42_IDE_Kcat_exp={10000:.2e}', color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)
        ax2.semilogy(result[:,0]/365/24, result[:,2], label=f'AB42_IDE_Kcat_exp={10000:.2e}', color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=3, markevery=30)

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
#ax2.legend(title='AB42_IDE_Kcat_exp', fontsize=9, title_fontsize=10, loc='lower left', bbox_to_anchor=(0.0, 1.0))
plt.tight_layout(pad=2.0)
plt.savefig('AB42_IDE_Kcat_exp_sensitivity.png', dpi=300, bbox_inches='tight')
plt.show() 