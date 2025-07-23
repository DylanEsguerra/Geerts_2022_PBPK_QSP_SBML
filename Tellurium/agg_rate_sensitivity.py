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

# Define sensitivity ranges at the top for consistency across all analyses
# These are in units of 1/h and need to be converted to 1/s before being used 
# in K_rates_extrapolate as the "original" values in the units of the Garai paper
kb0_fortytwo_values = np.linspace(1e-04, 50.0, 100)     # kb0 range for AB42 dimer to monomer (1/h)
kb1_fortytwo_values = np.linspace(1e-05, 1.0, 100)    # kb1 range for AB42 trimer to dimer (1/h)

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
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model_gantenerumab.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

# More robust integrator settings
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-8  # Less strict tolerance
rr.integrator.relative_tolerance = 1e-8  # Less strict tolerance
rr.integrator.setValue('stiff', True)

def plot_extrapolated_rates_sensitivity(kb0_sim_results, kb1_sim_results):
    """
    Plot how changing base rates (kb0 and kb1) affects the extrapolated backward rate curves and gain factors for AB42
    """
    oligomer_sizes = list(range(4, 25))  # Sizes 4 to 24
    
    # Set global font parameters for better readability
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
    
    # Create figure with 2x2 layout plus colorbar space
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    cax1 = fig.add_subplot(gs[0, 2])  # Colorbar for kb0
    cax2 = fig.add_subplot(gs[1, 2])  # Colorbar for kb1
    
    # --- Plot 1: kb0 variation effect on backward rates ---
    colors_kb0 = plt.cm.viridis(np.linspace(0, 1, len(kb0_fortytwo_values)))
    
    for i, kb0 in enumerate(kb0_fortytwo_values):
        #Garai_kb0 = kb0 / 3600  # Convert to 1/s
        rates = calculate_k_rates(kb0_fortytwo=kb0)
        
        # Extract backward rates for AB42
        backward_rates_42 = []
        for size in oligomer_sizes:
            if size < 17:
                backward_rates_42.append(rates[f'k_O{size}_O{size-1}_fortytwo'])
            elif size == 17:
                backward_rates_42.append(rates[f'k_F{size}_O{size-1}_fortytwo'])
            else:
                backward_rates_42.append(rates[f'k_F{size}_F{size-1}_fortytwo'])
        
        ax1.plot(oligomer_sizes, backward_rates_42, 'o-', color=colors_kb0[i], 
                linewidth=3, markersize=6)
    
    ax1.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Backward Rate (h⁻¹)', fontsize=14, fontweight='bold')
    ax1.set_title('kb0 (Dimer→Monomer)', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4, linestyle=':')
    ax1.set_xlim(3.5, 24.5)
    
    # Add colorbar for kb0
    norm1 = Normalize(vmin=kb0_fortytwo_values.min(), vmax=kb0_fortytwo_values.max())
    sm1 = cm.ScalarMappable(cmap='viridis', norm=norm1)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, cax=cax1)
    cbar1.set_label('kb0 Values (h⁻¹)', fontsize=14, fontweight='bold')
    cbar1.ax.tick_params(labelsize=12)
    
    # --- Plot 2: kb1 variation effect on backward rates ---
    colors_kb1 = plt.cm.plasma(np.linspace(0, 1, len(kb1_fortytwo_values)))
    
    for i, kb1 in enumerate(kb1_fortytwo_values):
        #Garai_kb1 = kb1 / 3600  # Convert to 1/s
        rates = calculate_k_rates(kb1_fortytwo=kb1)
        
        # Extract backward rates for AB42
        backward_rates_42 = []
        for size in oligomer_sizes:
            if size < 17:
                backward_rates_42.append(rates[f'k_O{size}_O{size-1}_fortytwo'])
            elif size == 17:
                backward_rates_42.append(rates[f'k_F{size}_O{size-1}_fortytwo'])
            else:
                backward_rates_42.append(rates[f'k_F{size}_F{size-1}_fortytwo'])
        
        ax2.plot(oligomer_sizes, backward_rates_42, 'o-', color=colors_kb1[i], 
                linewidth=3, markersize=6)
    
    ax2.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Backward Rate (h⁻¹)', fontsize=14, fontweight='bold')
    ax2.set_title('kb1 (Trimer→Dimer)', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4, linestyle=':')
    ax2.set_xlim(3.5, 24.5)
    
    # --- Plot 3: Concentration-dependent gain factors for kb0 variation ---
    for i, kb0 in enumerate(kb0_fortytwo_values):
        #Garai_kb0 = kb0 / 3600
        rates = calculate_k_rates(kb0_fortytwo=kb0)
        
        # Use stored simulation results instead of re-running simulations
        if kb0 not in kb0_sim_results:
            continue
            
        sim_result = kb0_sim_results[kb0]['result']
        rates = kb0_sim_results[kb0]['rates']
        
        # Get final concentrations
        final_concentrations = {}
        final_concentrations['AB42_Monomer'] = sim_result[-1, 1]
        
        col_idx = 2
        for size in range(2, 25):
            if size < 17:
                species_name = f'AB42_Oligomer{size:02d}'
            else:
                species_name = f'AB42_Fibril{size}'
            concentration = sim_result[-1, col_idx]
            
            if concentration < 0:
                concentration = 0
            
            final_concentrations[species_name] = concentration
            col_idx += 1
        
        concentration_dependent_gains = []
        for size in oligomer_sizes:
            # Get rate constants
            if size < 17:
                kf = rates[f'k_O{size-1}_O{size}_fortytwo']
                kb = rates[f'k_O{size}_O{size-1}_fortytwo']
                reactant_species = f'AB42_Oligomer{size-1:02d}' if size > 2 else 'AB42_Monomer'
                product_species = f'AB42_Oligomer{size:02d}'
            elif size == 17:
                kf = rates[f'k_O{size-1}_F{size}_fortytwo']
                kb = rates[f'k_F{size}_O{size-1}_fortytwo']
                reactant_species = f'AB42_Oligomer{size-1:02d}'
                product_species = f'AB42_Fibril{size}'
            else:
                kf = rates[f'k_F{size-1}_F{size}_fortytwo']
                kb = rates[f'k_F{size}_F{size-1}_fortytwo']
                reactant_species = f'AB42_Fibril{size-1}'
                product_species = f'AB42_Fibril{size}'
            
            # Get concentrations
            monomer_conc = final_concentrations['AB42_Monomer']
            reactant_conc = final_concentrations[reactant_species]
            product_conc = final_concentrations[product_species]
            
            # Calculate reaction velocities
            forward_velocity = kf * reactant_conc * monomer_conc
            backward_velocity = kb * product_conc
            
            # Concentration-dependent gain factor
            if backward_velocity > 0:
                gain = forward_velocity / backward_velocity
            else:
                gain = float('inf')
            
            concentration_dependent_gains.append(gain)
        
        ax3.plot(oligomer_sizes, concentration_dependent_gains, 'o-', color=colors_kb0[i], 
                linewidth=3, markersize=6)
    
    ax3.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax3.axhline(y=7, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Geerts value')
    ax3.set_yscale('log')
    ax3.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Concentration-Dependent Gain\n(Forward Velocity / Backward Velocity)', fontsize=14, fontweight='bold')
    ax3.set_title('Geerts Gain Factors vs kb0', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.4, linestyle=':')
    ax3.set_xlim(3.5, 24.5)
    
    # --- Plot 4: Concentration-dependent gain factors for kb1 variation ---
    for i, kb1 in enumerate(kb1_fortytwo_values):
        #Garai_kb1 = kb1 / 3600
        rates = calculate_k_rates(kb1_fortytwo=kb1)
        
        # Use stored simulation results instead of re-running simulations
        if kb1 not in kb1_sim_results:
            continue
            
        sim_result = kb1_sim_results[kb1]['result']
        rates = kb1_sim_results[kb1]['rates']
        
        # Get final concentrations
        final_concentrations = {}
        final_concentrations['AB42_Monomer'] = sim_result[-1, 1]
        
        col_idx = 2
        for size in range(2, 25):
            if size < 17:
                species_name = f'AB42_Oligomer{size:02d}'
            else:
                species_name = f'AB42_Fibril{size}'
            concentration = sim_result[-1, col_idx]
            
            if concentration < 0:
                concentration = 0
            
            final_concentrations[species_name] = concentration
            col_idx += 1
        
        concentration_dependent_gains = []
        for size in oligomer_sizes:
            # Get rate constants
            if size < 17:
                kf = rates[f'k_O{size-1}_O{size}_fortytwo']
                kb = rates[f'k_O{size}_O{size-1}_fortytwo']
                reactant_species = f'AB42_Oligomer{size-1:02d}' if size > 2 else 'AB42_Monomer'
                product_species = f'AB42_Oligomer{size:02d}'
            elif size == 17:
                kf = rates[f'k_O{size-1}_F{size}_fortytwo']
                kb = rates[f'k_F{size}_O{size-1}_fortytwo']
                reactant_species = f'AB42_Oligomer{size-1:02d}'
                product_species = f'AB42_Fibril{size}'
            else:
                kf = rates[f'k_F{size-1}_F{size}_fortytwo']
                kb = rates[f'k_F{size}_F{size-1}_fortytwo']
                reactant_species = f'AB42_Fibril{size-1}'
                product_species = f'AB42_Fibril{size}'
            
            # Get concentrations
            monomer_conc = final_concentrations['AB42_Monomer']
            reactant_conc = final_concentrations[reactant_species]
            product_conc = final_concentrations[product_species]
            
            # Calculate reaction velocities
            forward_velocity = kf * reactant_conc * monomer_conc
            backward_velocity = kb * product_conc
            
            # Concentration-dependent gain factor
            if backward_velocity > 0:
                gain = forward_velocity / backward_velocity
            else:
                gain = float('inf')
            
            concentration_dependent_gains.append(gain)
        
        ax4.plot(oligomer_sizes, concentration_dependent_gains, 'o-', color=colors_kb1[i], 
                linewidth=3, markersize=6)
    
    ax4.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax4.axhline(y=7, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Geerts value')
    ax4.set_yscale('log')
    ax4.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Concentration-Dependent Gain\n(Forward Velocity / Backward Velocity)', fontsize=14, fontweight='bold')
    ax4.set_title('Geerts Gain Factors vs kb1', fontsize=16, fontweight='bold', pad=20)
    ax4.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.4, linestyle=':')
    ax4.set_xlim(3.5, 24.5)
    
    # Add colorbar for kb1
    norm2 = Normalize(vmin=kb1_fortytwo_values.min(), vmax=kb1_fortytwo_values.max())
    sm2 = cm.ScalarMappable(cmap='plasma', norm=norm2)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, cax=cax2)
    cbar2.set_label('kb1 Values (h⁻¹)', fontsize=14, fontweight='bold')
    cbar2.ax.tick_params(labelsize=12)
    
    # Add global title
    fig.suptitle('AB42 Backward Rates and Gain Factors Sensitivity Analysis\nOriginal values: kb0 = 45.72 h⁻¹, kb1 = 1.08 h⁻¹', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(sensitivity_figures_dir, 'agg_rate_extrapolation_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reset font parameters
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

# Main sensitivity analysis for kb0_fortytwo (AB42 dimer to monomer backward rate)
# original value is 45.72 h⁻¹ - using range defined at top of script
colors = plt.cm.viridis(np.linspace(0, 1, len(kb0_fortytwo_values)))
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
kb0_simulation_results = {}
successful_sims = 0
for i, kb0 in enumerate(kb0_fortytwo_values):
    try:
        rr.reset()
        # Calculate all rates for this kb0
        #Garai_kb0 = kb0 / 3600  # Convert to 1/s
        rates = calculate_k_rates(kb0_fortytwo=kb0)
        
        # Set the main parameter first
        rr.k_O2_M_fortytwo = kb0
        
        # Set clearance parameters
        #rr.CL_AB42_IDE = 10000  # Initial condition of rate rule
        #rr.AB42_IDE_Kcat_exp = 10000
        
        # Update related AB42 rates, excluding any with 'Plaque' in the name
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    # Fix the bug: use setattr instead of rr.key = value
                    setattr(rr, key, value)
                except Exception as e:
                    print(f"Could not set {key}: {e}")
        
        # Run simulation with more conservative output points
        result = rr.simulate(0, t1, 200, selections=simulation_selections)
        
        # Check if result is valid (no NaNs or extreme values)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for k_O2_M_fortytwo={kb0:.1f}")
            continue
            
        # Store simulation result and rates for plotting function
        kb0_simulation_results[kb0] = {
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
        print(f"Simulation failed for k_O2_M_fortytwo={kb0:.0f}: {e}")

print(f"Successful simulations for kb0: {successful_sims}/{len(kb0_fortytwo_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar
norm = Normalize(vmin=kb0_fortytwo_values.min(), vmax=kb0_fortytwo_values.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('kb0 Values (h⁻¹)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add global title for kb0 sensitivity
fig.suptitle('AB42 Aggregation Sensitivity vs kb0 (Dimer→Monomer)\nOriginal value: 45.72 h⁻¹', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'agg_rate_sensitivity_kb0.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Sensitivity analysis for kb1_fortytwo ---
# original value is 1.08 h⁻¹ - using range defined at top of script
colors2 = plt.cm.plasma(np.linspace(0, 1, len(kb1_fortytwo_values)))

# Create figure with colorbar for kb1
fig2 = plt.figure(figsize=(18, 12))
gs2 = fig2.add_gridspec(2, 3, width_ratios=[1, 1, 0.1], height_ratios=[1, 1])

ax1 = fig2.add_subplot(gs2[0, 0])
ax2 = fig2.add_subplot(gs2[0, 1])
ax3 = fig2.add_subplot(gs2[1, 0])
ax4 = fig2.add_subplot(gs2[1, 1])
cax2 = fig2.add_subplot(gs2[:, 2])  # Colorbar axis

# Store simulation results for reuse in plotting function
kb1_simulation_results = {}
successful_sims2 = 0
for i, kb1 in enumerate(kb1_fortytwo_values):
    try:
        rr.reset()
        # More robust integrator reset
        rr.setIntegrator('cvode')
        rr.integrator.absolute_tolerance = 1e-8
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.setValue('stiff', True)
        
        #Garai_kb1 = kb1 / 3600  # Convert to 1/s for calculate_k_rates
        rates = calculate_k_rates(kb1_fortytwo=kb1)
        
        # Set the main parameter
        rr.k_O3_O2_fortytwo = kb1
        
        # Set clearance parameters
        #rr.CL_AB42_IDE = 10000  # Initial condition of rate rule
        #rr.AB42_IDE_Kcat_exp = 10000
        
        # Update related rates with better error handling
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    # Fix the bug: use setattr instead of rr.key = value
                    setattr(rr, key, value)
                except Exception as e:
                    print(f"Could not set {key}: {e}")
        
        # Run simulation
        result = rr.simulate(0, t1, 200, selections=simulation_selections)
        
        # Validate result
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for k_O3_O2_fortytwo={kb1:.2f}")
            continue
            
        # Store simulation result and rates for plotting function
        kb1_simulation_results[kb1] = {
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
        print(f"Simulation failed for k_O3_O2_fortytwo={kb1:.2f}: {e}")

print(f"Successful simulations for kb1: {successful_sims2}/{len(kb1_fortytwo_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar for kb1
norm2 = Normalize(vmin=kb1_fortytwo_values.min(), vmax=kb1_fortytwo_values.max())
sm2 = cm.ScalarMappable(cmap='plasma', norm=norm2)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=cax2)
cbar2.set_label('kb1 Values (h⁻¹)', fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=12)

# Add global title for kb1 sensitivity
fig2.suptitle('AB42 Aggregation Sensitivity vs kb1 (Trimer→Dimer)\nOriginal value: 1.08 h⁻¹', 
              fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'agg_rate_sensitivity_kb1.png'), dpi=300, bbox_inches='tight')
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

# Create the extrapolated rates visualization
print("\nGenerating extrapolated rates sensitivity plots...")
plot_extrapolated_rates_sensitivity(kb0_simulation_results, kb1_simulation_results) 
