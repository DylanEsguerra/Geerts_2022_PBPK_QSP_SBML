import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm

# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from K_rates_extrapolate import calculate_k_rates

# Create directory for sensitivity analysis figures if it doesn't exist
sensitivity_figures_dir = os.path.join('simulation_plots', 'sensitivity_analysis')
os.makedirs(sensitivity_figures_dir, exist_ok=True)

# Define sensitivity ranges for AB42 forward rates
# These are in units of M⁻¹s⁻¹ (original literature units)
# Original values: kf0_fortytwo = 9.9 × 10² M⁻¹s⁻¹, kf1_fortytwo = 38.0 M⁻¹s⁻¹
kf0_fortytwo_values = np.linspace(10.0, 2000.0, 100)     # kf0 range for AB42 monomer to dimer (M⁻¹s⁻¹)
kf1_fortytwo_values = np.linspace(1.0, 100.0, 100)    # kf1 range for AB42 dimer to trimer (M⁻¹s⁻¹)

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
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'combined_master_model.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

# More robust integrator settings
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-8
rr.integrator.relative_tolerance = 1e-8
rr.integrator.setValue('stiff', True)

def plot_extrapolated_rates_sensitivity(kf0_sim_results, kf1_sim_results):
    """
    Plot how changing base forward rates (kf0 and kf1) affects the extrapolated forward rate curves and gain factors for AB42
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
    cax1 = fig.add_subplot(gs[0, 2])  # Colorbar for kf0
    cax2 = fig.add_subplot(gs[1, 2])  # Colorbar for kf1
    
    # --- Plot 1: kf0 variation effect on forward rates ---
    colors_kf0 = plt.cm.viridis(np.linspace(0, 1, len(kf0_fortytwo_values)))
    
    for i, kf0 in enumerate(kf0_fortytwo_values):
        rates = calculate_k_rates(original_kf0_fortytwo=kf0)
        
        # Extract forward rates for AB42
        forward_rates_42 = []
        for size in oligomer_sizes:
            if size < 17:
                forward_rates_42.append(rates[f'k_O{size-1}_O{size}_fortytwo'])
            elif size == 17:
                forward_rates_42.append(rates[f'k_O{size-1}_F{size}_fortytwo'])
            else:
                forward_rates_42.append(rates[f'k_F{size-1}_F{size}_fortytwo'])
        
        ax1.plot(oligomer_sizes, forward_rates_42, 'o-', color=colors_kf0[i], 
                linewidth=3, markersize=6)
    
    ax1.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Forward Rate (nM⁻¹h⁻¹)', fontsize=14, fontweight='bold')
    ax1.set_title('kf0 (Monomer→Dimer)', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4, linestyle=':')
    ax1.set_xlim(3.5, 24.5)
    
    # Add colorbar for kf0
    norm1 = LogNorm(vmin=kf0_fortytwo_values.min(), vmax=kf0_fortytwo_values.max())
    sm1 = cm.ScalarMappable(cmap='viridis', norm=norm1)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, cax=cax1)
    cbar1.set_label('kf0 Values (M⁻¹s⁻¹)', fontsize=14, fontweight='bold')
    cbar1.ax.tick_params(labelsize=12)
    
    # --- Plot 2: kf1 variation effect on forward rates ---
    colors_kf1 = plt.cm.plasma(np.linspace(0, 1, len(kf1_fortytwo_values)))
    
    for i, kf1 in enumerate(kf1_fortytwo_values):
        rates = calculate_k_rates(original_kf1_fortytwo=kf1)
        
        # Extract forward rates for AB42
        forward_rates_42 = []
        for size in oligomer_sizes:
            if size < 17:
                forward_rates_42.append(rates[f'k_O{size-1}_O{size}_fortytwo'])
            elif size == 17:
                forward_rates_42.append(rates[f'k_O{size-1}_F{size}_fortytwo'])
            else:
                forward_rates_42.append(rates[f'k_F{size-1}_F{size}_fortytwo'])
        
        ax2.plot(oligomer_sizes, forward_rates_42, 'o-', color=colors_kf1[i], 
                linewidth=3, markersize=6)
    
    ax2.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Forward Rate (nM⁻¹h⁻¹)', fontsize=14, fontweight='bold')
    ax2.set_title('kf1 (Dimer→Trimer)', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4, linestyle=':')
    ax2.set_xlim(3.5, 24.5)
    
    # --- Plot 3: Concentration-dependent gain factors for kf0 variation ---
    for i, kf0 in enumerate(kf0_fortytwo_values):
        rates = calculate_k_rates(original_kf0_fortytwo=kf0)
        
        # Use stored simulation results instead of re-running simulations
        if kf0 not in kf0_sim_results:
            continue
            
        sim_result = kf0_sim_results[kf0]['result']
        rates = kf0_sim_results[kf0]['rates']
        
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
        
        ax3.plot(oligomer_sizes, concentration_dependent_gains, 'o-', color=colors_kf0[i], 
                linewidth=3, markersize=6)
    
    ax3.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax3.axhline(y=7, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Geerts value')
    ax3.set_yscale('log')
    ax3.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Concentration-Dependent Gain\n(Forward Velocity / Backward Velocity)', fontsize=14, fontweight='bold')
    ax3.set_title('Geerts Gain Factors vs kf0', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.4, linestyle=':')
    ax3.set_xlim(3.5, 24.5)
    
    # --- Plot 4: Concentration-dependent gain factors for kf1 variation ---
    for i, kf1 in enumerate(kf1_fortytwo_values):
        rates = calculate_k_rates(original_kf1_fortytwo=kf1)
        
        # Use stored simulation results instead of re-running simulations
        if kf1 not in kf1_sim_results:
            continue
            
        sim_result = kf1_sim_results[kf1]['result']
        rates = kf1_sim_results[kf1]['rates']
        
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
        
        ax4.plot(oligomer_sizes, concentration_dependent_gains, 'o-', color=colors_kf1[i], 
                linewidth=3, markersize=6)
    
    ax4.axvline(x=17, color='grey', linestyle='-', alpha=0.8, linewidth=2)
    ax4.axhline(y=7, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Geerts value')
    ax4.set_yscale('log')
    ax4.set_xlabel('Oligomer Size', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Concentration-Dependent Gain\n(Forward Velocity / Backward Velocity)', fontsize=14, fontweight='bold')
    ax4.set_title('Geerts Gain Factors vs kf1', fontsize=16, fontweight='bold', pad=20)
    ax4.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.4, linestyle=':')
    ax4.set_xlim(3.5, 24.5)
    
    # Add colorbar for kf1
    norm2 = LogNorm(vmin=kf1_fortytwo_values.min(), vmax=kf1_fortytwo_values.max())
    sm2 = cm.ScalarMappable(cmap='plasma', norm=norm2)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, cax=cax2)
    cbar2.set_label('kf1 Values (M⁻¹s⁻¹)', fontsize=14, fontweight='bold')
    cbar2.ax.tick_params(labelsize=12)
    
    # Add global title
    fig.suptitle('AB42 Forward Rates and Gain Factors Sensitivity Analysis\nOriginal values: kf0 = 990 M⁻¹s⁻¹, kf1 = 38.0 M⁻¹s⁻¹', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(sensitivity_figures_dir, 'forward_rate_extrapolation_sensitivity.png'), dpi=300, bbox_inches='tight')
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

# Main sensitivity analysis for kf0_fortytwo (AB42 monomer to dimer forward rate)
# original value is 9.9 × 10² M⁻¹s⁻¹ - using range defined at top of script
colors = plt.cm.viridis(np.linspace(0, 1, len(kf0_fortytwo_values)))
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
kf0_simulation_results = {}
successful_sims = 0
for i, kf0 in enumerate(kf0_fortytwo_values):
    try:
        rr.reset()
        # Calculate all rates for this kf0
        rates = calculate_k_rates(original_kf0_fortytwo=kf0)
        
        # Convert to model units (nM⁻¹h⁻¹)
        kf0_model_units = kf0 * 3.6e-6  # Convert from M⁻¹s⁻¹ to nM⁻¹h⁻¹
        
        # Set the main parameter in model
        rr.k_M_O2_fortytwo = kf0_model_units
        
        # Update related AB42 rates, excluding any with 'Plaque' in the name
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    setattr(rr, key, value)
                except Exception as e:
                    print(f"Could not set {key}: {e}")
        
        # Run simulation with more conservative output points
        result = rr.simulate(0, t1, 200, selections=simulation_selections)
        
        # Check if result is valid (no NaNs or extreme values)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for k_M_O2_fortytwo={kf0:.1f}")
            continue
            
        # Store simulation result and rates for plotting function
        kf0_simulation_results[kf0] = {
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
        print(f"Simulation failed for k_M_O2_fortytwo={kf0:.0f}: {e}")

print(f"Successful simulations for kf0: {successful_sims}/{len(kf0_fortytwo_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar with logarithmic normalization
norm = LogNorm(vmin=kf0_fortytwo_values.min(), vmax=kf0_fortytwo_values.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('kf0 Values (M⁻¹s⁻¹)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add global title for kf0 sensitivity
fig.suptitle('AB42 Aggregation Sensitivity vs kf0 (Monomer→Dimer)\nOriginal value: 990 M⁻¹s⁻¹', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'forward_rate_sensitivity_kf0.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- Sensitivity analysis for kf1_fortytwo ---
# original value is 38.0 M⁻¹s⁻¹ - using range defined at top of script
colors2 = plt.cm.plasma(np.linspace(0, 1, len(kf1_fortytwo_values)))

# Create figure with colorbar for kf1
fig2 = plt.figure(figsize=(18, 12))
gs2 = fig2.add_gridspec(2, 3, width_ratios=[1, 1, 0.1], height_ratios=[1, 1])

ax1 = fig2.add_subplot(gs2[0, 0])
ax2 = fig2.add_subplot(gs2[0, 1])
ax3 = fig2.add_subplot(gs2[1, 0])
ax4 = fig2.add_subplot(gs2[1, 1])
cax2 = fig2.add_subplot(gs2[:, 2])  # Colorbar axis

# Store simulation results for reuse in plotting function
kf1_simulation_results = {}
successful_sims2 = 0
for i, kf1 in enumerate(kf1_fortytwo_values):
    try:
        rr.reset()
        # More robust integrator reset
        rr.setIntegrator('cvode')
        rr.integrator.absolute_tolerance = 1e-8
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.setValue('stiff', True)
        
        rates = calculate_k_rates(original_kf1_fortytwo=kf1)
        
        # Convert to model units (nM⁻¹h⁻¹)
        kf1_model_units = kf1 * 3.6e-6  # Convert from M⁻¹s⁻¹ to nM⁻¹h⁻¹
        
        # Set the main parameter in model
        rr.k_O2_O3_fortytwo = kf1_model_units
        
        # Update related rates with better error handling
        for key, value in rates.items():
            if '_fortytwo' in key and 'Plaque' not in key:
                try:
                    setattr(rr, key, value)
                except Exception as e:
                    print(f"Could not set {key}: {e}")
        
        # Run simulation
        result = rr.simulate(0, t1, 200, selections=simulation_selections)
        
        # Validate result
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for k_O2_O3_fortytwo={kf1:.2f}")
            continue
            
        # Store simulation result and rates for plotting function
        kf1_simulation_results[kf1] = {
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
        print(f"Simulation failed for k_O2_O3_fortytwo={kf1:.2f}: {e}")

print(f"Successful simulations for kf1: {successful_sims2}/{len(kf1_fortytwo_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

# Add colorbar for kf1 with logarithmic normalization
norm2 = LogNorm(vmin=kf1_fortytwo_values.min(), vmax=kf1_fortytwo_values.max())
sm2 = cm.ScalarMappable(cmap='plasma', norm=norm2)
sm2.set_array([])
cbar2 = plt.colorbar(sm2, cax=cax2)
cbar2.set_label('kf1 Values (M⁻¹s⁻¹)', fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=12)

# Add global title for kf1 sensitivity
fig2.suptitle('AB42 Aggregation Sensitivity vs kf1 (Dimer→Trimer)\nOriginal value: 38.0 M⁻¹s⁻¹', 
              fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'forward_rate_sensitivity_kf1.png'), dpi=300, bbox_inches='tight')
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
print("\nGenerating extrapolated forward rates sensitivity plots...")
plot_extrapolated_rates_sensitivity(kf0_simulation_results, kf1_simulation_results) 