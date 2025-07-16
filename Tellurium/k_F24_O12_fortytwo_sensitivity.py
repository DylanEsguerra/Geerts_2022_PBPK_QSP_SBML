import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Add parent directory to path to import K_rates_extrapolate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from K_rates_extrapolate import calculate_k_rates

# Create directory for sensitivity analysis figures if it doesn't exist
sensitivity_figures_dir = os.path.join('simulation_plots', 'sensitivity_analysis')
os.makedirs(sensitivity_figures_dir, exist_ok=True)

def calculate_suvr_from_result(result, c1=2.52, c2=1.3, c3=3.5, c4=400000, volume_scale_factor_isf=0.2505):
    """
    Calculate SUVR using the weighted sum formula from simulation result data.
    
    Args:
        result: Simulation result array with columns [time, AB42_Monomer, AB42_Oligomer02-16, AB42_Fibril17-24, AB42_Plaque_unbound]
        c1, c2, c3, c4: SUVR parameters
        volume_scale_factor_isf: Volume scaling factor for ISF compartment
        
    Returns:
        SUVR array
    """
    n_timepoints = len(result)
    suvr = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        # Calculate AB42 oligomer sum (weighted by size) - columns 2-16 (indices 1-15)
        ab42_oligo = 0.0
        for i in range(2, 17):  # Oligomer sizes 2-16
            size = i
            ab42_oligo += size * result[t, i]
        
        # Calculate AB42 protofibril sum (fibrils 17-24) - columns 17-24 (indices 16-23)
        ab42_proto = 0.0
        for i in range(17, 25):  # Fibril sizes 17-24
            size = i
            ab42_proto += size * result[t, i]
        
        # Get AB42 plaque - column 25 (index 24)
        ab42_plaque = result[t, 24]
        
        # Calculate the weighted sum
        weighted_sum = (ab42_oligo + ab42_proto + c2 * ab42_plaque) / volume_scale_factor_isf
        
        # Calculate the numerator and denominator for SUVR
        numerator = c1 * (weighted_sum ** c3)
        denominator = (weighted_sum ** c3) + (c4 ** c3)
        
        # Calculate SUVR
        if denominator > 0:
            suvr[t] = 1.0 + (numerator / denominator)
        else:
            suvr[t] = 1.0  # Default value if denominator is zero
    
    return suvr

# Define sensitivity ranges for k_F24_O12_fortytwo parameter
# Original value is typically around 0.1-1.0, so we'll test a range around that
k_F24_O12_fortytwo_values = np.linspace(0.1, 10, 100)  # Range from 1 to 100

# Define simulation selections for AB42 species (focusing on AB42 since this is AB42-specific parameter)
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
xml_path = os.path.join(parent_dir, 'generated', 'sbml', 'Geerts_2023_1.xml')
with open(xml_path, "r") as f:
    sbml_str = f.read()

rr = te.loadSBMLModel(sbml_str)

# More robust integrator settings
rr.setIntegrator('cvode')
rr.integrator.absolute_tolerance = 1e-8  # Less strict tolerance
rr.integrator.relative_tolerance = 1e-8  # Less strict tolerance
rr.integrator.setValue('stiff', True)

# Main sensitivity analysis for k_F24_O12_fortytwo
colors = plt.cm.viridis(np.linspace(0, 1, len(k_F24_O12_fortytwo_values)))
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

# Create figure with colorbar - now 3x2 layout to include SUVR
fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.1], height_ratios=[1, 1, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])  # New SUVR plot
cax = fig.add_subplot(gs[:, 2])  # Colorbar axis

# Store simulation results for reuse in plotting function
simulation_results = {}
successful_sims = 0
for i, k_rate in enumerate(k_F24_O12_fortytwo_values):
    try:
        rr.reset()
        # More robust integrator reset
        rr.setIntegrator('cvode')
        rr.integrator.absolute_tolerance = 1e-8
        rr.integrator.relative_tolerance = 1e-8
        rr.integrator.setValue('stiff', True)
        
        # Set the specific parameter value directly
        try:
            setattr(rr, 'k_F24_O12_fortytwo', k_rate)
        except Exception as e:
            print(f"Could not set k_F24_O12_fortytwo={k_rate}: {e}")
            continue
        
        # Run simulation with more conservative output points
        result = rr.simulate(0, t1, 200, selections=simulation_selections_ab42)
        
        # Check if result is valid (no NaNs or extreme values)
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            print(f"Invalid result for k_F24_O12_fortytwo={k_rate:.3f}")
            continue
            
        # Calculate SUVR
        suvr = calculate_suvr_from_result(result)
            
        # Store simulation result for plotting function
        simulation_results[k_rate] = {
            'result': result,
            'k_rate': k_rate,
            'suvr': suvr
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
        ax5.plot(time, suvr, color=colors[i], linewidth=2.5, alpha=0.8)  # SUVR plot
        successful_sims += 1
        
    except Exception as e:
        print(f"Simulation failed for k_F24_O12_fortytwo={k_rate:.3f}: {e}")

print(f"Successful simulations for k_F24_O12_fortytwo: {successful_sims}/{len(k_F24_O12_fortytwo_values)}")

ax1.set_title('AB42 Monomer Concentration', fontsize=16, fontweight='bold', pad=20)
ax2.set_title('Total AB42 Oligomer Load', fontsize=16, fontweight='bold', pad=20)
ax3.set_title('Total AB42 Fibril Load', fontsize=16, fontweight='bold', pad=20)
ax4.set_title('AB42 Plaque Load', fontsize=16, fontweight='bold', pad=20)
ax5.set_title('SUVR Progression', fontsize=16, fontweight='bold', pad=20)

for ax in (ax1, ax2, ax3, ax4):
    ax.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle=':')

ax5.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
ax5.set_ylabel('SUVR', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.4, linestyle=':')

# Add colorbar with logarithmic normalization
from matplotlib.colors import LogNorm
norm = LogNorm(vmin=k_F24_O12_fortytwo_values.min(), vmax=k_F24_O12_fortytwo_values.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('k_F24_O12_fortytwo (1/h)', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=12)

# Add global title for k_F24_O12_fortytwo sensitivity
fig.suptitle('AB42 Aggregation and SUVR Sensitivity vs k_F24_O12_fortytwo\n(Fibril24 → Oligomer12 breakdown rate)', 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'k_F24_O12_fortytwo_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.show()

# Create additional analysis plots to show the specific effects
fig2, axes = plt.subplots(3, 2, figsize=(15, 18))

# Plot 1: Fibril24 concentration specifically
ax1 = axes[0, 0]
for k_rate, data in simulation_results.items():
    time = data['result'][:,0]/365/24
    fibril24 = data['result'][:,24]  # AB42_Fibril24 is at index 24
    ax1.semilogy(time, fibril24, color=colors[np.where(k_F24_O12_fortytwo_values == k_rate)[0][0]], 
                 linewidth=2.5, alpha=0.8, label=f'{k_rate:.3f}')
ax1.set_title('AB42 Fibril24 Concentration', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.4, linestyle=':')

# Plot 2: Oligomer12 concentration specifically
ax2 = axes[0, 1]
for k_rate, data in simulation_results.items():
    time = data['result'][:,0]/365/24
    oligomer12 = data['result'][:,12]  # AB42_Oligomer12 is at index 12
    ax2.semilogy(time, oligomer12, color=colors[np.where(k_F24_O12_fortytwo_values == k_rate)[0][0]], 
                 linewidth=2.5, alpha=0.8, label=f'{k_rate:.3f}')
ax2.set_title('AB42 Oligomer12 Concentration', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (years)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Concentration (nM)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.4, linestyle=':')

# Plot 3: Final concentrations vs parameter value
ax3 = axes[1, 0]
final_monomer = []
final_oligomer = []
final_fibril = []
final_plaque = []
final_suvr = []
k_rates_used = []

for k_rate, data in simulation_results.items():
    final_time_idx = -1
    final_monomer.append(data['result'][final_time_idx, 1])
    final_oligomer.append(np.sum(data['result'][final_time_idx, 2:17]))
    final_fibril.append(np.sum(data['result'][final_time_idx, 17:25]))
    final_plaque.append(data['result'][final_time_idx, 25])
    final_suvr.append(data['suvr'][final_time_idx])
    k_rates_used.append(k_rate)

ax3.semilogy(k_rates_used, final_monomer, 'o-', label='Monomer', linewidth=2, markersize=8)
ax3.semilogy(k_rates_used, final_oligomer, 's-', label='Oligomer', linewidth=2, markersize=8)
ax3.semilogy(k_rates_used, final_fibril, '^-', label='Fibril', linewidth=2, markersize=8)
ax3.semilogy(k_rates_used, final_plaque, 'd-', label='Plaque', linewidth=2, markersize=8)
ax3.set_title('Final Concentrations vs k_F24_O12_fortytwo', fontsize=16, fontweight='bold')
ax3.set_xlabel('k_F24_O12_fortytwo (1/h)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Final Concentration (nM)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.4, linestyle=':')
ax3.legend()

# Plot 4: Peak concentrations vs parameter value
ax4 = axes[1, 1]
peak_monomer = []
peak_oligomer = []
peak_fibril = []
peak_plaque = []
peak_suvr = []

for k_rate, data in simulation_results.items():
    peak_monomer.append(np.max(data['result'][:, 1]))
    peak_oligomer.append(np.max(np.sum(data['result'][:, 2:17], axis=1)))
    peak_fibril.append(np.max(np.sum(data['result'][:, 17:25], axis=1)))
    peak_plaque.append(np.max(data['result'][:, 25]))
    peak_suvr.append(np.max(data['suvr']))

ax4.semilogy(k_rates_used, peak_monomer, 'o-', label='Monomer', linewidth=2, markersize=8)
ax4.semilogy(k_rates_used, peak_oligomer, 's-', label='Oligomer', linewidth=2, markersize=8)
ax4.semilogy(k_rates_used, peak_fibril, '^-', label='Fibril', linewidth=2, markersize=8)
ax4.semilogy(k_rates_used, peak_plaque, 'd-', label='Plaque', linewidth=2, markersize=8)
ax4.set_title('Peak Concentrations vs k_F24_O12_fortytwo', fontsize=16, fontweight='bold')
ax4.set_xlabel('k_F24_O12_fortytwo (1/h)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Peak Concentration (nM)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.4, linestyle=':')
ax4.legend()

# Plot 5: Final SUVR vs parameter value
ax5 = axes[2, 0]
ax5.semilogx(k_rates_used, final_suvr, 'o-', color='red', linewidth=2, markersize=8, label='Final SUVR')
ax5.set_title('Final SUVR vs k_F24_O12_fortytwo', fontsize=16, fontweight='bold')
ax5.set_xlabel('k_F24_O12_fortytwo (1/h)', fontsize=14, fontweight='bold')
ax5.set_ylabel('Final SUVR', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.4, linestyle=':')
ax5.legend()

# Plot 6: Peak SUVR vs parameter value
ax6 = axes[2, 1]
ax6.semilogx(k_rates_used, peak_suvr, 'o-', color='red', linewidth=2, markersize=8, label='Peak SUVR')
ax6.set_title('Peak SUVR vs k_F24_O12_fortytwo', fontsize=16, fontweight='bold')
ax6.set_xlabel('k_F24_O12_fortytwo (1/h)', fontsize=14, fontweight='bold')
ax6.set_ylabel('Peak SUVR', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.4, linestyle=':')
ax6.legend()

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(sensitivity_figures_dir, 'k_F24_O12_fortytwo_detailed_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\n=== k_F24_O12_fortytwo Sensitivity Analysis Summary ===")
print(f"Parameter range tested: {k_F24_O12_fortytwo_values.min():.3f} to {k_F24_O12_fortytwo_values.max():.3f}")
print(f"Successful simulations: {successful_sims}/{len(k_F24_O12_fortytwo_values)}")

if simulation_results:
    # Calculate summary statistics for final concentrations
    final_monomer_array = np.array(final_monomer)
    final_oligomer_array = np.array(final_oligomer)
    final_fibril_array = np.array(final_fibril)
    final_plaque_array = np.array(final_plaque)
    final_suvr_array = np.array(final_suvr)
    
    print(f"\nFinal concentration ranges:")
    print(f"  Monomer: {final_monomer_array.min():.2e} - {final_monomer_array.max():.2e} nM")
    print(f"  Oligomer: {final_oligomer_array.min():.2e} - {final_oligomer_array.max():.2e} nM")
    print(f"  Fibril: {final_fibril_array.min():.2e} - {final_fibril_array.max():.2e} nM")
    print(f"  Plaque: {final_plaque_array.min():.2e} - {final_plaque_array.max():.2e} nM")
    print(f"  SUVR: {final_suvr_array.min():.4f} - {final_suvr_array.max():.4f}")

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