import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from K_rates_extrapolate_Don_7_24 import calculate_k_rates


def calculate_suvr_at_year(result, suvr_func, year):
    """
    Find the model time points closest to year*365*24 and interpolate AB42 oligomer values
    to calculate SUVR at that specific timepoint.
    """
    target_time = year * 365 * 24  # target years in hours
    
    # Find the two closest time points
    model_times = result['time']
    time_diffs = np.abs(model_times - target_time)
    closest_indices = np.argsort(time_diffs)[:2]
    
    # Get the two closest time points
    t1, t2 = model_times[closest_indices[0]], model_times[closest_indices[1]]
    
    # Interpolate values for all AB42 oligomers (O2 through O25)
    interpolated_values = {}
    
    for i in range(2, 26):  # O2 through O25
        species_name = f'[AB42_O{i}_ISF]'
        if species_name in result.colnames:
            v1, v2 = result[species_name][closest_indices[0]], result[species_name][closest_indices[1]]
            
            # Linear interpolation
            if t1 != t2:
                interpolated_value = v1 + (v2 - v1) * (target_time - t1) / (t2 - t1)
            else:
                interpolated_value = v1
                
            interpolated_values[species_name] = interpolated_value
        else:
            # If species not available, set to 0
            interpolated_values[species_name] = 0.0
            print(f"{species_name}: Not available in simulation result")
    
    # Calculate oligomer weighted sum (O2-O17)
    oligomer_weighted_sum = interpolated_values['[AB42_O2_ISF]'] * 1
    for i in range(3, 18):
        oligomer_weighted_sum += interpolated_values[f'[AB42_O{i}_ISF]'] * (i-1)
    
    # Calculate proto weighted sum (O18-O24)
    proto_weighted_sum = interpolated_values['[AB42_O18_ISF]'] * 17
    for i in range(19, 25):
        proto_weighted_sum += interpolated_values[f'[AB42_O{i}_ISF]'] * (i-1)
    
    # Get plaque value (O25)
    plaque_sum = interpolated_values['[AB42_O25_ISF]']
    
    # Calculate SUVR at specified year
    suvr_at_year = suvr_func(oligomer_weighted_sum, proto_weighted_sum, plaque_sum)
    
    # Ensure SUVR is a scalar value
    if hasattr(suvr_at_year, '__len__'):
        suvr_at_year = suvr_at_year[0] if len(suvr_at_year) > 0 else 1.0
    
    return {
        'interpolated_values': interpolated_values,
        'oligomer_weighted_sum': oligomer_weighted_sum,
        'proto_weighted_sum': proto_weighted_sum,
        'plaque_sum': plaque_sum,
        'suvr_at_year': suvr_at_year,
        'closest_times': [t1, t2]
    }

def setup_and_run_simulation(r, param_values, microglia_params, ab42_params, ab40_params):
    """
    Helper function to set up model parameters and run a two-stage simulation.
    """
    try:
        # Reset and set optimized parameters
        r.reset()
        for name, value in param_values.items():
            r[name] = value
        
        # Set microglia parameters
        r['Microglia_EC50_AB42'] = microglia_params['EC50_AB42']
        r['Microglia_Vmax_AB42'] = microglia_params['Vmax_AB42']
        r['Microglia_EC50_AB40'] = microglia_params['EC50_AB40']
        r['Microglia_Vmax_AB40'] = microglia_params['Vmax_AB40']
        
        # Calculate and set AB42 rates
        rates_ab42 = calculate_k_rates(r['k_O1_O2_AB42_ISF'], r['k_O2_O3_AB42_ISF'], 
                                 r['k_O2_O1_AB42_ISF'], r['k_O3_O2_AB42_ISF'],
                                 forAsymp4x=ab42_params['forAsymp4x'], 
                                 forHill4x=ab42_params['forHill4x'], 
                                 backAsymp4x=ab42_params['backAsymp4x'], 
                                 backHill4x=ab42_params['backHill4x'])
        
        oligomer_sizes = list(range(4, 25))
        for size in oligomer_sizes:
            k_forward = f'k_O{size-1}_O{size}_AB42_ISF'
            r[k_forward] = rates_ab42[f'k_O{size-1}_O{size}_AB4x_ISF']
            k_backward = f'k_O{size}_O{size-1}_AB42_ISF'
            r[k_backward] = rates_ab42[f'k_O{size}_O{size-1}_AB4x_ISF']

        # Calculate and set AB40 rates
        rates_ab40 = calculate_k_rates(r['k_O1_O2_AB40_ISF'], r['k_O2_O3_AB40_ISF'], 
                                 r['k_O2_O1_AB40_ISF'], r['k_O3_O2_AB40_ISF'],
                                 forAsymp4x=ab40_params['forAsymp4x'], 
                                 forHill4x=ab40_params['forHill4x'], 
                                 backAsymp4x=ab40_params['backAsymp4x'], 
                                 backHill4x=ab40_params['backHill4x'])
        
        for size in oligomer_sizes:
            k_forward = f'k_O{size-1}_O{size}_AB40_ISF'
            r[k_forward] = rates_ab40[f'k_O{size-1}_O{size}_AB4x_ISF']
            k_backward = f'k_O{size}_O{size-1}_AB40_ISF'
            r[k_backward] = rates_ab40[f'k_O{size}_O{size-1}_AB4x_ISF']
        
        # Run simulation
        r.simulate(0, 20*365*24, 1000)
        
        selections = ['time', 
            '[AB42_O1_ISF]', '[AB42_O25_ISF]', 
            '[AB42_O2_ISF]', '[AB42_O3_ISF]', '[AB42_O4_ISF]', '[AB42_O5_ISF]', '[AB42_O6_ISF]', '[AB42_O7_ISF]', 
            '[AB42_O8_ISF]', '[AB42_O9_ISF]', '[AB42_O10_ISF]', '[AB42_O11_ISF]', '[AB42_O12_ISF]', '[AB42_O13_ISF]',
            '[AB42_O14_ISF]', '[AB42_O15_ISF]', '[AB42_O16_ISF]', '[AB42_O17_ISF]', '[AB42_O18_ISF]', '[AB42_O19_ISF]',
            '[AB42_O20_ISF]', '[AB42_O21_ISF]', '[AB42_O22_ISF]', '[AB42_O23_ISF]', '[AB42_O24_ISF]', '[AB42_O1_SAS]',
            '[AB40_O1_ISF]', '[AB40_O25_ISF]', 'AB42_IDE_Kcat_ISF','AB40_IDE_Kcat_ISF',
            '[AB40_O2_ISF]', '[AB40_O3_ISF]', '[AB40_O4_ISF]', '[AB40_O5_ISF]', '[AB40_O6_ISF]', '[AB40_O7_ISF]', 
            '[AB40_O8_ISF]', '[AB40_O9_ISF]', '[AB40_O10_ISF]', '[AB40_O11_ISF]', '[AB40_O12_ISF]', '[AB40_O13_ISF]',
            '[AB40_O14_ISF]', '[AB40_O15_ISF]', '[AB40_O16_ISF]', '[AB40_O17_ISF]', '[AB40_O18_ISF]', '[AB40_O19_ISF]',
            '[AB40_O20_ISF]', '[AB40_O21_ISF]', '[AB40_O22_ISF]', '[AB40_O23_ISF]', '[AB40_O24_ISF]', '[AB40_O1_SAS]',
            '[Antibody_centralAntibody]', '[Antibody_SAS]', '[AB42_O25__Antibody_ISF]', '[Antibody_ISF]', 'Anti_ABeta_ISF_sum',
            '[AB42_O1__Antibody_ISF]','[Antibody_BBB]','[Antibody_BCSFB]','[Antibody_LV]','[Antibody_TFV]','[Antibody_CM]',
            '[Antibody__FCRn_BBB]','[Antibody__FCRn_BCSFB]','Anti_ABeta_PVS_sum','Microglia','Microglia_high_frac',
            '[Antibody_BrainPlasma]','[FCRn_BBB]','[FCRn_BCSFB]','[Antibody_SubCutComp]']
        result = r.simulate(20*365*24, 100*365*24, 20000, selections)
        
        return result
        
    except Exception as e:
        print(f"Error in setup_and_run_simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_model_and_simulation():
    """
    Main function to run model and simulation without optimization.
    """
    # Define the parameters (using the optimized values from the original script)
    optimized_params = {
        'k_O1_O2_AB42_ISF': 0.0005828384337595838,
        'k_O2_O3_AB42_ISF': 0.002119837166351925,
        'k_O2_O1_AB42_ISF': 45.000801218271796,
        'k_O3_O2_AB42_ISF': 8.262779349556007e-06,
        'k_O24_O12_AB42_ISF': 5.230834584248408,
        'Baseline_AB42_O_P': 0.05999994963269613
    }

    # Microglia parameters
    Microglia_EC50_AB40_APOE4 = 20
    Microglia_Vmax_AB40_APOE4 = 0.0001
    Microglia_EC50_AB40_nonAPOE4 = 8
    Microglia_Vmax_AB40_nonAPOE4 = 0.00015
    Microglia_EC50_AB42_APOE4 = 300
    Microglia_Vmax_AB42_APOE4 = 0.0001
    Microglia_EC50_AB42_nonAPOE4 = 120
    Microglia_Vmax_AB42_nonAPOE4 = 0.00015

    apoe4_microglia_params = {
        'EC50_AB42': Microglia_EC50_AB42_APOE4,
        'Vmax_AB42': Microglia_Vmax_AB42_APOE4,
        'EC50_AB40': Microglia_EC50_AB40_APOE4,
        'Vmax_AB40': Microglia_Vmax_AB40_APOE4
    }
    
    nonapoe4_microglia_params = {
        'EC50_AB42': Microglia_EC50_AB42_nonAPOE4,
        'Vmax_AB42': Microglia_Vmax_AB42_nonAPOE4,
        'EC50_AB40': Microglia_EC50_AB40_nonAPOE4,
        'Vmax_AB40': Microglia_Vmax_AB40_nonAPOE4
    }
    
    ab42_params = {
        'forAsymp4x': 2.0,
        'forHill4x': 3.0,
        'backAsymp4x': 2.0,
        'backHill4x': 3.0
    }
    
    ab40_params = {
        'forAsymp4x': 0.3,
        'forHill4x': 2.0,
        'backAsymp4x': 0.3,
        'backHill4x': 2.0
    }

    # SUVR calculation function
    def suvr(oligo, proto, plaque, C1=2.5, C2=400000, C3=1.3, Hill=3.5):
        """
        Calculate SUVR using the provided formula.
        """
        numerator = oligo + proto + C3 * 24.0 * plaque
        denominator = numerator**Hill + C2**Hill
        
        # Handle scalar values properly
        if hasattr(denominator, '__len__'):
            # If it's an array
            if denominator.any() == 0:
                return 1.0  # Avoid division by zero
        else:
            # If it's a scalar
            if denominator == 0:
                return 1.0  # Avoid division by zero
                
        suvr = 1.0 + C1 * (numerator**Hill) / denominator
        return suvr

    # Load model
    r = te.loada('Antimony_Geerts_full_model_opt_5b.txt')
    r.setIntegrator('cvode')
    r.integrator.absolute_tolerance = 1e-8
    r.integrator.relative_tolerance = 1e-8
    r.integrator.setValue('stiff', True)

    # Load the CSV data
    csv_data_3 = pd.read_csv('Geerts 2023 Figure 3 units.csv')
    csv_data_3C = csv_data_3[csv_data_3['observation'].str.lower() == 'isf_ab42'].copy()
    csv_data_3A = csv_data_3[csv_data_3['observation'].str.lower() == 'suvr'].copy()
    csv_data_3B = csv_data_3[csv_data_3['observation'].str.lower() == 'plasma_abeta42_40_ratio'].copy()
    
    csv_data_3C_ApoE = csv_data_3C[csv_data_3C['series'].str.lower() == 'apoe4'].copy()
    csv_data_3C_nonApoE = csv_data_3C[csv_data_3C['series'].str.lower() == 'nonapoe4'].copy()
    
    csv_data_3A_ApoE = csv_data_3A[csv_data_3A['series'].str.lower() == 'apoe4'].copy()
    csv_data_3A_nonApoE = csv_data_3A[csv_data_3A['series'].str.lower() == 'nonapoe4'].copy()
    
    csv_data_3B_ApoE = csv_data_3B[csv_data_3B['series'].str.lower() == 'apoe4'].copy()
    csv_data_3B_nonApoE = csv_data_3B[csv_data_3B['series'].str.lower() == 'nonapoe4'].copy()

    csv_data_4A = pd.read_csv('Geerts 2023 Figure 4A.csv')
    csv_data_4A_model = csv_data_4A[csv_data_4A['series'].str.lower() == 'model'].copy()
    csv_data_4A_data = csv_data_4A[csv_data_4A['series'].str.lower() == 'data'].copy()

    # Run simulations with drug (current behavior)
    result1 = setup_and_run_simulation(r, optimized_params, apoe4_microglia_params, ab42_params, ab40_params)

    # Update model with optimized parameters for non-APOE4
    result2 = setup_and_run_simulation(r, optimized_params, nonapoe4_microglia_params, ab42_params, ab40_params)

    return result1, result2, csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, csv_data_4A_model, csv_data_4A_data

# Import plotting functions from the original script
def calculate_oligomer_sums(result, ab_type):
    """Calculate oligomer sums for a given AB type (AB42 or AB40)."""
    oligomer_sum = result[f'[{ab_type}_O2_ISF]']
    for i in range(3, 18):
        oligomer_sum += result[f'[{ab_type}_O{i}_ISF]']
    
    oligomer_weighted_sum = result[f'[{ab_type}_O2_ISF]'] * 1
    for i in range(3, 18):
        oligomer_weighted_sum += result[f'[{ab_type}_O{i}_ISF]'] * (i-1)
    
    return oligomer_sum, oligomer_weighted_sum

def calculate_proto_sums(result, ab_type):
    """Calculate proto sums for a given AB type (AB42 or AB40)."""
    proto_sum = result[f'[{ab_type}_O18_ISF]']
    for i in range(19, 25):
        proto_sum += result[f'[{ab_type}_O{i}_ISF]']
    
    proto_weighted_sum = result[f'[{ab_type}_O18_ISF]'] * 17
    for i in range(19, 25):
        proto_weighted_sum += result[f'[{ab_type}_O{i}_ISF]'] * (i-1)
    
    return proto_sum, proto_weighted_sum

def plot_oligomers(ax, result1, result2, time_years1, time_years2, ab_type):
    """Plot oligomers panel."""
    oligomer_sum1, oligomer_weighted_sum1 = calculate_oligomer_sums(result1, ab_type)
    oligomer_sum2, oligomer_weighted_sum2 = calculate_oligomer_sums(result2, ab_type)
    
    ax.plot(time_years1, oligomer_weighted_sum1, label=f'Oligomers weighted ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, oligomer_weighted_sum2, label=f'Oligomers weighted non-ApoE + drug', linewidth=2,  color='blue')
    
    
    ax.axvline(x=70, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='green', linestyle='--', linewidth=1.5)
    ax.plot([70], [12211], 'o', color='green', markersize=6)
    ax.plot([74], [12307], 'o', color='green', markersize=6)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Oligomers')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)

def plot_proto(ax, result1, result2, time_years1, time_years2, ab_type):
    """Plot proto panel."""
    proto_sum1, proto_weighted_sum1 = calculate_proto_sums(result1, ab_type)
    proto_sum2, proto_weighted_sum2 = calculate_proto_sums(result2, ab_type)
    
    ax.plot(time_years1, proto_weighted_sum1, label=f'Proto weighted ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, proto_weighted_sum2, label=f'Proto weighted non-ApoE + drug', linewidth=2, color='blue')
    
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='black', linestyle='--', linewidth=1.5)
    ax.plot([70], [70000], 'o', color='green', markersize=6)
    ax.plot([74], [70168], 'o', color='green', markersize=6)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Proto')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True)

def plot_suvr(ax, result1, result2, time_years1, time_years2, ab_type, suvr, csv_data_3A_ApoE, csv_data_3A_nonApoE):
    """Plot SUVR panel."""
    _, oligomer_weighted_sum1 = calculate_oligomer_sums(result1, ab_type)
    _, proto_weighted_sum1 = calculate_proto_sums(result1, ab_type)
    _, oligomer_weighted_sum2 = calculate_oligomer_sums(result2, ab_type)
    _, proto_weighted_sum2 = calculate_proto_sums(result2, ab_type)
    
    plaque_sum1 = result1[f'[{ab_type}_O25_ISF]']
    plaque_sum2 = result2[f'[{ab_type}_O25_ISF]']
    
    suvr_values1 = suvr(oligomer_weighted_sum1, proto_weighted_sum1, plaque_sum1)
    suvr_values2 = suvr(oligomer_weighted_sum2, proto_weighted_sum2, plaque_sum2)
    
    ax.plot(time_years1, suvr_values1, label='SUVR ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, suvr_values2, label='SUVR non-ApoE + drug', linewidth=2, color='blue')
    
    ax.axvline(x=70, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='green', linestyle='--', linewidth=1.5)
    ax.plot([70], [1.394], 'o', color='green', markersize=6)
    ax.plot([74], [1.623], 'o', color='green', markersize=6)
    ax.plot(csv_data_3A_ApoE['time']/24/365, csv_data_3A_ApoE['measurement'], 'r.', label='ApoE published', markersize=4)
    ax.plot(csv_data_3A_nonApoE['time']/24/365, csv_data_3A_nonApoE['measurement'], 'b.', label='non-ApoE published', markersize=4)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('SUVR')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)

def plot_o1_isf(ax, result1, result2, time_years1, time_years2, ab_type, csv_data_3C_ApoE, csv_data_3C_nonApoE):
    """Plot O1_ISF panel."""
    ax.plot(time_years1, result1[f'[{ab_type}_O1_ISF]'], label=f'{ab_type}_O1_ISF ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'[{ab_type}_O1_ISF]'], label=f'{ab_type}_O1_ISF non-ApoE + drug', linewidth=2, color='blue')
    
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    ax.plot(csv_data_3C_ApoE['time']/24/365, csv_data_3C_ApoE['measurement'], 'r.', label='ApoE published', markersize=4)
    ax.plot(csv_data_3C_nonApoE['time']/24/365, csv_data_3C_nonApoE['measurement'], 'b.', label='non-ApoE published', markersize=4)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title(f'{ab_type}_O1_ISF and {ab_type}_O1_CSF')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

def plot_plaque(ax, result1, result2, time_years1, time_years2, ab_type):
    """Plot plaque panel."""
    ax.plot(time_years1, result1[f'[{ab_type}_O25_ISF]'], label=f'{ab_type}_O25_ISF ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'[{ab_type}_O25_ISF]'], label=f'{ab_type}_O25_ISF non-ApoE + drug', linewidth=2, color='blue')
    
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='black', linestyle='--', linewidth=1.5)
    ax.plot([70], [5102], 'o', color='green', markersize=6)
    ax.plot([74], [6028], 'o', color='green', markersize=6)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Plaque')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

def plot_kcat(ax, result1, result2, time_years1, time_years2, ab_type):
    """Plot kcat panel."""
    ax.plot(time_years1, result1[f'AB42_IDE_Kcat_ISF'], label=f'AB42_IDE_Kcat_ISF ApoE', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'AB40_IDE_Kcat_ISF'], label=f'AB40_IDE_Kcat_ISF non-ApoE', linewidth=2, color='blue')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Kcat (1/s)')
    ax.set_title('Kcat')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True)

def create_plots(result1, result2, csv_data_3C_ApoE, csv_data_3C_nonApoE, 
                csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, 
                csv_data_4A_model, csv_data_4A_data):
    """Create the standard 6-panel plots."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Geerts Model Simulation Results (No Optimization)', fontsize=16)
    
    time_years1 = result1['time']/24/365
    time_years2 = result2['time']/24/365
    
    # Plot 1: Oligomers
    plot_oligomers(axes[0, 0], result1, result2, time_years1, time_years2, 'AB42')
    
    # Plot 2: Proto
    plot_proto(axes[0, 1], result1, result2, time_years1, time_years2, 'AB42')
    
    # Plot 3: SUVR
    plot_suvr(axes[1, 0], result1, result2, time_years1, time_years2, 'AB42', suvr, 
            csv_data_3A_ApoE, csv_data_3A_nonApoE)
    
    # Plot 4: O1_ISF
    plot_o1_isf(axes[1, 1], result1, result2, time_years1, time_years2, 'AB42', 
            csv_data_3C_ApoE, csv_data_3C_nonApoE)
    
    # Plot 5: Kcat
    plot_kcat(axes[2, 0], result1, result2, time_years1, time_years2, 'AB42')
    
    # Plot 6: Plaque
    plot_plaque(axes[2, 1], result1, result2, time_years1, time_years2, 'AB42')
    
    plt.tight_layout()
    plt.savefig('run_model_simple_AB42.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_suvr_and_centiloid_panels(result1, result2, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr_func):
    """
    Create a 2x2 plot:
    (1) SUVR vs time
    (2) d(SUVR)/dt vs time
    (3) Centiloid vs time
    (4) d(Centiloid)/dt vs time
    Also plot extracted data and their rates.
    """
    time1 = result1['time'] / (24*365)
    time2 = result2['time'] / (24*365)
    
    # Calculate SUVR time series
    _, oligomer_weighted_sum1 = calculate_oligomer_sums(result1, 'AB42')
    _, proto_weighted_sum1 = calculate_proto_sums(result1, 'AB42')
    _, oligomer_weighted_sum2 = calculate_oligomer_sums(result2, 'AB42')
    _, proto_weighted_sum2 = calculate_proto_sums(result2, 'AB42')
    plaque_sum1 = result1['[AB42_O25_ISF]']
    plaque_sum2 = result2['[AB42_O25_ISF]']
    suvr1 = suvr_func(oligomer_weighted_sum1, proto_weighted_sum1, plaque_sum1)
    suvr2 = suvr_func(oligomer_weighted_sum2, proto_weighted_sum2, plaque_sum2)
    
    # Calculate d(SUVR)/dt (per year)
    suvr1_rate = np.gradient(suvr1, time1)
    suvr2_rate = np.gradient(suvr2, time2)
    
    # Calculate Centiloid
    cl1 = 183 * suvr1 - 177
    cl2 = 183 * suvr2 - 177
    # d(Centiloid)/dt
    cl1_rate = np.gradient(cl1, time1)
    cl2_rate = np.gradient(cl2, time2)
    
    # Data points (ApoE and nonApoE)
    data_time_apoe = csv_data_3A_ApoE['time'].values / (24*365)
    data_suvr_apoe = csv_data_3A_ApoE['measurement'].values
    data_time_nonapoe = csv_data_3A_nonApoE['time'].values / (24*365)
    data_suvr_nonapoe = csv_data_3A_nonApoE['measurement'].values
    # d(SUVR)/dt for data (finite diff)
    if len(data_time_apoe) > 1:
        data_suvr_apoe_rate = np.gradient(data_suvr_apoe, data_time_apoe)
    else:
        data_suvr_apoe_rate = np.zeros_like(data_suvr_apoe)
    if len(data_time_nonapoe) > 1:
        data_suvr_nonapoe_rate = np.gradient(data_suvr_nonapoe, data_time_nonapoe)
    else:
        data_suvr_nonapoe_rate = np.zeros_like(data_suvr_nonapoe)
    # Centiloid for data
    data_cl_apoe = 183 * data_suvr_apoe - 177
    data_cl_nonapoe = 183 * data_suvr_nonapoe - 177
    # d(Centiloid)/dt for data
    if len(data_time_apoe) > 1:
        data_cl_apoe_rate = np.gradient(data_cl_apoe, data_time_apoe)
    else:
        data_cl_apoe_rate = np.zeros_like(data_cl_apoe)
    if len(data_time_nonapoe) > 1:
        data_cl_nonapoe_rate = np.gradient(data_cl_nonapoe, data_time_nonapoe)
    else:
        data_cl_nonapoe_rate = np.zeros_like(data_cl_nonapoe)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SUVR and Centiloid Analysis', fontsize=16)
    # Panel 1: SUVR vs time
    axes[0,0].plot(time1, suvr1, label='ApoE + drug', color='red')
    axes[0,0].plot(time2, suvr2, label='non-ApoE + drug', color='blue')
    axes[0,0].plot(data_time_apoe, data_suvr_apoe, 'r.', label='ApoE data', markersize=6)
    axes[0,0].plot(data_time_nonapoe, data_suvr_nonapoe, 'b.', label='non-ApoE data', markersize=6)
    axes[0,0].set_xlabel('Time (years)')
    axes[0,0].set_ylabel('SUVR')
    axes[0,0].set_title('SUVR vs Time')
    axes[0,0].legend()
    axes[0,0].grid(True)
    # Panel 2: d(SUVR)/dt vs time
    axes[0,1].plot(time1, suvr1_rate, label='ApoE + drug', color='red')
    axes[0,1].plot(time2, suvr2_rate, label='non-ApoE + drug', color='blue')
    axes[0,1].plot(data_time_apoe, data_suvr_apoe_rate, 'r.', label='ApoE data', markersize=6)
    axes[0,1].plot(data_time_nonapoe, data_suvr_nonapoe_rate, 'b.', label='non-ApoE data', markersize=6)
    axes[0,1].set_xlabel('Time (years)')
    axes[0,1].set_ylabel('d(SUVR)/dt (per year)')
    axes[0,1].set_title('SUVR Rate per Year')
    axes[0,1].legend()
    axes[0,1].grid(True)
    # Panel 3: Centiloid vs time
    axes[1,0].plot(time1, cl1, label='ApoE + drug', color='red')
    axes[1,0].plot(time2, cl2, label='non-ApoE + drug', color='blue')
    axes[1,0].plot(data_time_apoe, data_cl_apoe, 'r.', label='ApoE data', markersize=6)
    axes[1,0].plot(data_time_nonapoe, data_cl_nonapoe, 'b.', label='non-ApoE data', markersize=6)
    axes[1,0].set_xlabel('Time (years)')
    axes[1,0].set_ylabel('Centiloid')
    axes[1,0].set_title('Centiloid vs Time')
    axes[1,0].legend()
    axes[1,0].grid(True)
    # Panel 4: d(Centiloid)/dt vs time
    axes[1,1].plot(time1, cl1_rate, label='ApoE + drug', color='red')
    axes[1,1].plot(time2, cl2_rate, label='non-ApoE + drug', color='blue')
    axes[1,1].plot(data_time_apoe, data_cl_apoe_rate, 'r.', label='ApoE data', markersize=6)
    axes[1,1].plot(data_time_nonapoe, data_cl_nonapoe_rate, 'b.', label='non-ApoE data', markersize=6)
    axes[1,1].set_xlabel('Time (years)')
    axes[1,1].set_ylabel('d(Centiloid)/dt (per year)')
    axes[1,1].set_title('Centiloid Rate per Year')
    axes[1,1].legend()
    axes[1,1].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('run_model_simple_SUVR_Centiloid_panels.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Running model simulation without optimization...")
    
    # Run model and simulation
    results = run_model_and_simulation()
    
    # Check if we got valid results
    if results is None:
        print("Error: run_model_and_simulation() returned invalid results")
        exit(1)
    
    result1, result2, csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, csv_data_4A_model, csv_data_4A_data = results
    
    # Check if simulation results are valid
    if result1 is None or result2 is None:
        print("Error: Simulation failed - one or more results are None")
        exit(1)
    
    print(f"Simulation successful! All results have valid data.")
    
    # Calculate SUVR at 70 years
    print("\n" + "="*50)
    print("CALCULATING SUVR AT 70 YEARS")
    print("="*50)
    suvr_70_results = calculate_suvr_at_year(result1, suvr, 70)
    suvr_at_70 = suvr_70_results['suvr_at_year']
    print(f"\nFinal SUVR at 70 years: {suvr_at_70:.6f}")
    print("="*50)
    
    # Create plots
    create_plots(result1, result2, csv_data_3C_ApoE, csv_data_3C_nonApoE, 
                csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, 
                csv_data_4A_model, csv_data_4A_data)
    # New: Create SUVR/Centiloid 2x2 panel plot
    plot_suvr_and_centiloid_panels(result1, result2, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr) 