import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from K_rates_extrapolate_Don_7_24 import calculate_k_rates

def calculate_suvr_at_year(result, suvr_func, year):
    """
    Find the model time points closest to year*365*24 and interpolate AB42 oligomer values
    to calculate SUVR at that specific timepoint.
    
    Parameters:
    result: Simulation result containing time series data
    suvr_func: SUVR calculation function
    year: The target year to calculate SUVR for.
    
    Returns:
    dict: Dictionary containing interpolated values and calculated SUVR
    """
    target_time = year * 365 * 24  # target years in hours
    
    # Find the two closest time points
    model_times = result['time']
    time_diffs = np.abs(model_times - target_time)
    closest_indices = np.argsort(time_diffs)[:2]
    
    # Get the two closest time points
    t1, t2 = model_times[closest_indices[0]], model_times[closest_indices[1]]
    # print(f"Closest time points to {year} years: {t1/24/365:.2f} years and {t2/24/365:.2f} years")
    
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
            # print(f"{species_name}: {interpolated_value:.6f}")
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
    
    # print(f"\nCalculated values at {year} years:")
    # print(f"Oligomer weighted sum: {oligomer_weighted_sum:.6f}")
    # print(f"Proto weighted sum: {proto_weighted_sum:.6f}")
    # print(f"Plaque sum: {plaque_sum:.6f}")
    # print(f"SUVR at {year} years: {suvr_at_year:.6f}")
    
    return {
        'interpolated_values': interpolated_values,
        'oligomer_weighted_sum': oligomer_weighted_sum,
        'proto_weighted_sum': proto_weighted_sum,
        'plaque_sum': plaque_sum,
        'suvr_at_year': suvr_at_year,
        'closest_times': [t1, t2]
    }

def calculate_suvr_at_times(result, suvr_func, target_times):
    """
    Interpolate model results to calculate SUVR at specific timepoints.
    
    Parameters:
    result: Simulation result containing time series data
    suvr_func: SUVR calculation function
    target_times: Array of target times (in hours) to calculate SUVR for.
    
    Returns:
    np.array: Array of SUVR values.
    """
    model_times = result['time']
    
    # Interpolate all required species at the target times
    interpolated_species = {}
    for i in range(2, 26): # O2 to O25
        species_name = f'[AB42_O{i}_ISF]'
        if species_name in result.colnames:
            interpolated_species[species_name] = np.interp(target_times, model_times, result[species_name])
        else:
            interpolated_species[species_name] = np.zeros_like(target_times)
            print(f"{species_name}: Not available in simulation result")

    # Calculate oligomer weighted sum (O2-O17)
    oligomer_weighted_sum = interpolated_species['[AB42_O2_ISF]'] * 1
    for i in range(3, 18):
        oligomer_weighted_sum += interpolated_species[f'[AB42_O{i}_ISF]'] * (i-1)
    
    # Calculate proto weighted sum (O18-O24)
    proto_weighted_sum = interpolated_species['[AB42_O18_ISF]'] * 17
    for i in range(19, 25):
        proto_weighted_sum += interpolated_species[f'[AB42_O{i}_ISF]'] * (i-1)
    
    # Get plaque value (O25)
    plaque_sum = interpolated_species['[AB42_O25_ISF]']
    
    # Calculate SUVR at specified times
    suvr_values = suvr_func(oligomer_weighted_sum, proto_weighted_sum, plaque_sum)
    
    return suvr_values

def setup_and_run_simulation(r, param_values, microglia_params, ab42_params, ab40_params, for_mse=False):
    """
    Helper function to set up model parameters and run a two-stage simulation.
    
    Parameters:
    r: Tellurium model instance
    param_values: Dictionary of parameter names to values.
    microglia_params: Dictionary with microglia parameters.
    ab42_params: Dictionary with AB42 rate calculation parameters.
    ab40_params: Dictionary with AB40 rate calculation parameters.
    for_mse: Boolean, if true, uses selections for mse calculation.
    
    Returns:
    Simulation result
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
        
        if for_mse:
            selections = ['time', '[AB42_O1_ISF]', '[AB42_O2_ISF]', '[AB42_O3_ISF]', '[AB42_O4_ISF]', '[AB42_O5_ISF]', '[AB42_O6_ISF]', '[AB42_O7_ISF]', '[AB42_O8_ISF]', '[AB42_O9_ISF]', '[AB42_O10_ISF]', '[AB42_O11_ISF]', '[AB42_O12_ISF]', '[AB42_O13_ISF]', '[AB42_O14_ISF]', '[AB42_O15_ISF]', '[AB42_O16_ISF]', '[AB42_O17_ISF]', '[AB42_O18_ISF]', '[AB42_O19_ISF]', '[AB42_O20_ISF]', '[AB42_O21_ISF]', '[AB42_O22_ISF]', '[AB42_O23_ISF]', '[AB42_O24_ISF]', '[AB42_O25_ISF]']
            result = r.simulate(20*365*24, 100*365*24, 4380, selections)
        else:
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

def run_optimization_and_simulation():
    """
    Main function to run optimization and simulation.
    Wraps all functionality to avoid global variables.
    """
    # Define the parameters
    Microglia_EC50_AB40_APOE4 = 20 #Not APOE4; Microglia_EC50_AB42 has nanomol / L
    Microglia_Vmax_AB40_APOE4 = 0.0001 #Not APOE4; Microglia_Vmax_AB42 has nanomol / L / h
    Microglia_EC50_AB40_nonAPOE4 = 8 #Not APOE4; Microglia_EC50_AB42 has nanomol / L
    Microglia_Vmax_AB40_nonAPOE4 = 0.00015 #Not APOE4; Microglia_Vmax_AB42 has nanomol / L / h
    Microglia_EC50_AB42_APOE4 = 300 #Not APOE4; Microglia_EC50_AB42 has nanomol / L
    Microglia_Vmax_AB42_APOE4 = 0.0001 #Not APOE4; Microglia_Vmax_AB42 has nanomol / L / h
    Microglia_EC50_AB42_nonAPOE4 = 120 #Not APOE4; Microglia_EC50_AB42 has nanomol / L
    Microglia_Vmax_AB42_nonAPOE4 = 0.00015 #Not APOE4; Microglia_Vmax_AB42 has nanomol / L / h

    # Helper to interpolate model output to data times
    def interpolate_model_to_data_times(model_times, model_values, data_times):
        return np.interp(data_times, model_times, model_values)

    # Objective function for optimization
    def create_objective(csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, r, param_names, apoe4_microglia_params, nonapoe4_microglia_params, ab42_params, ab40_params):
        def objective(params):
            param_values = dict(zip(param_names, params))

            # Run for APOE4
            result1 = setup_and_run_simulation(r, param_values, apoe4_microglia_params, ab42_params, ab40_params, for_mse=True)
            if result1 is None:
                return 1e10 # Return a large error if simulation fails
            
            model_times = result1['time']
            model_values_isf = result1['[AB42_O1_ISF]']
            
            ISF_times_ApoE = csv_data_3C_ApoE['time'].values
            ISF_measurements_ApoE = csv_data_3C_ApoE['measurement'].values
            ISF_model_ApoE_at_data_times = interpolate_model_to_data_times(model_times, model_values_isf, ISF_times_ApoE)
            mse1 = np.sum((ISF_model_ApoE_at_data_times - ISF_measurements_ApoE) ** 2)
            
            suvr_times_apoe = csv_data_3A_ApoE['time'].values
            suvr_measurements_apoe = csv_data_3A_ApoE['measurement'].values
            model_suvr_apoe = calculate_suvr_at_times(result1, suvr, suvr_times_apoe)
            mse1 += np.sum((model_suvr_apoe - suvr_measurements_apoe) ** 2)

            # time = 70
            # suvr_results = calculate_suvr_at_year(result1, suvr, time)
            # mse1 = mse1 + (suvr_results['suvr_at_year']-1.394)**2 / 1.394
            # mse1 = mse1 + (suvr_results['plaque_sum']-5102)**2 / 5102 /10
            # mse1 = mse1 + (suvr_results['oligomer_weighted_sum']-12211)**2 / 12211 / 10
            # mse1 = mse1 + (suvr_results['proto_weighted_sum']-70000)**2 / 70000 / 10

            # Run for non-APOE4
            result2 = setup_and_run_simulation(r, param_values, nonapoe4_microglia_params, ab42_params, ab40_params, for_mse=True)
            if result2 is None:
                return 1e10 # Return a large error if simulation fails

            model_times_non = result2['time']
            model_values_isf_non = result2['[AB42_O1_ISF]']

            ISF_times_nonApoE = csv_data_3C_nonApoE['time'].values
            ISF_measurements_nonApoE = csv_data_3C_nonApoE['measurement'].values
            ISF_model_nonApoE_at_data_times = interpolate_model_to_data_times(model_times_non, model_values_isf_non, ISF_times_nonApoE)
            mse2 = np.sum((ISF_model_nonApoE_at_data_times - ISF_measurements_nonApoE) ** 2)
            
            suvr_times_nonapoe = csv_data_3A_nonApoE['time'].values
            suvr_measurements_nonapoe = csv_data_3A_nonApoE['measurement'].values
            model_suvr_nonapoe = calculate_suvr_at_times(result2, suvr, suvr_times_nonapoe)
            mse2 += np.sum((model_suvr_nonapoe - suvr_measurements_nonapoe) ** 2)

            # time = 74
            # suvr_results = calculate_suvr_at_year(result2, suvr, time)
            # mse2 = mse2 + (suvr_results['suvr_at_year']-1.623)**2 / 1.623
            # mse2 = mse2 + (suvr_results['plaque_sum']-6028)**2 / 6028 /10
            # mse2 = mse2 + (suvr_results['oligomer_weighted_sum']-12307)**2 / 12307 /10
            # mse2 = mse2 + (suvr_results['proto_weighted_sum']-70168)**2 / 70168 / 10
            
            mse = mse1 + mse2
            print(f"mse: {mse}, params: {param_values}")
            return mse
        return objective

    # SUVR calculation function
    def suvr(oligo, proto, plaque, C1=2.5, C2=400000, C3=1.3, Hill=3.5):
        """
        Calculate SUVR using the provided formula.
        
        Parameters:
        oligo, proto, plaque: input oligomer values
        C1, C2, C3, Hill: constants from the formula
        
        Returns:
        SUVR: predicted SUVR value
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
    r = te.loada(__file__.replace('.py', '.txt'))
    r.setIntegrator('cvode')
    r.integrator.absolute_tolerance = 1e-8
    r.integrator.relative_tolerance = 1e-8
    r.integrator.setValue('stiff', True)
    # print(r.getReactionIds())
    # print(r.getCurrentAntimony())
    # print(te.getODEsFromModel(r))
    # r.exportToSBML('Antimony_PBPK_model.xml') 

    # Load the CSV data
    csv_data_3 = pd.read_csv('Geerts 2023 Figure 3 units.csv')
    csv_data_3C = csv_data_3[csv_data_3['observation'].str.lower() == 'isf_ab42'].copy()
    csv_data_3A = csv_data_3[csv_data_3['observation'].str.lower() == 'suvr'].copy()
    csv_data_3B = csv_data_3[csv_data_3['observation'].str.lower() == 'plasma_abeta42_40_ratio'].copy()
    # For csv_data_3C
    
    csv_data_3C_ApoE = csv_data_3C[csv_data_3C['series'].str.lower() == 'apoe4'].copy()
    csv_data_3C_nonApoE = csv_data_3C[csv_data_3C['series'].str.lower() == 'nonapoe4'].copy()
    
    # For csv_data_3A
   
    csv_data_3A_ApoE = csv_data_3A[csv_data_3A['series'].str.lower() == 'apoe4'].copy()
    csv_data_3A_nonApoE = csv_data_3A[csv_data_3A['series'].str.lower() == 'nonapoe4'].copy()
    
    csv_data_3B_ApoE = csv_data_3B[csv_data_3B['series'].str.lower() == 'apoe4'].copy()
    csv_data_3B_nonApoE = csv_data_3B[csv_data_3B['series'].str.lower() == 'nonapoe4'].copy()

    csv_data_4A = pd.read_csv('Geerts 2023 Figure 4A.csv')
    csv_data_4A_model = csv_data_4A[csv_data_4A['series'].str.lower() == 'model'].copy()
    csv_data_4A_data = csv_data_4A[csv_data_4A['series'].str.lower() == 'data'].copy()
    # Define parameters to be optimized, with their bounds
    optimized_params = {
        'k_O1_O2_AB42_ISF': (1e-6, 0.0005820, 1),
        'k_O2_O3_AB42_ISF': (1e-6, 0.002119, 1),
        'k_O2_O1_AB42_ISF': (1e-3, 45, 1000),
        'k_O3_O2_AB42_ISF': (1e-15, 7.484e-06, 1),
        'k_O24_O12_AB42_ISF': (1, 5.23, 1000),
        'Baseline_AB42_O_P': (1e-8, 0.06, 1)
    }
    param_names = list(optimized_params.keys())
    param_values = [optimized_params[name][1] for name in param_names]  # Extract initial values
    bounds = [(optimized_params[name][0], optimized_params[name][2]) for name in param_names]  # Extract (lower, upper) bounds
    # Read 'Geerts 2023 Figure 7.csv'
    # csv_data_7 = pd.read_csv('Geerts 2023 Figure 7.csv')

    # csv_data_7_proto = csv_data_7[csv_data_7['observation'].str.lower() == 'proto'].copy()
    # csv_data_7_plaque = csv_data_7[csv_data_7['observation'].str.lower() == 'plaque'].copy()
    # csv_data_7_suvr = csv_data_7[csv_data_7['observation'].str.lower() == 'suvr'].copy()
    # csv_data_7_oligomer = csv_data_7[csv_data_7['observation'].str.lower() == 'oligomer'].copy()
    # csv_data_7_proto_placebo = csv_data_7_proto[csv_data_7_proto['series'].str.lower() == 'placebo'].copy()
    # csv_data_7_plaque_placebo = csv_data_7_plaque[csv_data_7_plaque['series'].str.lower() == 'placebo'].copy()
    # csv_data_7_suvr_placebo = csv_data_7_suvr[csv_data_7_suvr['series'].str.lower() == 'placebo'].copy()
    # csv_data_7_oligomer_placebo = csv_data_7_oligomer[csv_data_7_oligomer['series'].str.lower() == 'placebo'].copy()
 
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

    # # Create objective function with data
    objective = create_objective(csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, r, param_names, apoe4_microglia_params, nonapoe4_microglia_params, ab42_params, ab40_params)

    # # Initial guess (use current values from model)
    initial_guess = param_values  # Use the initial values from bounds
    print(f"Initial guess: {initial_guess}")
    
    # # Run optimization
    opt_result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    optimized_param_values = dict(zip(param_names, opt_result.x))
    print("\nOptimized Parameters:")
    for name, value in optimized_param_values.items():
        print(f"  {name}: {value}")

    # Run simulations with drug (current behavior)
    result1 = setup_and_run_simulation(r, optimized_param_values, apoe4_microglia_params, ab42_params, ab40_params)

    # Update model with optimized parameters for non-APOE4
    result2 = setup_and_run_simulation(r, optimized_param_values, nonapoe4_microglia_params, ab42_params, ab40_params)

    # Run simulations without drug (set Antibody_SubCutComp_Dose to 0.0)
    # First, save the original value
    no_drug_params = optimized_param_values.copy()
    no_drug_params['Antibody_SubCutComp_Dose'] = 0.0
    
    # Run ApoE4 simulation without drug
    result3 = setup_and_run_simulation(r, no_drug_params, apoe4_microglia_params, ab42_params, ab40_params)
    
    # Run nonApoE4 simulation without drug
    result4 = setup_and_run_simulation(r, no_drug_params, nonapoe4_microglia_params, ab42_params, ab40_params)
    
    return r, result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, csv_data_4A_model, csv_data_4A_data

def calculate_oligomer_sums(result, ab_type):
    """
    Calculate oligomer sums for a given AB type (AB42 or AB40).
    
    Parameters:
    result: Simulation result
    ab_type: 'AB42' or 'AB40'
    
    Returns:
    tuple: (oligomer_sum, oligomer_weighted_sum)
    """
    oligomer_sum = result[f'[{ab_type}_O2_ISF]']
    for i in range(3, 18):
        oligomer_sum += result[f'[{ab_type}_O{i}_ISF]']
    
    oligomer_weighted_sum = result[f'[{ab_type}_O2_ISF]'] * 1
    for i in range(3, 18):
        oligomer_weighted_sum += result[f'[{ab_type}_O{i}_ISF]'] * (i-1)
    
    return oligomer_sum, oligomer_weighted_sum

def calculate_proto_sums(result, ab_type):
    """
    Calculate proto sums for a given AB type (AB42 or AB40).
    
    Parameters:
    result: Simulation result
    ab_type: 'AB42' or 'AB40'
    
    Returns:
    tuple: (proto_sum, proto_weighted_sum)
    """
    proto_sum = result[f'[{ab_type}_O18_ISF]']
    for i in range(19, 25):
        proto_sum += result[f'[{ab_type}_O{i}_ISF]']
    
    proto_weighted_sum = result[f'[{ab_type}_O18_ISF]'] * 17
    for i in range(19, 25):
        proto_weighted_sum += result[f'[{ab_type}_O{i}_ISF]'] * (i-1)
    
    return proto_sum, proto_weighted_sum

def plot_oligomers(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot oligomers panel."""
    oligomer_sum1, oligomer_weighted_sum1 = calculate_oligomer_sums(result1, ab_type)
    oligomer_sum2, oligomer_weighted_sum2 = calculate_oligomer_sums(result2, ab_type)
    oligomer_sum3, oligomer_weighted_sum3 = calculate_oligomer_sums(result3, ab_type)
    oligomer_sum4, oligomer_weighted_sum4 = calculate_oligomer_sums(result4, ab_type)
    
    # ApoE with drug (red)
    ax.plot(time_years1, oligomer_sum1, label=f'Oligomers ApoE + drug', linewidth=2, linestyle='--', color='red')
    ax.plot(time_years1, oligomer_weighted_sum1, label=f'Oligomers weighted ApoE + drug', linewidth=2, color='red')
    
    # non-ApoE with drug (blue)
    ax.plot(time_years2, oligomer_sum2, label=f'Oligomers non-ApoE + drug', linewidth=2, linestyle='--', color='blue')
    ax.plot(time_years2, oligomer_weighted_sum2, label=f'Oligomers weighted non-ApoE + drug', linewidth=2,  color='blue')
    
    # ApoE without drug (green)
    ax.plot(time_years3, oligomer_sum3, label=f'Oligomers ApoE - drug', linewidth=2, linestyle='--', color='red')
    ax.plot(time_years3, oligomer_weighted_sum3, label=f'Oligomers weighted ApoE - drug', linewidth=2, linestyle='--', color='red')
    
    # non-ApoE without drug (teal)
    ax.plot(time_years4, oligomer_sum4, label=f'Oligomers non-ApoE - drug', linewidth=2, linestyle='--', color='teal')
    ax.plot(time_years4, oligomer_weighted_sum4, label=f'Oligomers weighted non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.axvline(x=70, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='green', linestyle='--', linewidth=1.5)
    ax.plot([70], [12211], 'o', color='green', markersize=6)
    ax.plot([74], [12307], 'o', color='green', markersize=6)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Oligomers')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)
    # ax.set_xlim(70, 74)
    # ax.set_ylim(0, 25000)

def plot_proto(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot proto panel."""
    proto_sum1, proto_weighted_sum1 = calculate_proto_sums(result1, ab_type)
    proto_sum2, proto_weighted_sum2 = calculate_proto_sums(result2, ab_type)
    proto_sum3, proto_weighted_sum3 = calculate_proto_sums(result3, ab_type)
    proto_sum4, proto_weighted_sum4 = calculate_proto_sums(result4, ab_type)
    
    # ApoE with drug (red)
    ax.plot(time_years1, proto_sum1, label=f'Proto ApoE + drug', linewidth=2, linestyle='--', color='red')
    ax.plot(time_years1, proto_weighted_sum1, label=f'Proto weighted ApoE + drug', linewidth=2, color='red')
    
    # non-ApoE with drug (blue)
    # ax.plot(time_years2, proto_sum2, label=f'Proto non-ApoE + drug', linewidth=2, linestyle='--', color='blue')
    ax.plot(time_years2, proto_weighted_sum2, label=f'Proto weighted non-ApoE + drug', linewidth=2, color='blue')
    
    # ApoE without drug (green)
    ax.plot(time_years3, proto_sum3, label=f'Proto ApoE - drug', linewidth=2, linestyle='--', color='green')
    ax.plot(time_years3, proto_weighted_sum3, label=f'Proto weighted ApoE - drug', linewidth=2, linestyle='--', color='red')
    
    # non-ApoE without drug (teal)
    ax.plot(time_years4, proto_sum4, label=f'Proto non-ApoE - drug', linewidth=2, linestyle='--', color='teal')
    ax.plot(time_years4, proto_weighted_sum4, label=f'Proto weighted non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='black', linestyle='--', linewidth=1.5)
    ax.plot([70], [70000], 'o', color='green', markersize=6)
    ax.plot([74], [70168], 'o', color='green', markersize=6)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Proto')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True)
    # ax.set_xlim(70, 74)
    # ax.set_ylim(30000, 75000)

def plot_suvr(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type, suvr, csv_data_3A_ApoE, csv_data_3A_nonApoE):
    """Plot SUVR panel."""
    _, oligomer_weighted_sum1 = calculate_oligomer_sums(result1, ab_type)
    _, proto_weighted_sum1 = calculate_proto_sums(result1, ab_type)
    _, oligomer_weighted_sum2 = calculate_oligomer_sums(result2, ab_type)
    _, proto_weighted_sum2 = calculate_proto_sums(result2, ab_type)
    _, oligomer_weighted_sum3 = calculate_oligomer_sums(result3, ab_type)
    _, proto_weighted_sum3 = calculate_proto_sums(result3, ab_type)
    _, oligomer_weighted_sum4 = calculate_oligomer_sums(result4, ab_type)
    _, proto_weighted_sum4 = calculate_proto_sums(result4, ab_type)
    
    plaque_sum1 = result1[f'[{ab_type}_O25_ISF]']
    plaque_sum2 = result2[f'[{ab_type}_O25_ISF]']
    plaque_sum3 = result3[f'[{ab_type}_O25_ISF]']
    plaque_sum4 = result4[f'[{ab_type}_O25_ISF]']
    
    suvr_values1 = suvr(oligomer_weighted_sum1, proto_weighted_sum1, plaque_sum1)
    suvr_values2 = suvr(oligomer_weighted_sum2, proto_weighted_sum2, plaque_sum2)
    suvr_values3 = suvr(oligomer_weighted_sum3, proto_weighted_sum3, plaque_sum3)
    suvr_values4 = suvr(oligomer_weighted_sum4, proto_weighted_sum4, plaque_sum4)
    
    # ApoE with drug (red)
    ax.plot(time_years1, suvr_values1, label='SUVR ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, suvr_values2, label='SUVR non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, suvr_values3, label='SUVR ApoE - drug', linestyle='--',color='red',linewidth=2)
    # non-ApoE without drug (teal)
    ax.plot(time_years4, suvr_values4, label='SUVR non-ApoE - drug', linewidth=2, linestyle='--',color='blue')
    
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
    # ax.set_xlim(70, 74)
    # ax.set_ylim(0.8, 1.8)

def plot_microglia(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot microglia panel."""
    ax.plot(time_years1, result1[f'Microglia'], label=f'Microglia ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'Microglia'], label=f'Microglia non-ApoE + drug', linewidth=2, color='blue')
    ax.plot(time_years3, result3[f'Microglia'], label=f'Microglia ApoE - drug', linewidth=2, linestyle='--', color='red')
    ax.plot(time_years4, result4[f'Microglia'], label=f'Microglia non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Microglia')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)
    # ax.set_xlim(70, 74)
    # ax.set_ylim(0, 1000)

def plot_microglia_high_frac(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot microglia high frac panel."""
    ax.plot(time_years1, result1[f'Microglia_high_frac'], label=f'Microglia high frac ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'Microglia_high_frac'], label=f'Microglia high frac non-ApoE + drug', linewidth=2, color='blue')
    ax.plot(time_years3, result3[f'Microglia_high_frac'], label=f'Microglia high frac ApoE - drug', linewidth=2, linestyle='--', color='red')
    ax.plot(time_years4, result4[f'Microglia_high_frac'], label=f'Microglia high frac non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Microglia high frac')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)
    # ax.set_xlim(50, 74)
    # ax.set_ylim(0, 1000)

def plot_antibody_central(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type, csv_data_4A_model, csv_data_4A_data):
    """Plot antibody central panel."""
    # Check if the column exists in the results
    column_name = f'[Antibody_centralAntibody]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody Central (Not Available)')
        return
    
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'{ab_type}_centralAntibody ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'{ab_type}_centralAntibody non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'{ab_type}_centralAntibody ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'{ab_type}_centralAntibody non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    ax.plot(csv_data_4A_data['time'], csv_data_4A_data['measurement'], 'r.', label='Gant data', markersize=4)
    ax.plot(csv_data_4A_model['time'], csv_data_4A_model['measurement'], 'r-', label='Gant model', markersize=4)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody Central')
    ax.legend(loc='upper right', fontsize=8)
    # ax.set_xlim(70, 70.3)
    ax.set_xlim(70, 74)
    ax.grid(True)
    
    # ax.set_yscale('log')
    # ax.set_ylim(1, 1e3)

def plot_antibody_BrainPlasma(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody BrainPlasma panel."""
    column_name = f'[Antibody_BrainPlasma]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody BrainPlasma (Not Available)')
        return
    
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'{ab_type}_BrainPlasma ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'{ab_type}_BrainPlasma non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'{ab_type}_BrainPlasma ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'{ab_type}_BrainPlasma non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    # ax.plot(csv_data_4A_data['time'], csv_data_4A_data['measurement'], 'r.', label='Gant data', markersize=4)
    # ax.plot(csv_data_4A_model['time'], csv_data_4A_model['measurement'], 'r-', label='Gant model', markersize=4)
    csv_data_4B = pd.read_csv('Geerts 2023 Figure 4B.csv')
    csv_data_4B_10 = csv_data_4B[csv_data_4B['series'].str.lower() == 'sim10mgkg'].copy()
    ax.plot(csv_data_4B_10['time'], csv_data_4B_10['measurement'], 'g.', label='Geerts data', markersize=4)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody BrainPlasma')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(70, 70.3)
    ax.set_yscale('log')
    ax.set_ylim(1, 1e4)
    ax.grid(True)

def plot_antibody_SAS(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody SAS panel."""
    # Check if the column exists in the results
    column_name = f'[Antibody_SAS]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody SAS (Not Available)')
        return
    
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'{ab_type}_SAS ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'{ab_type}_SAS non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'{ab_type}_SAS ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'{ab_type}_SAS non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody SAS')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(70, 73)
    ax.grid(True)

def plot_antibody_ISF(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody ISF panel."""
    column_name = f'Anti_ABeta_ISF_sum'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody ISF (Not Available)')
        return
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name]+result1[f'[AB42_O1__Antibody_ISF]'], label=f'{ab_type}_ISF ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name]+result2[f'[AB42_O1__Antibody_ISF]'], label=f'{ab_type}_ISF non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name]+result3[f'[AB42_O1__Antibody_ISF]'], label=f'{ab_type}_ISF ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name]+result4[f'[AB42_O1__Antibody_ISF]'], label=f'{ab_type}_ISF non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody total ISF')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(70, 73)
    ax.grid(True)

def plot_antibody_BBB(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody BBB panel."""
    column_name = f'[Antibody__FCRn_BBB]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody__FCRn_BBB (Not Available)')
        return
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'Antibody__FCRn_BBB ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'Antibody__FCRn_BBB non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'Antibody__FCRn_BBB ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'Antibody__FCRn_BBB non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody__FCRn_BBB')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(70, 73)
    ax.grid(True)

def plot_FCRn_BBB(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody BBB panel."""
    column_name = f'[FCRn_BBB]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('FCRn_BBB (Not Available)')
        return
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'FCRn_BBB ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'FCRn_BBB non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'FCRn_BBB ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'FCRn_BBB non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('FCRn_BBB')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(70, 73)
    ax.grid(True)

def plot_antibody_BCSFB(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody BCSFB panel."""
    column_name = f'[Antibody__FCRn_BCSFB]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody__FCRn_BCSFB (Not Available)')
        return
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'Antibody__FCRn_BCSFB ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'Antibody__FCRn_BCSFB non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'Antibody__FCRn_BCSFB ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'Antibody__FCRn_BCSFB non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody__FCRn_BCSFB')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    ax.set_xlim(70, 73)

def plot_antibody_SubCutComp(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody SubCutComp panel."""
    column_name = f'[Antibody_SubCutComp]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody_SubCutComp (Not Available)')
        return
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'Antibody_SubCutComp ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'Antibody_SubCutComp non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'Antibody_SubCutComp ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'Antibody_SubCutComp non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody_SubCutComp')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    ax.set_xlim(70, 70.3)

def plot_antibody_LV(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody LV panel."""
    column_name = f'[Antibody_LV]'
    if column_name not in result1.colnames:
        print(f"Warning: Column {column_name} not found in simulation results")
        print(f"Available columns: {result1.colnames}")
        # Plot a placeholder or skip this plot
        ax.text(0.5, 0.5, f'{column_name} not available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12)
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Concentration (nM)')
        ax.set_title('Antibody LV (Not Available)')
        return
    # ApoE with drug (red)
    ax.plot(time_years1, result1[column_name], label=f'Antibody_LV ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[column_name], label=f'Antibody_LV non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[column_name], label=f'Antibody_LV ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[column_name], label=f'Antibody_LV non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Antibody LV')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    ax.set_xlim(70, 73)

def plot_o1_isf(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type, csv_data_3C_ApoE, csv_data_3C_nonApoE):
    """Plot O1_ISF panel."""
    # ApoE with drug (red)
    ax.plot(time_years1, result1[f'[{ab_type}_O1_ISF]'], label=f'{ab_type}_O1_ISF ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[f'[{ab_type}_O1_ISF]'], label=f'{ab_type}_O1_ISF non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[f'[{ab_type}_O1_ISF]'], label=f'{ab_type}_O1_ISF ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[f'[{ab_type}_O1_ISF]'], label=f'{ab_type}_O1_ISF non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    ax.plot(csv_data_3C_ApoE['time']/24/365, csv_data_3C_ApoE['measurement'], 'r.', label='ApoE published', markersize=4)
    ax.plot(csv_data_3C_nonApoE['time']/24/365, csv_data_3C_nonApoE['measurement'], 'b.', label='non-ApoE published', markersize=4)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title(f'{ab_type}_O1_ISF and {ab_type}_O1_CSF')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

def plot_antibody_SAS_sum(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot antibody SAS sum panel."""
    ax.plot(time_years1, result1[f'Anti_ABeta_PVS_sum'], label=f'Anti_ABeta_PVS_sum ApoE + drug', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'Anti_ABeta_PVS_sum'], label=f'Anti_ABeta_PVS_sum non-ApoE + drug', linewidth=2, color='blue')
    ax.plot(time_years3, result3[f'Anti_ABeta_PVS_sum'], label=f'Anti_ABeta_PVS_sum ApoE - drug', linewidth=2, linestyle='--', color='red')
    ax.plot(time_years4, result4[f'Anti_ABeta_PVS_sum'], label=f'Anti_ABeta_PVS_sum non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title(f'Anti_ABeta_PVS_sum')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    ax.set_xlim(70, 73)

def plot_plaque(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type):
    """Plot plaque panel."""
    # ApoE with drug (red)
    ax.plot(time_years1, result1[f'[{ab_type}_O25_ISF]'], label=f'{ab_type}_O25_ISF ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2[f'[{ab_type}_O25_ISF]'], label=f'{ab_type}_O25_ISF non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3[f'[{ab_type}_O25_ISF]'], label=f'{ab_type}_O25_ISF ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4[f'[{ab_type}_O25_ISF]'], label=f'{ab_type}_O25_ISF non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    ax.axvline(x=74, color='black', linestyle='--', linewidth=1.5)
    ax.plot([70], [5102], 'o', color='green', markersize=6)
    ax.plot([74], [6028], 'o', color='green', markersize=6)
    
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Concentration (nM)')
    ax.set_title('Plaque')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    # ax.set_xlim(70, 74)
    # ax.set_ylim(0, 7500)

def plot_ratio(ax, result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, csv_data_3B_ApoE, csv_data_3B_nonApoE):
    """Plot AB42/AB40 ratio panel."""
    # ApoE with drug (red)
    ax.plot(time_years1, result1['[AB42_O1_SAS]']/result1['[AB40_O1_SAS]'], 
            label='AB42_O1_SAS/AB40_O1_SAS ApoE + drug', linewidth=2, color='red')
    # non-ApoE with drug (blue)
    ax.plot(time_years2, result2['[AB42_O1_SAS]']/result2['[AB40_O1_SAS]'], 
            label='AB42_O1_SAS/AB40_O1_SAS non-ApoE + drug', linewidth=2, color='blue')
    # ApoE without drug (green)
    ax.plot(time_years3, result3['[AB42_O1_SAS]']/result3['[AB40_O1_SAS]'], 
            label='AB42_O1_SAS/AB40_O1_SAS ApoE - drug', linewidth=2, linestyle='--', color='red')
    # non-ApoE without drug (teal)
    ax.plot(time_years4, result4['[AB42_O1_SAS]']/result4['[AB40_O1_SAS]'], 
            label='AB42_O1_SAS/AB40_O1_SAS non-ApoE - drug', linewidth=2, linestyle='--', color='blue')
    
    ax.plot(csv_data_3B_ApoE['time']/24/365, csv_data_3B_ApoE['measurement'], 'r.', 
            label='ApoE published', markersize=4)
    ax.plot(csv_data_3B_nonApoE['time']/24/365, csv_data_3B_nonApoE['measurement'], 'b.', 
            label='non-ApoE published', markersize=4)
    ax.axvline(x=70, color='black', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('Time (years)')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylabel('Ratio CSF AB42/AB40')

def plot_kcat(ax, result1, result2, time_years1, time_years2, ab_type):
    """Plot kcat panel."""
    ax.plot(time_years1, result1[f'AB42_IDE_Kcat_ISF'], label=f'AB42_IDE_Kcat_ISF ApoE', linewidth=2, color='red')
    ax.plot(time_years2, result2[f'AB40_IDE_Kcat_ISF'], label=f'AB40_IDE_Kcat_ISF non-ApoE', linewidth=2, color='blue')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Kcat (1/s)')
    ax.set_title('Kcat')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True)
    # ax.set_xlim(70, 73)

def create_standard_plots(result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, 
                         csv_data_3A_ApoE, csv_data_3A_nonApoE, csv_data_4A_model, csv_data_4A_data, suvr, ab_type, filename_suffix):
    """
    Create standard 6-panel plots for a given AB type.
    
    Parameters:
    result1, result2, result3, result4: Simulation results for ApoE with drug, non-ApoE with drug, ApoE without drug, non-ApoE without drug
    csv_data_*: CSV data for plotting
    suvr: SUVR calculation function
    ab_type: 'AB42' or 'AB40'
    filename_suffix: Suffix for the output filename
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Geerts Model Simulation Results', fontsize=16)
    
    time_years1 = result1['time']/24/365
    time_years2 = result2['time']/24/365
    time_years3 = result3['time']/24/365
    time_years4 = result4['time']/24/365
    
    # Plot 1: Oligomers
    if ab_type == 'AB42':
        plot_oligomers(axes[0, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    elif ab_type == 'AB40':
        plot_microglia(axes[0, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    elif ab_type == 'Antibody':
        plot_antibody_ISF(axes[0, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    
    # Plot 2: Proto
    if ab_type == 'AB42':
        plot_proto(axes[0, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    elif ab_type == 'Antibody':
        plot_antibody_central(axes[0, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type, csv_data_4A_model, csv_data_4A_data)
    elif ab_type == 'AB40':
        plot_microglia_high_frac(axes[0, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    
    # Plot 3: SUVR
    if ab_type != 'Antibody':
        plot_suvr(axes[1, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type, suvr, 
                csv_data_3A_ApoE, csv_data_3A_nonApoE)
    else:
        plot_antibody_SAS(axes[1, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    
    # Plot 4: O1_ISF
    if ab_type != 'Antibody':
        plot_o1_isf(axes[1, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type, 
                csv_data_3C_ApoE, csv_data_3C_nonApoE)
    else:
        # plot_antibody_BBB(axes[1, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
        plot_FCRn_BBB(axes[1, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    
    # Plot 5: Ratio (only for AB40 figure)
    if ab_type == 'AB40':
        plot_ratio(axes[2, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, 
                   csv_data_3B_ApoE, csv_data_3B_nonApoE)
    elif ab_type == 'AB42':
        plot_kcat(axes[2, 0], result1, result2, time_years1, time_years2, ab_type)
    elif ab_type == 'Antibody':
        # plot_antibody_BCSFB(axes[2, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
        # plot_antibody_SAS_sum(axes[2, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
        plot_antibody_BrainPlasma(axes[2, 0], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    # Plot 6: Plaque
    if ab_type != 'Antibody':
        plot_plaque(axes[2, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    else:
        # plot_antibody_LV(axes[2, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
        plot_antibody_SubCutComp(axes[2, 1], result1, result2, result3, result4, time_years1, time_years2, time_years3, time_years4, ab_type)
    
    plt.tight_layout()
    plt.savefig(__file__.replace('.py', f'_{filename_suffix}.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_plots(r, result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, csv_data_4A_model, csv_data_4A_data):
    """
    Create plots using the simulation results.
    """
    # Create AB42 plots
    create_standard_plots(result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, 
                         csv_data_3A_ApoE, csv_data_3A_nonApoE, csv_data_4A_model, csv_data_4A_data, suvr, 'AB42', 'AB42')
    
    # # Create AB40 plots
    # create_standard_plots(result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, 
    #                      csv_data_3A_ApoE, csv_data_3A_nonApoE, csv_data_4A_model, csv_data_4A_data, suvr, 'AB40', 'AB40')
    
    # # Create Antibody plots
    # create_standard_plots(result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, 
    #                      csv_data_3A_ApoE, csv_data_3A_nonApoE, csv_data_4A_model, csv_data_4A_data, suvr, 'Antibody', 'Antibody')

# Main execution
if __name__ == "__main__":
    # Run optimization and simulation
    results = run_optimization_and_simulation()
    
    # Check if we got valid results
    if results is None :
        print("Error: run_optimization_and_simulation() returned invalid results")
        exit(1)
    
    r, result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, csv_data_4A_model, csv_data_4A_data = results
    
    # Check if simulation results are valid
    if result1 is None or result2 is None or result3 is None or result4 is None:
        print("Error: Simulation failed - one or more results are None")
        exit(1)
    
    # Check if results have the expected structure
    if not hasattr(result1, 'colnames') or not hasattr(result2, 'colnames') or not hasattr(result3, 'colnames') or not hasattr(result4, 'colnames'):
        print(f"Error: Simulation results are not valid. result1 type: {type(result1)}, result2 type: {type(result2)}, result3 type: {type(result3)}, result4 type: {type(result4)}")
        exit(1)
    
    print(f"Simulation successful! result1 has {len(result1.colnames)} columns, result2 has {len(result2.colnames)} columns, result3 has {len(result3.colnames)} columns, result4 has {len(result4.colnames)} columns")
    
    # # Debug: Print available columns
    # print("\nAvailable columns in result1:")
    # for i, col in enumerate(result1.colnames):
    #     print(f"  {i}: {col}")
    
    # print("\nAvailable columns in result2:")
    # for i, col in enumerate(result2.colnames):
    #     print(f"  {i}: {col}")
    
    # print("\nAvailable columns in result3:")
    # for i, col in enumerate(result3.colnames):
    #     print(f"  {i}: {col}")
    
    # print("\nAvailable columns in result4:")
    # for i, col in enumerate(result4.colnames):
    #     print(f"  {i}: {col}")
    
    # Calculate SUVR at 70 years and store interpolated values
    print("\n" + "="*50)
    print("CALCULATING SUVR AT 70 YEARS")
    print("="*50)
    suvr_70_results = calculate_suvr_at_year(result1, suvr, 70)
    
    # Store the results for further use
    interpolated_values = suvr_70_results['interpolated_values']
    oligomer_weighted_sum_70 = suvr_70_results['oligomer_weighted_sum']
    proto_weighted_sum_70 = suvr_70_results['proto_weighted_sum']
    plaque_sum_70 = suvr_70_results['plaque_sum']
    suvr_at_70 = suvr_70_results['suvr_at_year']
    
    print(f"\nFinal SUVR at 70 years: {suvr_at_70:.6f}")
    print("="*50)
    
    # Create plots
    create_plots(r, result1, result2, result3, result4, csv_data_3C_ApoE, csv_data_3C_nonApoE, csv_data_3A_ApoE, csv_data_3A_nonApoE, suvr, csv_data_3B_ApoE, csv_data_3B_nonApoE, csv_data_4A_model, csv_data_4A_data) 