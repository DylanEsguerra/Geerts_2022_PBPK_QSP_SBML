import jax.numpy as jnp
import pandas as pd
import os
import sys

# Add parent directory to path to import K_rates_extrapolate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from K_rates_extrapolate import calculate_k_rates

def load_parameters(antibody_name='Gant', **agg_kwargs):
    """
    Load and process all parameters from various sources.
    
    Args:
        antibody_name: Name of the antibody to use ('Gant' or 'Lec')
        **agg_kwargs: Optional overrides for aggregation parameters (e.g., forAsymp42, backAsymp42, BackHill42, original_kb0_fortytwo, original_kb1_fortytwo, baseline_ab40_plaque_rate, baseline_ab42_plaque_rate, enable_forward_rate_multiplier)
    
    Returns:
        Dictionary of all parameters as JAX arrays
    """
    # Load parameters from Geerts_Params_2.csv
    geerts_params_df = pd.read_csv('./Geerts_Params_2.csv')
    
    # Create initial parameter dictionary (case-sensitive)
    params = {}
    params.update(dict(zip(geerts_params_df['name'], geerts_params_df['value'])))
    
    # Create case-insensitive parameter mapping
    case_insensitive_params = {}
    for name, value in zip(geerts_params_df['name'], geerts_params_df['value']):
        case_insensitive_params[name.lower()] = (name, value)
    
    # Extract rate-related parameters for calculate_k_rates
    rate_params = {
        'original_kf0_forty': params['original_kf0_forty'],
        'original_kf0_fortytwo': params['original_kf0_fortytwo'],
        'original_kf1_forty': params['original_kf1_forty'],
        'original_kf1_fortytwo': params['original_kf1_fortytwo'],
        'original_kb0_forty': params['original_kb0_forty'],
        'original_kb0_fortytwo': params['original_kb0_fortytwo'],
        'original_kb1_forty': params['original_kb1_forty'],
        'original_kb1_fortytwo': params['original_kb1_fortytwo'],
        'forAsymp40': params['forAsymp40'],
        'forAsymp42': params['forAsymp42'],
        'backAsymp40': params['backAsymp40'],
        'backAsymp42': params['backAsymp42'],
        'forHill40': params['forHill40'],
        'forHill42': params['forHill42'],
        'BackHill40': params['BackHill40'],
        'BackHill42': params['BackHill42'],
        'rate_cutoff': params.get('rate_cutoff', 0.00001),
        # Add plaque formation parameters
        'baseline_ab40_plaque_rate': params.get('k_O13_Plaque_forty', 0.000005),  # Use existing parameter as baseline
        'baseline_ab42_plaque_rate': params.get('k_O13_Plaque_fortytwo', 0.00005),  # Use existing parameter as baseline
        'enable_plaque_forward_rate_multiplier': True  # Default to enabled
    }
    
    # Only pass keys to calculate_k_rates that it expects
    k_rates_keys = {
        'original_kf0_forty', 'original_kf0_fortytwo', 'original_kf1_forty', 'original_kf1_fortytwo',
        'original_kb0_forty', 'original_kb0_fortytwo', 'original_kb1_forty', 'original_kb1_fortytwo',
        'forAsymp40', 'forAsymp42', 'backAsymp40', 'backAsymp42',
        'forHill40', 'forHill42', 'BackHill40', 'BackHill42', 'rate_cutoff',
        'baseline_ab40_plaque_rate', 'baseline_ab42_plaque_rate', 'enable_plaque_forward_rate_multiplier'
    }
    
    # Apply overrides from agg_kwargs
    for k, v in agg_kwargs.items():
        if k in k_rates_keys:
            rate_params[k] = v
    
    # Calculate all rates including plaque rates
    rates_dict = calculate_k_rates(**rate_params)
    params.update(rates_dict)
    
    # Add any extra overrides (not used by calculate_k_rates) to params
    for k, v in agg_kwargs.items():
        if k not in k_rates_keys:
            params[k] = v
    
    # Add antibody-specific parameters with correct naming
    if antibody_name == 'Gant':
        antibody_specific_params = {
            'fta0': ('Gant_fta0', None),
            'fta1': ('Gant_fta1', None),
            'fta2': ('Gant_fta2', None),
            'fta3': ('Gant_fta3', None),
            'PK_CL': ('CL_Gantenerumab', None),
            'PK_CLd2': ('CLd2_Gantenerumab', None),
            'PK_Vcent': ('Vcent_Gantenerumab', None),
            'PK_Vper': ('Vper_Gantenerumab', None),
            'PK_SC_ka': ('SC_ka_Gantenerumab', None),
            'PK_SC_bio': ('SC_bio_Gantenerumab', None)
        }
    else:  # 'Lec'
        antibody_specific_params = {
            'fta0': ('Lec_fta0', None),
            'fta1': ('Lec_fta1', None),
            'fta2': ('Lec_fta2', None),
            'fta3': ('Lec_fta3', None),
            'PK_CL': ('CL_BAN2401', None),
            'PK_CLd2': ('CLd2_BAN2401', None),
            'PK_Vcent': ('Vcent_BAN2401', None),
            'PK_Vper': ('Vper_BAN2401', None),
            'PK_SC_ka': ('Lec_SC_ka', None),
            'PK_SC_bio': ('Lec_SC_bio', None)
        }
    
    # Map antibody-specific parameters
    for param_name, (source_param, _) in antibody_specific_params.items():
        if source_param in params:
            params[param_name] = params[source_param]
        else:
            print(f"Warning: Parameter '{source_param}' not found for antibody {antibody_name}")
    
    # Convert all parameter values to JAX arrays
    for key in params:
        if isinstance(params[key], (int, float)):
            params[key] = jnp.array(params[key])
    
    # Create a class for case-insensitive parameter access
    class CaseInsensitiveParams:
        def __init__(self, params_dict, case_insensitive_dict):
            self.params = params_dict
            self.case_map = case_insensitive_dict
        
        def __getitem__(self, key):
            # First try direct access with the exact key
            if key in self.params:
                return self.params[key]
            
            # If not found, try case-insensitive lookup
            key_lower = key.lower()
            if key_lower in self.case_map:
                original_key, _ = self.case_map[key_lower]
                return self.params[original_key]
            
            # If trying to get a 'k_' parameter, try with 'K_' and vice versa
            if key.startswith('k_') or key.startswith('K_'):
                alternate_key = 'K' + key[1:] if key.startswith('k') else 'k' + key[1:]
                if alternate_key in self.params:
                    return self.params[alternate_key]
                
                # Try case-insensitive lookup for the alternate key
                alternate_key_lower = alternate_key.lower()
                if alternate_key_lower in self.case_map:
                    original_key, _ = self.case_map[alternate_key_lower]
                    return self.params[original_key]
            
            # If we reach here, the parameter is truly not found
            raise KeyError(f"Parameter '{key}' not found, even after case-insensitive lookup")
        
        def __contains__(self, key):
            if key in self.params:
                return True
            
            key_lower = key.lower()
            if key_lower in self.case_map:
                return True
                
            # Check for k_/K_ variants
            if key.startswith('k_') or key.startswith('K_'):
                alternate_key = 'K' + key[1:] if key.startswith('k') else 'k' + key[1:]
                if alternate_key in self.params:
                    return True
                
                alternate_key_lower = alternate_key.lower()
                if alternate_key_lower in self.case_map:
                    return True
            
            return False
        
        def keys(self):
            return self.params.keys()
        
        def get(self, key, default=None):
            try:
                return self[key]
            except KeyError:
                return default
    
    # Return parameters with case-insensitive access
    return CaseInsensitiveParams(params, case_insensitive_params) 