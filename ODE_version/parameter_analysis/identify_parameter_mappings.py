import pandas as pd
import os
import re
import sys

def identify_parameter_mappings(antibody_name='Gant'):
    """
    Identifies parameters that are used in the model under different names,
    focusing on antibody-specific parameters.
    
    Args:
        antibody_name: Name of the antibody to analyze ('Gant' or 'Lec')
    """
    # Load parameters from QSP_model_parameters.xlsx
    param_df = pd.read_excel('QSP_model_parameters.xlsx', sheet_name='QSP Model Parameter Table')
    param_names = set(param_df['Name'])
    
    # Define the mappings from parameter_loader.py
    if antibody_name == 'Gant':
        antibody_specific_mappings = {
            'fta0': 'Gant_fta0',
            'fta1': 'Gant_fta1',
            'fta2': 'Gant_fta2',
            'fta3': 'Gant_fta3',
            'PK_CL': 'CL_Gantenerumab',
            'PK_CLd2': 'CLd2_Gantenerumab',
            'PK_Vcent': 'Vcent_Gantenerumab',
            'PK_Vper': 'Vper_Gantenerumab',
            'PK_SC_ka': 'SC_ka_Gantenerumab',
            'PK_SC_bio': 'SC_bio_Gantenerumab'
        }
    else:  # 'Lec'
        antibody_specific_mappings = {
            'fta0': 'Lec_fta0',
            'fta1': 'Lec_fta1',
            'fta2': 'Lec_fta2',
            'fta3': 'Lec_fta3',
            'PK_CL': 'CL_BAN2401',
            'PK_CLd2': 'CLd2_BAN2401',
            'PK_Vcent': 'Vcent_BAN2401',
            'PK_Vper': 'Vper_BAN2401',
            'PK_SC_ka': 'Lec_SC_ka',
            'PK_SC_bio': 'Lec_SC_bio'
        }
    
    # Check which mappings have parameters in the QSP file
    found_mappings = {}
    alternative_mappings = {}
    missing_mappings = {}
    
    for model_param, source_param in antibody_specific_mappings.items():
        # First check if the exact parameter name exists
        if source_param in param_names:
            found_mappings[model_param] = source_param
        else:
            # Look for alternative parameters that might match
            # For example, if 'Gant_fta0' isn't found, look for 'taffa0_Gantenerumab'
            alt_param = None
            if source_param.startswith('Gant_fta'):
                suffix = source_param.split('_')[1]  # e.g., 'fta0'
                alt_param = f'taffa{suffix[-1]}_Gantenerumab'
            elif source_param.startswith('Lec_fta'):
                suffix = source_param.split('_')[1]  # e.g., 'fta0'
                alt_param = f'taffa{suffix[-1]}_BAN2401'
            
            if alt_param and alt_param in param_names:
                alternative_mappings[model_param] = (source_param, alt_param)
            else:
                missing_mappings[model_param] = source_param
    
    # Look for k_/K_ parameter mappings
    k_param_mappings = {}
    # Check case sensitivity mappings for parameters starting with k_/K_
    for param in param_names:
        if param.startswith('k_') or param.startswith('K_'):
            alt_case_param = 'K' + param[1:] if param.startswith('k') else 'k' + param[1:]
            if alt_case_param in param_names:
                k_param_mappings[param] = alt_case_param

    # Create output directory
    os.makedirs('parameter_analysis', exist_ok=True)
    
    # Write the mappings to a CSV file
    with open('parameter_analysis/parameter_mappings.csv', 'w') as f:
        f.write("model_parameter,source_parameter,in_qsp_file,value,unit,description\n")
        
        # Add the antibody-specific mappings
        for model_param, source_param in found_mappings.items():
            param_info = param_df[param_df['Name'] == source_param].iloc[0]
            value = param_info['Value'] if 'Value' in param_df else ''
            unit = param_info['Unit'] if 'Unit' in param_df else ''
            desc = param_info['Description'] if 'Description' in param_df else ''
            # Escape quotes in description
            if isinstance(desc, str):
                desc = desc.replace('"', '""')
            f.write(f"{model_param},{source_param},Yes,{value},\"{unit}\",\"{desc}\"\n")
        
        # Add alternative mappings
        for model_param, (orig_param, alt_param) in alternative_mappings.items():
            param_info = param_df[param_df['Name'] == alt_param].iloc[0]
            value = param_info['Value'] if 'Value' in param_df else ''
            unit = param_info['Unit'] if 'Unit' in param_df else ''
            desc = param_info['Description'] if 'Description' in param_df else ''
            # Escape quotes in description
            if isinstance(desc, str):
                desc = desc.replace('"', '""')
            f.write(f"{model_param},{orig_param} (alt: {alt_param}),Yes (alternative),{value},\"{unit}\",\"{desc}\"\n")
        
        # Add missing mappings
        for model_param, source_param in missing_mappings.items():
            f.write(f"{model_param},{source_param},No,,,\n")
        
        # Add k_/K_ mappings
        for param, alt_param in k_param_mappings.items():
            param_info = param_df[param_df['Name'] == param].iloc[0]
            value = param_info['Value'] if 'Value' in param_df else ''
            unit = param_info['Unit'] if 'Unit' in param_df else ''
            desc = param_info['Description'] if 'Description' in param_df else ''
            # Escape quotes in description
            if isinstance(desc, str):
                desc = desc.replace('"', '""')
            f.write(f"{param},{alt_param},Yes (case variant),{value},\"{unit}\",\"{desc}\"\n")
    
    # Print the results
    print(f"Parameter mapping analysis complete for antibody: {antibody_name}")
    print(f"Found direct mappings: {len(found_mappings)}")
    print(f"Found alternative mappings: {len(alternative_mappings)}")
    print(f"Missing mappings: {len(missing_mappings)}")
    print(f"Case-sensitive k_/K_ variants: {len(k_param_mappings)}")
    print(f"Results saved to parameter_analysis directory")
    
    return found_mappings, alternative_mappings, missing_mappings, k_param_mappings

def process_aggregation_parameters(aggregation_params, parameter_usage=None):
    """
    Process aggregation rate parameters and save them to a dedicated CSV file.
    
    Args:
        aggregation_params: List of parameter names, dict of {name: [equations]}, or list of dicts
        parameter_usage: Dictionary mapping parameters to equations they're used in
    
    Returns:
        Number of aggregation parameters processed
    """
    # Create output directory
    os.makedirs('parameter_analysis', exist_ok=True)
    
    # Convert all input formats to a standard dict format
    if isinstance(aggregation_params, list):
        # If it's a simple list of parameter names, convert to dict with empty equations
        params_dict = {param: [] for param in aggregation_params}
    elif isinstance(aggregation_params, dict):
        # If it's already a dict mapping params to equations, use it directly
        params_dict = aggregation_params
    else:
        # Unsupported format
        raise ValueError("Aggregation parameters must be a list of names or a dict mapping names to equation lists")
    
    # Add usage information if provided
    if parameter_usage:
        for param_name in params_dict:
            if param_name in parameter_usage and not params_dict[param_name]:
                params_dict[param_name] = parameter_usage[param_name]
    
    # Write to CSV file
    with open('parameter_analysis/aggregation_rate_parameters.csv', 'w') as f:
        f.write("name,type,description,equations_used_in\n")
        
        for param_name, equations in params_dict.items():
            # Determine the species (AB40 or AB42)
            species = "Unknown"
            if param_name.endswith('_forty'):
                species = "AB40"
            elif param_name.endswith('_fortytwo'):
                species = "AB42"
            
            # Use regex patterns for more reliable matching
            param_type = ""
            desc = ""
            
            # Pattern matching for different parameter types
            if re.match(r'k_O2_M_', param_name):
                param_type = f"{species} Oligomer"
                desc = f"Backward rate: {species} dimer to monomer"
            elif re.match(r'k_M_O2_', param_name):
                param_type = f"{species} Oligomer"
                desc = f"Forward rate: {species} monomer to dimer"
            elif re.match(r'k_O(\d+)_O(\d+)_', param_name):
                match = re.match(r'k_O(\d+)_O(\d+)_', param_name)
                if match:
                    from_size, to_size = match.groups()
                    from_num = int(from_size)
                    to_num = int(to_size)
                    param_type = f"{species} Oligomer"
                    if from_num < to_num:
                        desc = f"Forward rate: {species} oligomer size {from_size} to {to_size}"
                    else:
                        desc = f"Backward rate: {species} oligomer size {from_size} to {to_size}"
            elif re.match(r'k_F(\d+)_F(\d+)_', param_name):
                match = re.match(r'k_F(\d+)_F(\d+)_', param_name)
                if match:
                    from_size, to_size = match.groups()
                    from_num = int(from_size)
                    to_num = int(to_size)
                    param_type = f"{species} Fibril"
                    if from_num < to_num:
                        desc = f"Forward rate: {species} fibril size {from_size} to {to_size}"
                    else:
                        desc = f"Backward rate: {species} fibril size {from_size} to {to_size}"
            elif re.match(r'k_O(\d+)_F(\d+)_', param_name):
                match = re.match(r'k_O(\d+)_F(\d+)_', param_name)
                if match:
                    from_size, to_size = match.groups()
                    param_type = f"{species} Transition"
                    desc = f"Forward transition: {species} oligomer size {from_size} to fibril size {to_size}"
            elif re.match(r'k_F(\d+)_O(\d+)_', param_name):
                match = re.match(r'k_F(\d+)_O(\d+)_', param_name)
                if match:
                    from_size, to_size = match.groups()
                    param_type = f"{species} Transition"
                    desc = f"Backward transition: {species} fibril size {from_size} to oligomer size {to_size}"
            
            # If no description, at least provide the parameter type based on name format
            if not desc:
                if 'O' in param_name and not 'F' in param_name:
                    param_type = f"{species} Oligomer"
                elif 'F' in param_name and not 'O' in param_name:
                    param_type = f"{species} Fibril"
                elif 'O' in param_name and 'F' in param_name:
                    param_type = f"{species} Transition"
            
            # Convert equations list to string
            if isinstance(equations, list):
                eqs_used = ', '.join(equations)
            else:
                eqs_used = str(equations)
            
            # Escape quotes for CSV
            eqs_used = eqs_used.replace('"', '""')
            
            f.write(f"{param_name},\"{param_type}\",\"{desc}\",\"{eqs_used}\"\n")
    
    print(f"Processed {len(params_dict)} aggregation rate parameters")
    return len(params_dict)

if __name__ == "__main__":
    # Get antibody name from command line or use default
    antibody_name = sys.argv[1] if len(sys.argv) > 1 else "Gant"
    identify_parameter_mappings(antibody_name) 