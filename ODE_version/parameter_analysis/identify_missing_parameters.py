import pandas as pd
import re
import keyword
import sys
import os
from parameter_loader import load_parameters

def extract_variable_names(expression):
    """
    Find all variable names that are valid Python identifiers (exclude numbers)
    """
    tokens = re.findall(r'\b[a-zA-Z_]\w*\b', expression)
    return set(tokens)

def preprocess_equation(equation):
    """
    Replace '^' with '**' for exponentiation and 'exp' with 'jnp.exp'
    """
    equation = equation.replace('^', '**')
    equation = re.sub(r'\bexp\(', 'jnp.exp(', equation)
    return equation

def parse_equation(equation_str):
    """
    Convert the equation string from the form 'd(variable) / dt = expression'
    to extract the variable and expression
    """
    match = re.match(r"d\((.*?)\)\s*/\s*dt\s*=\s*(.*)", equation_str)
    if match:
        variable = match.group(1)
        expression = match.group(2)
        expression = preprocess_equation(expression)
        return variable, expression
    else:
        return None, None

def identify_parameters(antibody_name="Gant"):
    """
    Analyze ODEs and identify parameters in the equations and their source
    """
    # Load ODEs from Excel file
    file_equations = './Geerts_ODEs.xlsx'
    equations_sheets = pd.read_excel(file_equations, sheet_name=None)
    equations_df = equations_sheets['ODEs'][['Name', 'ODE']]
    
    # Load parameters
    params = load_parameters(antibody_name)
    
    # Define parameter replacements
    parameter_replacements = {
        'K_F17_Plaque_forty': 'Baseline_AB40_Oligomer_Fibril_Plaque',
        'K_F17_Plaque_fortytwo': 'Baseline_AB42_Oligomer_Fibril_Plaque',
        'K_F18_Plaque_forty': 'Baseline_AB40_Oligomer_Fibril_Plaque',
        'K_F18_Plaque_fortytwo': 'Baseline_AB42_Oligomer_Fibril_Plaque',
        'K_O13_Plaque_forty': 'Baseline_AB40_Oligomer_Fibril_Plaque',
        'K_O13_Plaque_fortytwo': 'Baseline_AB42_Oligomer_Fibril_Plaque',
        'K_O14_Plaque_forty': 'Baseline_AB40_Oligomer_Fibril_Plaque',
        'K_O14_Plaque_fortytwo': 'Baseline_AB42_Oligomer_Fibril_Plaque',
        'K_O15_Plaque_forty': 'Baseline_AB40_Oligomer_Fibril_Plaque',
        'K_O15_Plaque_fortytwo': 'Baseline_AB42_Oligomer_Fibril_Plaque',
        'K_O16_Plaque_forty': 'Baseline_AB40_Oligomer_Fibril_Plaque',
        'K_O16_Plaque_fortytwo': 'Baseline_AB42_Oligomer_Fibril_Plaque'
    }
    
    # Define lower order rate parameters that should remain in missing_parameters
    lower_order_rate_params = [
        'k_O2_M_forty', 'k_O2_M_fortytwo',
        'k_M_O2_forty', 'k_M_O2_fortytwo',
        'k_O2_O3_forty', 'k_O2_O3_fortytwo',
        'k_O3_O2_forty', 'k_O3_O2_fortytwo',
        'k_F24_O12_forty', 'k_F24_O12_fortytwo'
    ]
    
    # Parse all equations
    equations_dict = {}
    for _, row in equations_df.iterrows():
        variable, expression = parse_equation(row['ODE'])
        if variable and expression:
            equations_dict[variable] = expression
    
    # Create a mapping of state variable names to their indices
    state_variable_mapping = {var: idx for idx, var in enumerate(equations_dict.keys())}
    
    # Set of built-in function names and keywords to exclude
    builtin_names = set(keyword.kwlist + dir(__builtins__))
    exclude_names = builtin_names.union({'jnp', 'exp', 'params', 't', 'y'})
    
    # Collect all variable names used in equations
    all_variables = set()
    parameter_usage = {}  # Track where each parameter is used
    
    for var, equation in equations_dict.items():
        # Extract all variable names in this equation
        variable_names = extract_variable_names(equation)
        all_variables.update(variable_names)
        
        # Record where each parameter is used
        for param in variable_names:
            if param not in state_variable_mapping and param not in exclude_names:
                if param not in parameter_usage:
                    parameter_usage[param] = []
                parameter_usage[param].append(var)
    
    # Identify parameters (variables that are not state variables or built-ins)
    parameters = all_variables - set(state_variable_mapping.keys()) - exclude_names
    
    # Split parameters into those that exist in the params dictionary and those that don't
    existing_params = set()
    missing_params = set()
    aggregation_params = set()
    
    for param in parameters:
        try:
            # Try to access with case-insensitive matching
            params[param]
            existing_params.add(param)
        except KeyError:
            # Categorize missing parameters
            if ((param.startswith('k_O') or param.startswith('k_F')) and
                ('_forty' in param or '_fortytwo' in param) and
                param not in lower_order_rate_params):
                aggregation_params.add(param)
            else:
                missing_params.add(param)
    
    # Create output directory if it doesn't exist
    os.makedirs('parameter_analysis', exist_ok=True)
    
    # Write missing parameters to CSV with the replacement column
    with open('parameter_analysis/missing_parameters.csv', 'w') as f:
        f.write("name,equations_used_in,replacement\n")
        for param in sorted(missing_params):
            equations_used_in = ', '.join(parameter_usage.get(param, []))
            replacement = parameter_replacements.get(param, '')
            f.write(f"{param},\"{equations_used_in}\",\"{replacement}\"\n")
    
    # Write analysis to file for parameters to add to the model
    with open('parameter_analysis/parameters_to_add.csv', 'w') as f:
        f.write("name,value,units,Sup_Name,Source,Validated,Notes,replacement\n")
        for param in sorted(missing_params):
            equations_used_in = ', '.join(parameter_usage.get(param, []))
            replacement = parameter_replacements.get(param, '')
            f.write(f"{param},,,,,0,\"Used in equations for: {equations_used_in}\",\"{replacement}\"\n")
    
    # Write a comprehensive report
    with open('parameter_analysis/parameter_analysis_report.txt', 'w') as f:
        f.write("Parameter Analysis Report\n")
        f.write("=======================\n\n")
        
        f.write(f"Total unique parameters found in equations: {len(parameters)}\n")
        f.write(f"Parameters found in parameter file: {len(existing_params)}\n")
        f.write(f"Aggregation rate parameters: {len(aggregation_params)}\n")
        
        if missing_params:
            f.write(f"Missing parameters (excluding aggregation rates): {len(missing_params)}\n\n")
            
            f.write("Missing parameters with replacements:\n")
            for param in sorted(missing_params):
                if param in parameter_replacements:
                    equations_used_in = ', '.join(parameter_usage.get(param, []))
                    replacement = parameter_replacements.get(param, '')
                    f.write(f"  {param} - Used in: {equations_used_in} - Replaced by: {replacement}\n")
            
            f.write("\nMissing parameters without replacements:\n")
            for param in sorted(missing_params):
                if param not in parameter_replacements:
                    equations_used_in = ', '.join(parameter_usage.get(param, []))
                    f.write(f"  {param} - Used in: {equations_used_in}\n")
        else:
            f.write("All parameters (excluding aggregation rates) used in equations are present in the parameter file.\n")
        
        f.write("\nLower order rate parameters (kept in missing parameters):\n")
        for param in sorted(lower_order_rate_params):
            if param in parameter_usage:
                equations_used_in = ', '.join(parameter_usage.get(param, []))
                f.write(f"  {param} - Used in: {equations_used_in}\n")
    
    # Print summary
    print(f"Analysis complete. Found {len(missing_params)} missing parameters (excluding {len(aggregation_params)} aggregation rate parameters).")
    print(f"Missing parameters saved to parameter_analysis/missing_parameters.csv")
    print(f"Parameters to add saved to parameter_analysis/parameters_to_add.csv")
    print(f"Full analysis report saved to parameter_analysis/parameter_analysis_report.txt")
    
    return parameters, existing_params, missing_params, aggregation_params, parameter_usage

if __name__ == "__main__":
    # Get antibody name from command line argument or use default
    antibody_name = sys.argv[1] if len(sys.argv) > 1 else "Gant"
    identify_parameters(antibody_name) 