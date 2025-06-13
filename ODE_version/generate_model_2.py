import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import re           
import keyword
import jax.numpy as jnp
import sys
import os

# Add parent directory to path to import K_rates_extrapolate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameter_loader import load_parameters

# File paths
file_equations = './Geerts_ODEs.xlsx'

# Accept antibody name as input
antibody_name = input("Enter the antibody name (Gant or Lec): ")

# Load all parameters
params = load_parameters(antibody_name)

# Step 1: Extract and parse equations from the 'ODEs' sheet
equations_sheets = pd.read_excel(file_equations, sheet_name=None)
equations_df = equations_sheets['ODEs'][['Name', 'ODE']]

# Helper function to preprocess and parse the equations
def preprocess_equation(equation):
    # Replace '^' with '**' for exponentiation
    equation = equation.replace('^', '**')
    # Replace 'exp' with 'jnp.exp' for exponential function
    equation = re.sub(r'\bexp\(', 'jnp.exp(', equation)
    return equation

def parse_equation(equation_str):
    """
    Convert the equation string from the form 'd(variable) / dt = expression'
    to a Python expression: 'dy[index] = expression'.
    """
    match = re.match(r"d\((.*?)\)\s*/\s*dt\s*=\s*(.*)", equation_str)
    if match:
        variable = match.group(1)
        expression = match.group(2)
        # Preprocess the expression
        expression = preprocess_equation(expression)
        return variable, expression
    else:
        return None, None

# Parse all equations and convert them into a dictionary
equations_dict = {}
for _, row in equations_df.iterrows():
    variable, expression = parse_equation(row['ODE'])
    if variable and expression:
        equations_dict[variable] = expression

# Create a mapping of state variable names to indices
state_variable_mapping = {var: idx for idx, var in enumerate(equations_dict.keys())}

# Set of built-in function names and keywords to exclude
builtin_names = set(keyword.kwlist + dir(__builtins__))

# Helper function to extract variable names from an expression
def extract_variable_names(expression):
    # Find all variable names that are valid Python identifiers (exclude numbers)
    tokens = re.findall(r'\b[a-zA-Z_]\w*\b', expression)
    return set(tokens)

# Step 3: Translate equations using the state variable names instead of indices and replace parameter names
translated_equations = {}
all_variables_in_equations = set()
substituted_params = set()  # Keep track of parameters that were successfully substituted
warnings = []  # Track warnings for later output

def get_variables_in_equations():
    """
    Returns variables (parameters) used in equations.
    """
    # Exclude state variable names and built-in names
    variables_in_equations = all_variables_in_equations - set(state_variable_mapping.keys()) - builtin_names - {'jnp', 'exp', 'params', 't', 'y', 'state'}
    return sorted(variables_in_equations)

for var, equation in equations_dict.items():
    var_idx = state_variable_mapping[var]
    original_equation = equation  # Keep a copy of the original equation

    # Replace state variable names with state['var_name']
    for state_var, idx in state_variable_mapping.items():
        equation = re.sub(rf'\b{state_var}\b', f"state['{state_var}']", equation)

    # Extract variable names used in the equation
    variable_names = extract_variable_names(equation)

    # Collect all variables used in equations
    all_variables_in_equations.update(variable_names)

    # Replace parameter names with params['param_name']
    for param_name in variable_names:
        if param_name in params:
            equation = re.sub(rf'\b{param_name}\b', f"params['{param_name}']", equation)
            substituted_params.add(param_name)
        elif param_name not in state_variable_mapping and param_name not in builtin_names and param_name not in {'jnp', 'exp', 'params', 't', 'y', 'state'}:
            # Try different case variations for k_ parameters
            if param_name.lower().startswith('k_'):
                alternate_param = 'K' + param_name[1:] if param_name.startswith('k') else 'k' + param_name[1:]
                if alternate_param in params:
                    equation = re.sub(rf'\b{param_name}\b', f"params['{alternate_param}']", equation)
                    substituted_params.add(alternate_param)
                    continue
            
            warnings.append(f"Warning: Parameter '{param_name}' not found in params (used in equation for {var}).")

    # Generate translated equation
    translated_equations[var] = {
        'equation': f"new_state['{var}'] = {equation}",
        'variable_name': var
    }

# Get all variables used in equations
variables_in_equations = get_variables_in_equations()
missing_params = []

# Check which variables in equations are not in params
for variable in variables_in_equations:
    try:
        # Try to access the parameter to see if it exists (including case-insensitive variations)
        params[variable]
    except KeyError:
        missing_params.append(variable)

# Step 4: Generate the Python function as a string
function_code = '''
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Dict, Any

def qsp_ode_model(t: float, state: Dict[str, float], params: Dict[str, Any]) -> Dict[str, float]:
    """
    Differential equations for the QSP model.
    Args:
        state: state variable dictionary mapping variable names to their values
        t: time (float)
        params: dictionary of known parameter values
    Returns:
        new_state: dictionary with derivatives of all state variables
    """
    new_state = {}
'''

# Add the translated equations to the function with comments
for var_name in sorted(translated_equations.keys(), key=lambda k: state_variable_mapping[k]):
    equation = translated_equations[var_name]['equation']
    function_code += f"    # Equation for {var_name}\n"
    function_code += f"    {equation}\n"

# Add the return statement
function_code += '''
    return new_state
'''

# Also generate a helper function to convert between array and dict formats
helper_code = '''

def state_array_to_dict(y, state_names):
    """
    Convert state array to state dictionary.
    
    Args:
        y: state variable array
        state_names: list of state variable names in the same order as y
        
    Returns:
        state_dict: dictionary mapping state variable names to their values
    """
    return {name: jnp.asarray(val) for name, val in zip(state_names, y)}

def state_dict_to_array(state_dict, state_names):
    """
    Convert state dictionary to state array.
    
    Args:
        state_dict: dictionary mapping state variable names to their values
        state_names: list of state variable names to determine the order
        
    Returns:
        y: state variable array in the specified order
    """
    return jnp.array([state_dict[name] for name in state_names])

def get_state_names():
    """
    Return the list of state variable names in the order used in the original model.
    """
    return ''' + str(list(sorted(state_variable_mapping.keys(), key=lambda k: state_variable_mapping[k]))) + '''

def array_ode_wrapper(t, y, params):
    """
    Wrapper for the ODE model that works with array-based solvers.
    
    Args:
        t: time
        y: state variable array
        params: parameter dictionary
        
    Returns:
        dy: derivatives of state variables as array
    """
    state_names = get_state_names()
    state_dict = state_array_to_dict(y, state_names)
    dydt_dict = qsp_ode_model(t, state_dict, params)
    return state_dict_to_array(dydt_dict, state_names)
'''

function_code += helper_code

# Step 5: Write the function to a Python script
output_script = './generated_model_2.py'
with open(output_script, "w") as f:
    f.write(function_code)

# Create a report file with detailed information about the model generation
report_file = './model_generation_report.txt'
with open(report_file, 'w') as f:
    f.write("Model Generation Report\n")
    f.write("======================\n\n")
    
    # Information about state variables
    f.write(f"State Variables: {len(state_variable_mapping)}\n")
    for var, idx in state_variable_mapping.items():
        f.write(f"  {idx}: {var}\n")
    f.write("\n")
    
    # Information about parameters
    f.write(f"Parameters Used: {len(substituted_params)}\n")
    for param in sorted(substituted_params):
        try:
            param_value = params[param]
            f.write(f"  {param}: {param_value}\n")
        except:
            f.write(f"  {param}: [Error accessing value]\n")
    f.write("\n")
    
    # List any missing parameters if found during this run
    if missing_params:
        f.write(f"Parameters Not Found in Geerts_Params_2.csv: {len(missing_params)}\n")
        for param in sorted(missing_params):
            f.write(f"  {param}\n")
        f.write("\n")
    
    # List any warnings
    if warnings:
        f.write(f"Warnings: {len(warnings)}\n")
        for warning in warnings:
            f.write(f"  {warning}\n")
        f.write("\n")

# Display the location of the generated script
print(f"Generated script saved to: {output_script}")
print(f"Generation report saved to: {report_file}")

# Display results summary
if missing_params:
    print(f"\nFound {len(missing_params)} parameters in equations that are not in Geerts_Params_2.csv")
    print("You can run identify_missing_parameters.py to generate a template for missing parameters.")
else:
    print("\nAll parameters used in equations are present in Geerts_Params_2.csv.")


