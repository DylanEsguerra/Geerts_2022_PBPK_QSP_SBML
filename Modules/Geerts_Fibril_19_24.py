"""
Module for modeling large Aβ fibril and plaque formation and dynamics (19-24mers).
This is a QSP component that models the formation and behavior of large Aβ fibrils.
The module includes:
- Fibril formation from smaller fibrils and monomers
- Fibril dissociation to smaller species
- Plaque formation from fibrils
- Antibody binding to fibrils
- Microglia-mediated clearance of fibrils
- Transport and reflection coefficients for fibrils
"""

# updated to include fibril 24 to 12 reaction
# deleted formation of next fibril from current fibril and monomer
# deleted formation of next fibril from current fibril and monomer plaque catalyzed
import libsbml
from pathlib import Path
import pandas as pd
from K_rates_extrapolate import calculate_k_rates
import numpy as np

# Add sink and source functions after the import statements
def create_sink_for_species(species_id, model):
    """Create a sink species for a given species ID"""
    sink_id = f"Sink_{species_id}"
    
    # First check if the sink species already exists
    existing_sink = model.getSpecies(sink_id)
    if existing_sink is not None:
        return sink_id
    
    # Create sink boundary species
    sink = model.createSpecies()
    sink.setId(sink_id)
    sink.setName(sink_id)
    sink.setCompartment("comp_ISF_brain")
    sink.setInitialConcentration(0.0)
    sink.setConstant(True)
    sink.setHasOnlySubstanceUnits(False)
    sink.setBoundaryCondition(True)  # Boundary species
    sink.setSubstanceUnits("nanomole_per_litre")
    return sink_id

def create_source_for_species(species_id, model):
    """Create a source species for a given species ID"""
    source_id = f"Source_{species_id}"
    
    # First check if the source species already exists
    existing_source = model.getSpecies(source_id)
    if existing_source is not None:
        return source_id
    
    # Create source boundary species
    source = model.createSpecies()
    source.setId(source_id)
    source.setName(source_id)
    source.setCompartment("comp_ISF_brain")
    source.setInitialConcentration(0.0)
    source.setConstant(True)
    source.setHasOnlySubstanceUnits(False)
    source.setBoundaryCondition(True)  # Boundary species
    source.setSubstanceUnits("nanomole_per_litre")
    return source_id

# Helper function to get unique reaction ID
def get_unique_reaction_id(base_id, created_reaction_ids):
    """Create a unique reaction ID by incrementing counter if base ID already exists
    
    Args:
        base_id: Base ID for the reaction
        created_reaction_ids: Dictionary tracking used reaction IDs
    
    Returns:
        Unique reaction ID string
    """
    if base_id not in created_reaction_ids:
        created_reaction_ids[base_id] = 1
        return base_id
    else:
        created_reaction_ids[base_id] += 1
        return f"{base_id}_{created_reaction_ids[base_id]}"

# Helper function to check if a species exists
def check_species_exists(model, species_id):
    """Check if a species already exists in the model
    
    Args:
        model: The SBML model
        species_id: ID of the species to check
        
    Returns:
        True if species exists, False otherwise
    """
    return model.getSpecies(species_id) is not None

# Helper function to ensure sigma parameter exists
def ensure_sigma_parameter(model, params, j, ab_type):
    """Ensure that sigma parameter for fibril j exists"""
    sigma_param = f"sigma_ISF_ABeta{ab_type}_oligomer{j:02d}"
    
    # Create parameter if it doesn't exist
    if model.getParameter(sigma_param) is None:
        param = model.createParameter()
        param.setId(sigma_param)
        param.setValue(params[sigma_param])
        param.setConstant(True)
        param.setUnits("dimensionless")
    
    return sigma_param

# Helper function to ensure rate parameter exists
def ensure_rate_parameter(model, params, param_id):
    """Ensure that a rate parameter exists"""
    # Create parameter if it doesn't exist
    if model.getParameter(param_id) is None:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(params[param_id])
        param.setConstant(True)
        
        # Determine units based on parameter type
        if param_id.startswith('k_F') and "_F" in param_id.split("k_")[1]:
            param.setUnits("per_nanomole_per_hour")  # Second-order rate constant
        else:
            param.setUnits("per_hour")  # First-order rate constant
    
    return param_id

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} in Fibril 19-24 module ===")
    print(f"Loading parameters from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print("\nFirst few rows of parameter file:")
    print(df.head())
    
    # Create parameters dictionary with mapped values
    params = dict(zip(df['name'], df['value']))
    params_with_units = dict(zip(df['name'], zip(df['value'], df['units'])))
    
    # Handle drug-specific parameter mapping
    is_lecanemab = drug_type.lower() == "lecanemab"
    
    # Map antibody-specific parameters
    param_mapping = {
        'fta0': 'Lec_fta0' if is_lecanemab else 'Gant_fta0',
        'fta1': 'Lec_fta1' if is_lecanemab else 'Gant_fta1',
        'fta2': 'Lec_fta2' if is_lecanemab else 'Gant_fta2',
        'fta3': 'Lec_fta3' if is_lecanemab else 'Gant_fta3',
    }
    
    # Apply parameter mapping
    for generic_name, specific_name in param_mapping.items():
        if specific_name in params:
            params[generic_name] = params[specific_name]
            if specific_name in params_with_units:
                params_with_units[generic_name] = params_with_units[specific_name]
        else:
            print(f"WARNING: {specific_name} not found in parameters! {generic_name} will be unavailable.")
    
    # Print verification of key parameters
    print("\nAntibody binding parameters for Fibril 19-24:")
    for param in ['fta0', 'fta1', 'fta2', 'fta3']:
        if param in params:
            print(f"  {param}: {params[param]}")
        else:
            print(f"  {param}: NOT FOUND")
    
    # Calculate extrapolated rates using the new parameterized version
    extrapolated_rates = calculate_k_rates(
        original_kf0_forty=params['original_kf0_forty'],
        original_kf0_fortytwo=params['original_kf0_fortytwo'],
        original_kf1_forty=params['original_kf1_forty'],
        original_kf1_fortytwo=params['original_kf1_fortytwo'],
        original_kb0_forty=params['original_kb0_forty'],
        original_kb0_fortytwo=params['original_kb0_fortytwo'],
        original_kb1_forty=params['original_kb1_forty'],
        original_kb1_fortytwo=params['original_kb1_fortytwo'],
        forAsymp40=params['forAsymp40'],
        forAsymp42=params['forAsymp42'],
        backAsymp40=params['backAsymp40'],
        backAsymp42=params['backAsymp42'],
        forHill40=params['forHill40'],
        forHill42=params['forHill42'],
        BackHill40=params['BackHill40'],
        BackHill42=params['BackHill42'],
        rate_cutoff=params['rate_cutoff']
    )
    
    # Add extrapolated rates to parameters dictionary
    params.update(extrapolated_rates)
    
    # Add units for extrapolated rates
    for rate_name in extrapolated_rates:
        params_with_units[rate_name] = (extrapolated_rates[rate_name], '1/h')
    
    print("\nExtrapolated rate constants added to parameters")
    
    return params, params_with_units

def create_fibril_19_24_model(params, params_with_units):
    """Create a parameterized SBML model for the Geerts Fibril 19-24 model
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
        
    Returns:
        SBML document
    """
    print("\nCreating Geerts Fibril 19-24 model...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_Fibril_19_24_Model")
    
    # Add a global dictionary to track used reaction IDs across all AB types
    created_reaction_ids = {}
    
    # Create dictionaries to hold sink and source species
    sinks = {}
    sources = {}
    
    # Add helper functions to global variables used in exec
    global_vars = globals().copy()
    global_vars.update({
        'model': model, 
        'libsbml': libsbml, 
        'get_unique_reaction_id': lambda base_id: get_unique_reaction_id(base_id, created_reaction_ids),
        'created_reaction_ids': created_reaction_ids,
        'check_species_exists': lambda species_id: check_species_exists(model, species_id),
        'ensure_sigma_parameter': lambda j, ab_type: ensure_sigma_parameter(model, params, j, ab_type),
        'ensure_rate_parameter': lambda param_id: ensure_rate_parameter(model, params, param_id),
        'create_sink_for_species': lambda species_id: create_sink_for_species(species_id, model),
        'create_source_for_species': lambda species_id: create_source_for_species(species_id, model),
        'sinks': sinks,
        'sources': sources
    })
    
    # Define units
    hour = model.createUnitDefinition()
    hour.setId('hour')
    hour_unit = hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setMultiplier(3600.0)
    hour_unit.setScale(0)
    hour_unit.setExponent(1.0)
    
    # Flow rate unit (litre/hour)
    flow = model.createUnitDefinition()
    flow.setId("litre_per_hour")
    litre_unit = flow.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)
    hour_unit = flow.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # Binding rate constants (1/hour)
    rate = model.createUnitDefinition()
    rate.setId("per_hour")
    hour_unit = rate.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # Per mole per hour unit
    per_nanomole_per_hour = model.createUnitDefinition()
    per_nanomole_per_hour.setId("per_nanomole_per_hour")
    
    # Mole unit (per nanomole)
    mole_unit = per_nanomole_per_hour.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(-1)
    mole_unit.setScale(-9)  # nano
    mole_unit.setMultiplier(1.0)
    
    hour_unit = per_nanomole_per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    concentration = model.createUnitDefinition()
    concentration.setId("nanomole_per_litre")
    mole_unit = concentration.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(1.0)
    mole_unit.setScale(-9)  # nano
    mole_unit.setMultiplier(1.0)
    litre_unit = concentration.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(-1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)

    # Add per_mole_per_hour unit definition
    per_mole_per_hour = model.createUnitDefinition()
    per_mole_per_hour.setId("per_mole_per_hour")
    mole_unit = per_mole_per_hour.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(-1)
    mole_unit.setScale(0)
    mole_unit.setMultiplier(1.0)
    hour_unit = per_mole_per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # NOW set model time units
    model.setTimeUnits("hour")

    # Create common compartments
    compartments = {
        "comp_ISF_brain": params["VIS_brain"],
        "comp_microglia": params["V_microglia"],
    }

    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)
        comp.setUnits('litre')

    # Define parameters needed for fibril 19-24 reactions
    fibril_params = []
    
    # Add reflection coefficients for fibrils 19-24
    for j in range(19, 25):
        fibril_params.extend([
            (f"sigma_ISF_ABeta40_oligomer{j:02d}", params[f"sigma_ISF_ABeta40_oligomer{j:02d}"]),
            (f"sigma_ISF_ABeta42_oligomer{j:02d}", params[f"sigma_ISF_ABeta42_oligomer{j:02d}"]),
        ])


    
    # Add fibril formation parameters for 19-24
    for j in range(19, 25):
        if j < 24:  # For fibrils 19-23
            fibril_params.extend([
                (f"k_F{j-1}_F{j}_forty", params[f"k_F{j-1}_F{j}_forty"]),
                (f"k_F{j-1}_F{j}_fortytwo", params[f"k_F{j-1}_F{j}_fortytwo"]),
                (f"k_F{j}_F{j-1}_forty", params[f"k_F{j}_F{j-1}_forty"]),
                (f"k_F{j}_F{j-1}_fortytwo", params[f"k_F{j}_F{j-1}_fortytwo"]),
            ])
    
    # Add special parameters for Fibril24
    fibril_params.extend([
        # Breakdown to Oligomer12
        ("k_F24_O12_forty", params["k_F24_O12_forty"]),
        ("k_F24_O12_fortytwo", params["k_F24_O12_fortytwo"]),
    ])
    
    # Add other non-fibrillization parameters
    fibril_params.extend([
        # Antibody binding parameters
        ("fta2", params["fta2"]),  # Antibody binding to larger fibrils
        ("fta3", params["fta3"]),  # Antibody binding to plaque
        
        # Microglia parameters
        ("Microglia_CL_high_AB40", params["Microglia_CL_high_AB40"]),
        ("Microglia_CL_low_AB40", params["Microglia_CL_low_AB40"]),
        ("Microglia_CL_high_AB42", params["Microglia_CL_high_AB42"]),
        ("Microglia_CL_low_AB42", params["Microglia_CL_low_AB42"]),
        ("Microglia_Vmax_forty", params["Microglia_Vmax_forty"]),
        ("Microglia_Vmax_fortytwo", params["Microglia_Vmax_fortytwo"]),
        # Other parameters needed for reactions
        ("VIS_brain", params["VIS_brain"]),  # ISF volume
    ])
    
    # Add the parameters to the model
    print("\nCreating fibril 19-24 parameters...")
    for param_id, value in fibril_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Assign appropriate units based on parameter type
        if param_id.startswith('k_F'):
            if "_F" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")
            else:
                param.setUnits("per_hour")
        elif param_id.startswith('sigma_'):
            param.setUnits("dimensionless")  # Reflection coefficient
        elif param_id.startswith('Q_'):
            param.setUnits("litre_per_hour")  # Flow rate
        elif param_id.startswith('vol_'):
            param.setUnits("litre")  # Volume
        else:
            param.setUnits("dimensionless")
    
    # Define fibril species
    fibril_species = []
    
    # Add fibril species for AB40 and AB42 (18-24)
    # Note: Including Fibril18 since it's referenced in Fibril19 reactions
    for i in range(18, 25):
        fibril_species.extend([
            (f"AB40_Fibril{i:02d}", "comp_ISF_brain", 0.0),
            (f"AB42_Fibril{i:02d}", "comp_ISF_brain", 0.0),
            (f"AB40_Fibril{i:02d}_Antibody_bound", "comp_ISF_brain", 0.0),
            (f"AB42_Fibril{i:02d}_Antibody_bound", "comp_ISF_brain", 0.0),
        ])
    
    # Also include necessary other species for reaction connectivity
    fibril_species.extend([
        # Monomers
        ("AB40_Monomer", "comp_ISF_brain", 0.0),
        ("AB42_Monomer", "comp_ISF_brain", 0.0),
        
        # Oligomer12 (for Fibril24 breakdown)
        ("AB40_Oligomer12", "comp_ISF_brain", 0.0),
        ("AB42_Oligomer12", "comp_ISF_brain", 0.0),
        
        # Antibody and plaque species
        ("Ab_t", "comp_ISF_brain", 0.0),
        ("AB40_Plaque_unbound", "comp_ISF_brain", 0.0),
        ("AB42_Plaque_unbound", "comp_ISF_brain", 0.0),
        ("AB40_Plaque_Antibody_bound", "comp_ISF_brain", 0.0),
        ("AB42_Plaque_Antibody_bound", "comp_ISF_brain", 0.0),
    ])
    
    # Add microglia species
    fibril_species.extend([
        ("Microglia_Hi_Fract", "comp_microglia", params["Microglia_Hi_Fract_0"]),
        ("Microglia_cell_count", "comp_microglia", params["Microglia_cell_count_0"]),
    ])
    
    # Create all species
    print("\nCreating fibril 19-24 species with initial concentrations from parameters...")
    for species_id, compartment_id, initial_value in fibril_species:
        # Skip if species already exists
        if check_species_exists(model, species_id):
            continue
        
        spec = model.createSpecies()
        spec.setId(species_id)
        spec.setCompartment(compartment_id)
        
        # Check for parameter in CSV first with proper naming
        param_name = f"{species_id}_0"
        if param_name in params:
            initial_conc = params[param_name]
            print(f"  Setting {species_id} initial concentration from CSV: {initial_conc}")
        else:
            initial_conc = initial_value
            print(f"  Parameter {param_name} not found, using default: {initial_value}")
        
        spec.setInitialConcentration(initial_conc)
        spec.setSubstanceUnits("nanomole_per_litre")
        spec.setHasOnlySubstanceUnits(False)
        spec.setBoundaryCondition(False)
        spec.setConstant(False)
    
    # Placeholder for reaction creation functions
    # These will be implemented next based on the equations
    
    # Create reactions for AB40 and AB42
    print("\nCreating reactions for AB40 and AB42...")
    create_fibril_19_24_reactions(model, params, "40")
    create_fibril_19_24_reactions(model, params, "42")
    
    # Log the model creation status
    print(f"\nFibril 19-24 model structure created with:")
    print(f"  - {model.getNumCompartments()} compartments")
    print(f"  - {model.getNumSpecies()} species")
    print(f"  - {model.getNumParameters()} parameters")
    print(f"  - {model.getNumReactions()} reactions")
    
    return document

def create_fibril_19_23_reactions(model, params, ab_type, suffix):
    """Create all reactions for fibrils 19-23 of a specific AB type (40 or 42)
    
    Args:
        model: The SBML model
        params: Dictionary of parameter values
        ab_type: AB type (40 or 42)
        suffix: Parameter suffix (forty or fortytwo)
    """
    print(f"\nCreating fibril 19-23 reactions for AB{ab_type}...")
    
    # Create a dictionary to track reaction IDs
    created_reaction_ids = {}
    
    # Dictionary to hold sinks for species
    sinks = {}
    
    # Helper function to get unique reaction IDs
    def get_unique_reaction_id(base_id):
        if base_id not in created_reaction_ids:
            created_reaction_ids[base_id] = 0
            return base_id
        else:
            created_reaction_ids[base_id] += 1
            return f"{base_id}_{created_reaction_ids[base_id]}"
    
    # Initialize global variables dictionary with all necessary variables
    global_vars = {
        'model': model,
        'libsbml': libsbml,
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids,
        'ensure_sigma_parameter': lambda j, ab_type: ensure_sigma_parameter(model, params, j, ab_type),
        'ensure_rate_parameter': lambda param_id: ensure_rate_parameter(model, params, param_id),
        'params': params,
        'ab_type': ab_type,
        'suffix': suffix,
        'create_sink_for_species': lambda species_id: create_sink_for_species(species_id, model),
        'sinks': sinks
    }
    
    # We'll create chain reactions between fibrils 19-23
    for j in range(19, 24):
        # Determine current fibril and previous/next fibrils
        current_fibril = f"AB{ab_type}_Fibril{j:02d}"
        prev_fibril = f"AB{ab_type}_Fibril{j-1:02d}"
        next_fibril = f"AB{ab_type}_Fibril{j+1:02d}" 
        monomer = f"AB{ab_type}_Monomer"
        
        # List to hold all reaction blocks for this fibril
        fibril_reaction_blocks = []
        
        # Ensure sigma parameter exists
        sigma_param = ensure_sigma_parameter(model, params, j, ab_type)
        
        # Ensure rate parameters exist
        k_prev_current = ensure_rate_parameter(model, params, f"k_F{j-1}_F{j}_{suffix}")  # Previous fibril to current fibril
        k_current_prev = ensure_rate_parameter(model, params, f"k_F{j}_F{j-1}_{suffix}")  # Current fibril to previous fibril
        
       
        k_current_next = ensure_rate_parameter(model, params, f"k_F{j}_F{j+1}_{suffix}")  # Current fibril to next fibril
        k_next_current = ensure_rate_parameter(model, params, f"k_F{j+1}_F{j}_{suffix}")  # Next fibril to current fibril
        
        # Reaction 1: Formation from previous fibril + monomer
        formation_reaction = f'''
# Formation of {current_fibril} from {prev_fibril} and monomer
reaction_fibril{j}_formation = model.createReaction()
reaction_fibril{j}_formation.setId(get_unique_reaction_id("AB{ab_type}_fibril{j}_formation"))
reaction_fibril{j}_formation.setReversible(False)

# Reactants: {prev_fibril} and monomer
reactant_prev_fibril = reaction_fibril{j}_formation.createReactant()
reactant_prev_fibril.setSpecies("{prev_fibril}")
reactant_prev_fibril.setStoichiometry(1.0)
reactant_prev_fibril.setConstant(True)

reactant_monomer = reaction_fibril{j}_formation.createReactant()
reactant_monomer.setSpecies("{monomer}")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Product: Current fibril
product_fibril = reaction_fibril{j}_formation.createProduct()
product_fibril.setSpecies("{current_fibril}")
product_fibril.setStoichiometry(1.0)
product_fibril.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_formation = reaction_fibril{j}_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"{k_prev_current} * {prev_fibril} * {monomer} * VIS_brain")
klaw_formation.setMath(math_ast)
'''
        fibril_reaction_blocks.append(formation_reaction)
        
        # Reaction 2: Dissociation to previous fibril
        dissociation_reaction = f'''
# Dissociation of {current_fibril} to {prev_fibril}
reaction_fibril{j}_dissociation = model.createReaction()
reaction_fibril{j}_dissociation.setId(get_unique_reaction_id("AB{ab_type}_fibril{j}_dissociation"))
reaction_fibril{j}_dissociation.setReversible(False)

# Reactant: Current fibril
reactant_fibril = reaction_fibril{j}_dissociation.createReactant()
reactant_fibril.setSpecies("{current_fibril}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

# Product: Previous fibril
product_prev_fibril = reaction_fibril{j}_dissociation.createProduct()
product_prev_fibril.setSpecies("{prev_fibril}")
product_prev_fibril.setStoichiometry(1.0)
product_prev_fibril.setConstant(True)

# Product: monomer
product_monomer = reaction_fibril{j}_dissociation.createProduct()
product_monomer.setSpecies("{monomer}")
product_monomer.setStoichiometry(1.0)
product_monomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_dissociation = reaction_fibril{j}_dissociation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"{k_current_prev} * {current_fibril} * VIS_brain")
klaw_dissociation.setMath(math_ast)
'''
        fibril_reaction_blocks.append(dissociation_reaction)
        
        
        # Reaction 5: Antibody binding
        antibody_binding = f'''
# Antibody binding to {current_fibril}
reaction_antibody_binding = model.createReaction()
reaction_antibody_binding.setId(get_unique_reaction_id("AB{ab_type}_antibody_fibril{j}_binding"))
reaction_antibody_binding.setReversible(False)

# Reactants: Current fibril and Antibody
reactant_fibril = reaction_antibody_binding.createReactant()
reactant_fibril.setSpecies("{current_fibril}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

reactant_antibody = reaction_antibody_binding.createReactant()
reactant_antibody.setSpecies("Ab_t")
reactant_antibody.setStoichiometry(1.0)
reactant_antibody.setConstant(True)

# Product: Bound fibril
product_bound = reaction_antibody_binding.createProduct()
product_bound.setSpecies(f"{current_fibril}_Antibody_bound")
product_bound.setStoichiometry(1.0)
product_bound.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_antibody_binding = reaction_antibody_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta2 * {current_fibril} * Ab_t * VIS_brain")
klaw_antibody_binding.setMath(math_ast)
'''
        fibril_reaction_blocks.append(antibody_binding)
        
        # Reaction 6: Microglia clearance
        microglia_clearance = f'''
# Microglia clearance of {current_fibril}
reaction_microglia_clearance = model.createReaction()
reaction_microglia_clearance.setId(get_unique_reaction_id("AB{ab_type}_microglia_fibril{j}_clearance"))
reaction_microglia_clearance.setReversible(False)

# Reactant: Current fibril
reactant_fibril = reaction_microglia_clearance.createReactant()
reactant_fibril.setSpecies("{current_fibril}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

# Modifiers: Add microglia species as modifiers
modifier_cell_count = reaction_microglia_clearance.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction_microglia_clearance.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Add Sink as product
sink_id = create_sink_for_species("{current_fibril}")
product = reaction_microglia_clearance.createProduct()
product.setSpecies(sink_id)
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_microglia_clearance = reaction_microglia_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"Microglia_Vmax_{suffix} * {current_fibril} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
klaw_microglia_clearance.setMath(math_ast)
'''
        fibril_reaction_blocks.append(microglia_clearance)
        
       
        
        # Execute each reaction block for this fibril
        for block in fibril_reaction_blocks:
            if block:  # Skip empty blocks
                exec(block, global_vars)

def create_fibril_24_reactions(model, params, ab_type, suffix):
    """Create all reactions for Fibril24 of a specific AB type (40 or 42)
    
    Args:
        model: The SBML model
        params: Dictionary of parameter values
        ab_type: AB type (40 or 42)
        suffix: Parameter suffix (forty or fortytwo)
    """
    print(f"\nCreating fibril 24 reactions for AB{ab_type}...")
    
    # Create a dictionary to track reaction IDs
    created_reaction_ids = {}
    
    # Dictionary to hold sinks for species
    sinks = {}
    
    # Helper function to get unique reaction IDs
    def get_unique_reaction_id(base_id):
        if base_id not in created_reaction_ids:
            created_reaction_ids[base_id] = 0
            return base_id
        else:
            created_reaction_ids[base_id] += 1
            return f"{base_id}_{created_reaction_ids[base_id]}"
    
    # Initialize global variables dictionary with all necessary variables
    global_vars = {
        'model': model,
        'libsbml': libsbml,
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids,
        'ensure_sigma_parameter': lambda j, ab_type: ensure_sigma_parameter(model, params, j, ab_type),
        'ensure_rate_parameter': lambda param_id: ensure_rate_parameter(model, params, param_id),
        'params': params,
        'ab_type': ab_type,
        'suffix': suffix,
        'create_sink_for_species': lambda species_id: create_sink_for_species(species_id, model),
        'sinks': sinks
    }
    
    # Define species IDs
    fibril24 = f"AB{ab_type}_Fibril24"
    fibril23 = f"AB{ab_type}_Fibril23"
    oligomer12 = f"AB{ab_type}_Oligomer12"
    monomer = f"AB{ab_type}_Monomer"
    
    # Ensure sigma parameter exists
    sigma_param = ensure_sigma_parameter(model, params, 24, ab_type)
    
    # List to hold all reaction blocks for Fibril24
    fibril24_reaction_blocks = []
    
    # Reaction 1: Formation from Fibril23 + Monomer
    formation_reaction = f'''
# Formation of {fibril24} from {fibril23} and monomer
reaction_fibril24_formation = model.createReaction()
reaction_fibril24_formation.setId(get_unique_reaction_id("AB{ab_type}_fibril24_formation"))
reaction_fibril24_formation.setReversible(False)

# Reactants: {fibril23} and monomer
reactant_fibril = reaction_fibril24_formation.createReactant()
reactant_fibril.setSpecies("{fibril23}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

reactant_monomer = reaction_fibril24_formation.createReactant()
reactant_monomer.setSpecies("{monomer}")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Product: {fibril24}
product_fibril = reaction_fibril24_formation.createProduct()
product_fibril.setSpecies("{fibril24}")
product_fibril.setStoichiometry(1.0)
product_fibril.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_formation = reaction_fibril24_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F23_F24_{suffix} * {fibril23} * {monomer} * VIS_brain")
klaw_formation.setMath(math_ast)
'''
    fibril24_reaction_blocks.append(formation_reaction)
    
    # Reaction 2: Dissociation to Fibril23
    dissociation_reaction = f'''
# Dissociation of {fibril24} to {fibril23}
reaction_fibril24_dissociation = model.createReaction()
reaction_fibril24_dissociation.setId(get_unique_reaction_id("AB{ab_type}_fibril24_dissociation"))
reaction_fibril24_dissociation.setReversible(False)

# Reactant: {fibril24}
reactant_fibril = reaction_fibril24_dissociation.createReactant()
reactant_fibril.setSpecies("{fibril24}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

# Product: {fibril23}
product_fibril = reaction_fibril24_dissociation.createProduct()
product_fibril.setSpecies("{fibril23}")
product_fibril.setStoichiometry(1.0)
product_fibril.setConstant(True)

# Product: monomer
product_monomer = reaction_fibril24_dissociation.createProduct()
product_monomer.setSpecies("{monomer}")
product_monomer.setStoichiometry(1.0)
product_monomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_dissociation = reaction_fibril24_dissociation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F24_F23_{suffix} * {fibril24} * VIS_brain")
klaw_dissociation.setMath(math_ast)
'''
    fibril24_reaction_blocks.append(dissociation_reaction)
    
    # Reaction 3: Breakdown to Oligomer12
    breakdown_reaction = f'''
# Breakdown of {fibril24} to two {oligomer12} molecules
reaction_fibril24_breakdown = model.createReaction()
reaction_fibril24_breakdown.setId(get_unique_reaction_id("AB{ab_type}_fibril24_breakdown"))
reaction_fibril24_breakdown.setReversible(False)

# Reactant: {fibril24}
reactant_fibril = reaction_fibril24_breakdown.createReactant()
reactant_fibril.setSpecies("{fibril24}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

# Product: Two {oligomer12} molecules
product_oligomer = reaction_fibril24_breakdown.createProduct()
product_oligomer.setSpecies("{oligomer12}")
product_oligomer.setStoichiometry(2.0)
product_oligomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
# csv has * k_F24_F23_{suffix} as well as k_F24_O12_{suffix}
klaw_breakdown = reaction_fibril24_breakdown.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F24_O12_{suffix} * k_F24_F23_{suffix} * {fibril24} * VIS_brain")
klaw_breakdown.setMath(math_ast)
'''
    fibril24_reaction_blocks.append(breakdown_reaction)
    
    # Reaction 4: Antibody binding
    antibody_binding = f'''
# Antibody binding to {fibril24}
reaction_antibody_binding = model.createReaction()
reaction_antibody_binding.setId(get_unique_reaction_id("AB{ab_type}_antibody_fibril24_binding"))
reaction_antibody_binding.setReversible(False)

# Reactants: {fibril24} and Antibody
reactant_fibril = reaction_antibody_binding.createReactant()
reactant_fibril.setSpecies("{fibril24}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

reactant_antibody = reaction_antibody_binding.createReactant()
reactant_antibody.setSpecies("Ab_t")
reactant_antibody.setStoichiometry(1.0)
reactant_antibody.setConstant(True)

# Product: Bound fibril
product_bound = reaction_antibody_binding.createProduct()
product_bound.setSpecies(f"{fibril24}_Antibody_bound")
product_bound.setStoichiometry(1.0)
product_bound.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_antibody_binding = reaction_antibody_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta2 * {fibril24} * Ab_t * VIS_brain")
klaw_antibody_binding.setMath(math_ast)
'''
    fibril24_reaction_blocks.append(antibody_binding)
    
    # Reaction 5: Microglia clearance
    microglia_clearance = f'''
# Microglia clearance of {fibril24}
reaction_microglia_clearance = model.createReaction()
reaction_microglia_clearance.setId(get_unique_reaction_id("AB{ab_type}_microglia_fibril24_clearance"))
reaction_microglia_clearance.setReversible(False)

# Reactant: {fibril24}
reactant_fibril = reaction_microglia_clearance.createReactant()
reactant_fibril.setSpecies("{fibril24}")
reactant_fibril.setStoichiometry(1.0)
reactant_fibril.setConstant(True)

# Modifiers: Add microglia species as modifiers
modifier_cell_count = reaction_microglia_clearance.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction_microglia_clearance.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Add Sink as product
sink_id = create_sink_for_species("{fibril24}")
product = reaction_microglia_clearance.createProduct()
product.setSpecies(sink_id)
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_microglia_clearance = reaction_microglia_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"Microglia_Vmax_{suffix} * {fibril24} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
klaw_microglia_clearance.setMath(math_ast)
'''
    fibril24_reaction_blocks.append(microglia_clearance)
    
    
    
    # Execute each reaction block
    for block in fibril24_reaction_blocks:
        if block:  # Skip empty blocks
            exec(block, global_vars)

def create_plaque_reactions(model, params, ab_type, suffix):
    """Create all plaque-related reactions for a specific AB type (40 or 42)
    
    Args:
        model: The SBML model
        params: Dictionary of parameter values
        ab_type: AB type (40 or 42)
        suffix: Parameter suffix (forty or fortytwo)
    """
    print(f"\nCreating plaque reactions for AB{ab_type}...")
    
    # Create a dictionary to track reaction IDs
    created_reaction_ids = {}
    
    # Dictionary to hold sinks for species
    sinks = {}
    
    # Helper function to get unique reaction IDs
    def get_unique_reaction_id(base_id):
        if base_id not in created_reaction_ids:
            created_reaction_ids[base_id] = 0
            return base_id
        else:
            created_reaction_ids[base_id] += 1
            return f"{base_id}_{created_reaction_ids[base_id]}"
    
    # Initialize global variables dictionary with all necessary variables
    global_vars = {
        'model': model,
        'libsbml': libsbml,
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids,
        'params': params,
        'ab_type': ab_type,
        'suffix': suffix,
        'create_sink_for_species': lambda species_id: create_sink_for_species(species_id, model),
        'sinks': sinks
    }
    
    # Define species IDs
    plaque_unbound = f"AB{ab_type}_Plaque_unbound"
    plaque_bound = f"AB{ab_type}_Plaque_Antibody_bound"
    monomer = f"AB{ab_type}_Monomer"
    
    # List to hold all reaction blocks for plaque
    plaque_reaction_blocks = []
    
    # Reaction 1: Antibody binding to plaque
    antibody_binding = f'''
# Antibody binding to {plaque_unbound}
reaction_antibody_binding = model.createReaction()
reaction_antibody_binding.setId(get_unique_reaction_id("AB{ab_type}_antibody_plaque_binding"))
reaction_antibody_binding.setReversible(False)

# Reactants: Plaque and Antibody
reactant_plaque = reaction_antibody_binding.createReactant()
reactant_plaque.setSpecies("{plaque_unbound}")
reactant_plaque.setStoichiometry(1.0)
reactant_plaque.setConstant(True)

reactant_antibody = reaction_antibody_binding.createReactant()
reactant_antibody.setSpecies("Ab_t")
reactant_antibody.setStoichiometry(1.0)
reactant_antibody.setConstant(True)

# product 
product_bound_plaque = reaction_antibody_binding.createProduct()
product_bound_plaque.setSpecies("{plaque_bound}")
product_bound_plaque.setStoichiometry(1.0)
product_bound_plaque.setConstant(True)


# Kinetic law: Update to include VIS_brain in formula
klaw_antibody_binding = reaction_antibody_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta3 * {plaque_unbound} * Ab_t * VIS_brain")
klaw_antibody_binding.setMath(math_ast)
'''
    plaque_reaction_blocks.append(antibody_binding)
    
    # Reaction 2: Microglia clearance of plaque (with 0.5 factor)
    microglia_clearance = f'''
# Microglia clearance of {plaque_unbound}
reaction_microglia_clearance = model.createReaction()
reaction_microglia_clearance.setId(get_unique_reaction_id("AB{ab_type}_microglia_plaque_clearance"))
reaction_microglia_clearance.setReversible(False)

# Reactant: Plaque
reactant_plaque = reaction_microglia_clearance.createReactant()
reactant_plaque.setSpecies("{plaque_unbound}")
reactant_plaque.setStoichiometry(1.0)
reactant_plaque.setConstant(True)

# Modifiers: Add microglia species as modifiers since they influence but aren't consumed
modifier_cell_count = reaction_microglia_clearance.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction_microglia_clearance.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Add Sink as product
sink_id = create_sink_for_species("{plaque_unbound}")
product = reaction_microglia_clearance.createProduct()
product.setSpecies(sink_id)
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_microglia_clearance = reaction_microglia_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"0.5 * Microglia_Vmax_{suffix} * {plaque_unbound} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
klaw_microglia_clearance.setMath(math_ast)
'''
    plaque_reaction_blocks.append(microglia_clearance)
    
    # Execute each reaction block
    for block in plaque_reaction_blocks:
        if block:  # Skip empty blocks
            exec(block, global_vars)

def create_fibril_19_24_reactions(model, params, ab_type):
    """Create all reactions for fibrils 19-24 of a specific AB type"""
    print(f"\nCreating reactions for AB{ab_type}...")
    suffix = "forty" if ab_type == "40" else "fortytwo"
    
    # Add debug logging
    print("\nVerifying all species exist before creating reactions:")
    for j in range(19, 25):
        species_id = f"AB{ab_type}_Fibril{j:02d}"
        bound_species_id = f"{species_id}_Antibody_bound"
        exists = model.getSpecies(species_id) is not None
        bound_exists = model.getSpecies(bound_species_id) is not None
        print(f"  {species_id}: {'exists' if exists else 'MISSING'}")
        print(f"  {bound_species_id}: {'exists' if bound_exists else 'MISSING'}")
    
    # Create reactions
    create_fibril_19_23_reactions(model, params, ab_type, suffix)
    create_fibril_24_reactions(model, params, ab_type, suffix)
    create_plaque_reactions(model, params, ab_type, suffix)

def create_fibril_19_24_model_runner():
    """Create a function that loads parameters and creates a parameterized model
        
    Returns:
        Function that takes a parameter CSV path and returns an SBML document
    """
    def model_runner(csv_path, drug_type="gantenerumab"):
        # Load parameters with drug-specific parameter mapping
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        
        # Create model - now using properly mapped parameters
        document = create_fibril_19_24_model(params, params_with_units)
        
        return document
    
    return model_runner

if __name__ == "__main__":
    # Test the model creation with both drug types
    runner = create_fibril_19_24_model_runner()
    
    print("\n*** Testing with GANTENERUMAB ***")
    doc_gant = runner("parameters/PK_Geerts.csv", drug_type="gantenerumab")
    print("Gantenerumab model validation complete")
    
    print("\n*** Testing with LECANEMAB ***")
    doc_lec = runner("parameters/PK_Geerts.csv", drug_type="lecanemab")
    print("Lecanemab model validation complete") 