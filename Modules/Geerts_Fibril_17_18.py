"""
Module for modeling small Aβ fibril formation and dynamics (17-18mers).
This is a QSP component that models the formation and behavior of small Aβ fibrils.
The module includes:
- Fibril formation from oligomers and monomers
- Fibril dissociation to smaller species
- Plaque formation from fibrils
- Antibody binding to fibrils
- Microglia-mediated clearance of fibrils
- Transport and reflection coefficients for fibrils
"""

# deleted fibril 19 formation and dissociation, they are in the fibril 19-24 model
import libsbml
from pathlib import Path
import pandas as pd
from K_rates_extrapolate import calculate_k_rates
import numpy as np

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

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} in Fibril 17-18 module ===")
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
    print("\nAntibody binding parameters for Fibril 17-18:")
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

def ensure_sigma_parameter(model, params, j, ab_type):
    """Ensure that sigma parameter for fibril j exists
    
    Args:
        model: The SBML model
        params: Dictionary of parameter values
        j: Fibril size
        ab_type: AB type (40 or 42)
    
    Returns:
        Name of the sigma parameter
    """
    sigma_param = f"sigma_ISF_ABeta{ab_type}_oligomer{j:02d}"
    
    # Create parameter if it doesn't exist
    if model.getParameter(sigma_param) is None:
        sigma_value = params.get(sigma_param, 0.95)  # Default value
        param = model.createParameter()
        param.setId(sigma_param)
        param.setValue(sigma_value)
        param.setConstant(True)
        param.setUnits("dimensionless")
    
    return sigma_param

def ensure_rate_parameter(model, params, param_id, default_value=0.1):
    """Ensure that a rate parameter exists
    
    Args:
        model: The SBML model
        params: Dictionary of parameter values
        param_id: Parameter ID
        default_value: Default value for the parameter
    
    Returns:
        Name of the parameter
    """
    # Create parameter if it doesn't exist
    if model.getParameter(param_id) is None:
        param_value = params.get(param_id, default_value)
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(param_value)
        param.setConstant(True)
        
        # Determine units based on parameter type
        if param_id.startswith('k_F') and "_F" in param_id.split("k_")[1]:
            param.setUnits("per_mole_per_hour")
        else:
            param.setUnits("per_hour")
    
    return param_id

def create_fibril_17_18_model(params, params_with_units):
    """Create a parameterized SBML model for the Geerts Fibril 17-18 model
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
        
    Returns:
        SBML document
    """
    print("\nCreating Geerts Fibril 17-18 model...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_Fibril_17_18_Model")
    
    # Add a global dictionary to track used reaction IDs across all AB types
    created_reaction_ids = {}
    
    # Helper function to ensure unique reaction IDs
    def get_unique_reaction_id(base_id):
        """Create a unique reaction ID by incrementing counter if base ID already exists"""
        if base_id not in created_reaction_ids:
            created_reaction_ids[base_id] = 0
            return base_id
        else:
            created_reaction_ids[base_id] += 1
            return f"{base_id}_{created_reaction_ids[base_id]}"
    
    # Helper function to check if a species exists
    def check_species_exists(species_id):
        """Check if a species already exists in the model"""
        return model.getSpecies(species_id) is not None
    
    # Helper function to ensure sigma parameter exists
    def ensure_sigma_parameter(j, ab_type):
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
    def ensure_rate_parameter(param_id):
        """Ensure that a rate parameter exists"""
        # Create parameter if it doesn't exist
        if model.getParameter(param_id) is None:
            param = model.createParameter()
            param.setId(param_id)
            param.setValue(params[param_id])
            param.setConstant(True)
            
            # Determine units based on parameter type
            if param_id.startswith('k_F') and "_F" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")
            else:
                param.setUnits("per_hour")
        
        return param_id
    
    # Helper function to create Fibril17 reactions
    def create_fibril_17_reactions(ab_type, suffix):
        """Create all reactions for Fibril17 of a specific AB type"""
        print(f"  Creating Fibril17 reactions for AB{ab_type}...")
        
        # Ensure sigma parameter exists
        sigma_param = ensure_sigma_parameter(17, ab_type)
        monomer = f"AB{ab_type}_Monomer"
        # Fibril17 species ID
        fibril17 = f"AB{ab_type}_Fibril17"
        
        # 1. Formation from Oligomer16 + Monomer
        formation_from_oligomer = f'''
# Formation of {fibril17} from Oligomer16 + Monomer
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril17_formation_from_oligomer"))
reaction.setReversible(False)

# Reactant 1: Oligomer16
reactant1 = reaction.createReactant()
reactant1.setSpecies(f"AB{ab_type}_Oligomer16")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Monomer
reactant2 = reaction.createReactant()
reactant2.setSpecies(f"AB{ab_type}_Monomer")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Fibril17
product = reaction.createProduct()
product.setSpecies("{fibril17}")
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: k_O16_F17_{suffix} * AB{ab_type}_Oligomer16 * AB{ab_type}_Monomer * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_O16_F17_{suffix} * AB{ab_type}_Oligomer16 * AB{ab_type}_Monomer * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 2. Dissociation to Oligomer16
        dissociation_to_oligomer = f'''
# Dissociation of {fibril17} to Oligomer16
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril17_dissociation_to_oligomer"))
reaction.setReversible(False)

# Reactant: Fibril17
reactant = reaction.createReactant()
reactant.setSpecies("{fibril17}")
reactant.setStoichiometry(1.0)
reactant.setConstant(True)

# Product: Oligomer16
product = reaction.createProduct()
product.setSpecies(f"AB{ab_type}_Oligomer16")
product.setStoichiometry(1.0)
product.setConstant(True)

# Product: monomer
product_monomer = reaction.createProduct()
product_monomer.setSpecies("{monomer}")
product_monomer.setStoichiometry(1.0)
product_monomer.setConstant(True)

# Kinetic law: k_F17_O16_{suffix} * {fibril17} * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F17_O16_{suffix} * {fibril17} * VIS_brain")
klaw.setMath(math_ast)
'''
        
        
        # 5. Plaque formation
        plaque_formation = f'''
# Plaque formation from {fibril17} + Monomer
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Plaque_formation_from_Fibril17"))
reaction.setReversible(False)

# Reactant 1: Fibril17
reactant1 = reaction.createReactant()
reactant1.setSpecies("{fibril17}")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Monomer
reactant2 = reaction.createReactant()
reactant2.setSpecies(f"AB{ab_type}_Monomer")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Plaque
product = reaction.createProduct()
product.setSpecies(f"AB{ab_type}_Plaque_unbound")
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: k_F17_Plaque_{suffix} * {fibril17} * AB{ab_type}_Monomer * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F17_Plaque_{suffix} * {fibril17} * AB{ab_type}_Monomer * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 6. Plaque-driven Oligomer16 to Fibril17 conversion
        plaque_driven_formation = f'''
# Plaque-driven formation of {fibril17} from Oligomer16 + Monomer
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril17_plaque_driven_formation"))
reaction.setReversible(False)

# Reactant 1: Oligomer16
reactant1 = reaction.createReactant()
reactant1.setSpecies(f"AB{ab_type}_Oligomer16")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Monomer
reactant2 = reaction.createReactant()
reactant2.setSpecies(f"AB{ab_type}_Monomer")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Fibril17
product = reaction.createProduct()
product.setSpecies("{fibril17}")
product.setStoichiometry(1.0)
product.setConstant(True)

# Modifier: Plaque (used in rate equation but not consumed)
modifier = reaction.createModifier()
modifier.setSpecies(f"AB{ab_type}_Plaque_unbound")

# Kinetic law: k_O16_F17_{suffix} * AB{ab_type}_Plaque_Driven_Monomer_Addition_Vmax * AB{ab_type}_Monomer * AB{ab_type}_Oligomer16 * (AB{ab_type}_Plaque_unbound / (AB{ab_type}_Plaque_unbound + AB{ab_type}_Plaque_Driven_Monomer_Addition_EC50)) * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_O16_F17_{suffix} * AB{ab_type}_Plaque_Driven_Monomer_Addition_Vmax * AB{ab_type}_Monomer * AB{ab_type}_Oligomer16 * (AB{ab_type}_Plaque_unbound / (AB{ab_type}_Plaque_unbound + AB{ab_type}_Plaque_Driven_Monomer_Addition_EC50)) * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 7. Antibody binding/clearance
        antibody_binding = f'''
# Antibody binding to {fibril17}
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril17_antibody_binding"))
reaction.setReversible(False)

# Reactant 1: Fibril17
reactant1 = reaction.createReactant()
reactant1.setSpecies("{fibril17}")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Antibody
reactant2 = reaction.createReactant()
reactant2.setSpecies("Ab_t")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Bound fibril
product = reaction.createProduct()
product.setSpecies(f"{fibril17}_Antibody_bound")
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: fta1 * Ab_t * {fibril17} * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta1 * Ab_t * {fibril17} * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 8. Microglia clearance
        microglia_clearance = f'''
# Microglia clearance of {fibril17}
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril17_microglia_clearance"))
reaction.setReversible(False)

# Reactant: Fibril17
reactant = reaction.createReactant()
reactant.setSpecies("{fibril17}")
reactant.setStoichiometry(1.0)
reactant.setConstant(True)

# Modifiers: Add microglia species as modifiers
modifier_cell_count = reaction.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Add Sink as product
sink_id = create_sink_for_species("{fibril17}", model)
product = reaction.createProduct()
product.setSpecies(sink_id)
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: Baseline clearance only (old form)
klaw = reaction.createKineticLaw()
# math_ast = libsbml.parseL3Formula(f"{fibril17} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
# Kinetic law: Saturable clearance (Vmax/EC50, current form)
math_ast = libsbml.parseL3Formula(f"{fibril17} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_Hi_Lo_ratio * (Microglia_Vmax_{suffix}/(Microglia_EC50_{suffix} + {fibril17})) + (1 - Microglia_Hi_Fract) * (Microglia_Vmax_{suffix}/(Microglia_EC50_{suffix} + {fibril17}))) * VIS_brain")
klaw.setMath(math_ast)
'''
        
        
        
        # Execute all the reaction blocks
        reaction_blocks = [
            formation_from_oligomer,
            dissociation_to_oligomer,
            plaque_formation,
            plaque_driven_formation,
            antibody_binding,
            microglia_clearance
        ]
        
        print(f"  Creating {len(reaction_blocks)} reactions for {fibril17}...")
        for block in reaction_blocks:
            exec(block, global_vars)
    
    # Helper function to create Fibril18 reactions
    def create_fibril_18_reactions(ab_type, suffix):
        """Create all reactions for Fibril18 of a specific AB type"""
        print(f"  Creating Fibril18 reactions for AB{ab_type}...")
        
        # Ensure sigma parameter exists
        sigma_param = ensure_sigma_parameter(18, ab_type)
        
        # Fibril18 species ID
        fibril18 = f"AB{ab_type}_Fibril18"
        monomer = f"AB{ab_type}_Monomer"
        fibril17 = f"AB{ab_type}_Fibril17"
        
        # Note: Formation from Fibril17 + Monomer and dissociation to Fibril17
        # are already implemented in the Fibril17 reactions
    # 3. Formation of Fibril18 from Fibril17 + Monomer
        formation_of_fibril18 = f'''
# Formation of Fibril18 from {fibril17} + Monomer
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril18_formation"))
reaction.setReversible(False)

# Reactant 1: Fibril17
reactant1 = reaction.createReactant()
reactant1.setSpecies("{fibril17}")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Monomer
reactant2 = reaction.createReactant()
reactant2.setSpecies(f"AB{ab_type}_Monomer")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Fibril18
product = reaction.createProduct()
product.setSpecies(f"AB{ab_type}_Fibril18")
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: k_F17_F18_{suffix} * {fibril17} * AB{ab_type}_Monomer * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F17_F18_{suffix} * {fibril17} * AB{ab_type}_Monomer * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 4. Dissociation from Fibril18
        dissociation_from_fibril18 = f'''
# Dissociation from Fibril18 to {fibril17}
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril17_formation_from_fibril18"))
reaction.setReversible(False)

# Reactant: Fibril18
reactant = reaction.createReactant()
reactant.setSpecies(f"AB{ab_type}_Fibril18")
reactant.setStoichiometry(1.0)
reactant.setConstant(True)

# Product: Fibril17
product = reaction.createProduct()
product.setSpecies("{fibril17}")
product.setStoichiometry(1.0)
product.setConstant(True)

# Product: monomer
product_monomer = reaction.createProduct()
product_monomer.setSpecies("{monomer}")
product_monomer.setStoichiometry(1.0)
product_monomer.setConstant(True)

# Kinetic law: k_F18_F17_{suffix} * AB{ab_type}_Fibril18 * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F18_F17_{suffix} * AB{ab_type}_Fibril18 * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 3. Plaque formation
        plaque_formation = f'''
# Plaque formation from {fibril18} + Monomer
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Plaque_formation_from_Fibril18"))
reaction.setReversible(False)

# Reactant 1: Fibril18
reactant1 = reaction.createReactant()
reactant1.setSpecies("{fibril18}")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Monomer
reactant2 = reaction.createReactant()
reactant2.setSpecies(f"AB{ab_type}_Monomer")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Plaque
product = reaction.createProduct()
product.setSpecies(f"AB{ab_type}_Plaque_unbound")
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: k_F18_Plaque_{suffix} * {fibril18} * AB{ab_type}_Monomer * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_F18_Plaque_{suffix} * {fibril18} * AB{ab_type}_Monomer * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 4. Antibody binding/clearance (using fta2 instead of fta1 for larger fibrils)
        antibody_binding = f'''
# Antibody binding to {fibril18}
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril18_antibody_binding"))
reaction.setReversible(False)

# Reactant 1: Fibril18
reactant1 = reaction.createReactant()
reactant1.setSpecies("{fibril18}")
reactant1.setStoichiometry(1.0)
reactant1.setConstant(True)

# Reactant 2: Antibody
reactant2 = reaction.createReactant()
reactant2.setSpecies("Ab_t")
reactant2.setStoichiometry(1.0)
reactant2.setConstant(True)

# Product: Bound fibril
product = reaction.createProduct()
product.setSpecies(f"{fibril18}_Antibody_bound")
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: fta2 * Ab_t * {fibril18} * VIS_brain
klaw = reaction.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta2 * Ab_t * {fibril18} * VIS_brain")
klaw.setMath(math_ast)
'''
        
        # 5. Microglia clearance
        microglia_clearance = f'''
# Microglia clearance of {fibril18}
reaction = model.createReaction()
reaction.setId(get_unique_reaction_id("AB{ab_type}_Fibril18_microglia_clearance"))
reaction.setReversible(False)

# Reactant: Fibril18
reactant = reaction.createReactant()
reactant.setSpecies("{fibril18}")
reactant.setStoichiometry(1.0)
reactant.setConstant(True)

# Modifiers: Add microglia species as modifiers
modifier_cell_count = reaction.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Add Sink as product
sink_id = create_sink_for_species("{fibril18}", model)
product = reaction.createProduct()
product.setSpecies(sink_id)
product.setStoichiometry(1.0)
product.setConstant(True)

# Kinetic law: Baseline clearance only (old form)
klaw = reaction.createKineticLaw()
# math_ast = libsbml.parseL3Formula(f"{fibril18} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
# Kinetic law: Saturable clearance (Vmax/EC50, current form)
math_ast = libsbml.parseL3Formula(f"{fibril18} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_Hi_Lo_ratio * (Microglia_Vmax_{suffix}/(Microglia_EC50_{suffix} + {fibril18})) + (1 - Microglia_Hi_Fract) * (Microglia_Vmax_{suffix}/(Microglia_EC50_{suffix} + {fibril18}))) * VIS_brain")
klaw.setMath(math_ast)
'''
        
        
        
        # Execute all the reaction blocks
        reaction_blocks = [
            formation_of_fibril18,
            dissociation_from_fibril18,
            plaque_formation,
            antibody_binding,
            microglia_clearance
        ]
        
        print(f"  Creating {len(reaction_blocks)} reactions for {fibril18}...")
        for block in reaction_blocks:
            exec(block, global_vars)
            # Add debug logging
            print(f"    Created reaction: {block.split('reaction.setId(')[1].split(')')[0]}")
    
    # Helper function to create all fibril reactions
    def create_fibril_17_18_reactions(ab_type):
        """Create all reactions for fibrils 17-18 of a specific AB type"""
        print(f"\nCreating fibril 17-18 reactions for AB{ab_type}...")
        suffix = "forty" if ab_type == "40" else "fortytwo"
        
        create_fibril_17_reactions(ab_type, suffix)
        create_fibril_18_reactions(ab_type, suffix)
        
        print(f"Created reactions for AB{ab_type} fibrils 17-18")
    
    # Add helper functions to global variables used in exec
    global_vars = globals().copy()
    
    # Create dictionaries to hold sink and source species
    sinks = {}
    sources = {}
    
    global_vars.update({
        'model': model, 
        'libsbml': libsbml,
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids,
        'check_species_exists': check_species_exists,
        'ensure_sigma_parameter': ensure_sigma_parameter,
        'ensure_rate_parameter': ensure_rate_parameter,
        'create_sink_for_species': create_sink_for_species,
        'create_source_for_species': create_source_for_species,
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

    # Binding affinity constants (mole/litre)
    affinity = model.createUnitDefinition()
    affinity.setId("nanomole_per_litre")
    nanomole_unit = affinity.createUnit()
    nanomole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nanomole_unit.setExponent(1.0)
    nanomole_unit.setScale(0)
    nanomole_unit.setMultiplier(1.0)
    litre_unit = affinity.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(-1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)

    # Per mole per hour unit
    per_nanomole_per_hour = model.createUnitDefinition()
    per_nanomole_per_hour.setId("per_nanomole_per_hour")
    
    # Mole unit (per mole)
    nanomole_unit = per_nanomole_per_hour.createUnit()
    nanomole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nanomole_unit.setExponent(-1)
    nanomole_unit.setScale(0)
    nanomole_unit.setMultiplier(1.0)
    
    hour_unit = per_nanomole_per_hour.createUnit()
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

    # Define parameters needed for fibril 17-18 reactions
    fibril_params = []
    
    # Add reflection coefficients for fibrils 17-18
    for j in range(17, 19):
        fibril_params.extend([
            (f"sigma_ISF_ABeta40_oligomer{j:02d}", params[f"sigma_ISF_ABeta40_oligomer{j:02d}"]),
            (f"sigma_ISF_ABeta42_oligomer{j:02d}", params[f"sigma_ISF_ABeta42_oligomer{j:02d}"]),
        ])

    # Add plaque formation parameters
    for j in range(17, 19):
        fibril_params.extend([
            (f"k_F{j}_Plaque_forty", params[f"k_F{j}_Plaque_forty"]),
            (f"k_F{j}_Plaque_fortytwo", params[f"k_F{j}_Plaque_fortytwo"]),
        ])
    
    # Add fibril formation parameters
    for j in range(17, 19):
        if j > 17:
            fibril_params.extend([
                (f"k_F{j-1}_F{j}_forty", params[f"k_F{j-1}_F{j}_forty"]),
                (f"k_F{j-1}_F{j}_fortytwo", params[f"k_F{j-1}_F{j}_fortytwo"]),
                (f"k_F{j}_F{j-1}_forty", params[f"k_F{j}_F{j-1}_forty"]),
                (f"k_F{j}_F{j-1}_fortytwo", params[f"k_F{j}_F{j-1}_fortytwo"]),
            ])
    
    # Add special parameters for Fibril17
    fibril_params.extend([
        # Connection to Oligomer16
        ("k_O16_F17_forty", params["k_O16_F17_forty"]),
        ("k_O16_F17_fortytwo", params["k_O16_F17_fortytwo"]),
        ("k_F17_O16_forty", params["k_F17_O16_forty"]),
        ("k_F17_O16_fortytwo", params["k_F17_O16_fortytwo"]),
    ])
    
    # Add connection to Fibril19 (needed for Fibril18 reactions)
    fibril_params.extend([
        ("k_F18_F19_forty", params["k_F18_F19_forty"]),
        ("k_F18_F19_fortytwo", params["k_F18_F19_fortytwo"]),
        ("k_F19_F18_forty", params["k_F19_F18_forty"]),
        ("k_F19_F18_fortytwo", params["k_F19_F18_fortytwo"]),
    ])
    
    # Add other non-fibrillization parameters
    fibril_params.extend([
        # Plaque-driven parameters
        ("AB40_Plaque_Driven_Monomer_Addition_Vmax", params["AB40_Plaque_Driven_Monomer_Addition_Vmax"]),
        ("AB42_Plaque_Driven_Monomer_Addition_Vmax", params["AB42_Plaque_Driven_Monomer_Addition_Vmax"]),
        ("AB40_Plaque_Driven_Monomer_Addition_EC50", params["AB40_Plaque_Driven_Monomer_Addition_EC50"]),
        ("AB42_Plaque_Driven_Monomer_Addition_EC50", params["AB42_Plaque_Driven_Monomer_Addition_EC50"]),
        
        # Antibody binding parameters
        ("fta1", params["fta1"]),  # Antibody binding to smaller fibrils
        ("fta2", params["fta2"]),  # Antibody binding to larger fibrils
        ("fta3", params["fta3"]),  # Antibody binding to plaque
        
        # Microglia parameters - updated to use Vmax/EC50 approach
        ("Microglia_Vmax_forty", params["Microglia_Vmax_forty"]),
        ("Microglia_Vmax_fortytwo", params["Microglia_Vmax_fortytwo"]),
        ("Microglia_EC50_forty", params["Microglia_EC50_forty"]),
        ("Microglia_EC50_fortytwo", params["Microglia_EC50_fortytwo"]),
        ("Microglia_Hi_Lo_ratio", params["Microglia_Hi_Lo_ratio"]),
        # Microglia CL parameters for baseline clearance
        ("Microglia_CL_high_AB40", params["Microglia_CL_high_AB40"]),
        ("Microglia_CL_low_AB40", params["Microglia_CL_low_AB40"]),
        ("Microglia_CL_high_AB42", params["Microglia_CL_high_AB42"]),
        ("Microglia_CL_low_AB42", params["Microglia_CL_low_AB42"]),
        
        
        # Other parameters needed for reactions
        ("VIS_brain", params["VIS_brain"]),  # ISF volume
    ])
    
    # Add the parameters to the model
    print("\nCreating fibril 17-18 parameters...")
    for param_id, value in fibril_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Assign appropriate units based on parameter type
        if param_id.startswith('k_F'):
            if "_F" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")  # Second-order rate constant
            else:
                param.setUnits("per_hour")  # First-order rate constant
        elif param_id.startswith('k_O'):
            if "_F" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")  # Second-order rate constant
            else:
                param.setUnits("per_hour")  # First-order rate constant
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
    
    # Add fibril species for AB40
    for i in range(17, 19):
        fibril_species.append((f"AB40_Fibril{i:02d}", "comp_ISF_brain", 0.0))
    
    # Add fibril species for AB42
    for i in range(17, 19):
        fibril_species.append((f"AB42_Fibril{i:02d}", "comp_ISF_brain", 0.0))
    
    # Pre-create all sinks needed for fibril species
    for ab_type in ["40", "42"]:
        for i in range(17, 19):
            fibril_id = f"AB{ab_type}_Fibril{i:02d}"
            sinks[fibril_id] = create_sink_for_species(fibril_id, model)
    
    # Also include necessary other species for reaction connectivity
    fibril_species.extend([
        # Monomers
        ("AB40_Monomer", "comp_ISF_brain", 0.0),
        ("AB42_Monomer", "comp_ISF_brain", 0.0),
        
        # Oligomer16 (connects to Fibril17)
        ("AB40_Oligomer16", "comp_ISF_brain", 0.0),
        ("AB42_Oligomer16", "comp_ISF_brain", 0.0),
        
        # Fibril19 (needed for Fibril18 reactions)
        ("AB40_Fibril19", "comp_ISF_brain", 0.0),
        ("AB42_Fibril19", "comp_ISF_brain", 0.0),
        
        # Antibody and plaque species
        ("Ab_t", "comp_ISF_brain", 0.0),
        ("AB40_Plaque_unbound", "comp_ISF_brain", 0.0),
        ("AB42_Plaque_unbound", "comp_ISF_brain", 0.0),
    ])
    
    # Add bound fibril species for AB40 and AB42
    for i in range(17, 19):
        fibril_species.extend([
            (f"AB40_Fibril{i:02d}_Antibody_bound", "comp_ISF_brain", 0.0),
            (f"AB42_Fibril{i:02d}_Antibody_bound", "comp_ISF_brain", 0.0),
        ])

    # Add microglia species
    fibril_species.extend([
        ("Microglia_Hi_Fract", "comp_microglia", params["Microglia_Hi_Fract_0"]),
        ("Microglia_cell_count", "comp_microglia", params["Microglia_cell_count_0"]),
    ])
    
    # Create all species
    print("\nCreating fibril 17-18 species with initial concentrations from parameters...")
    for species_id, compartment_id, initial_value in fibril_species:
        # Skip if species already exists
        if check_species_exists(species_id):
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
    
    # Create reactions for fibrils 17-18
    create_fibril_17_18_reactions("40")
    create_fibril_17_18_reactions("42")
    
    # Log the number of reactions created
    print(f"\nFibril 17-18 model created with:")
    print(f"  - {model.getNumCompartments()} compartments")
    print(f"  - {model.getNumSpecies()} species")
    print(f"  - {model.getNumParameters()} parameters")
    print(f"  - {model.getNumReactions()} reactions")
    
    return document

def create_fibril_17_18_model_runner():
    """Create a function that loads parameters and creates a parameterized model
        
    Returns:
        Function that takes a parameter CSV path and returns an SBML document
    """
    def model_runner(csv_path, drug_type="gantenerumab"):
        # Load parameters with drug-specific parameter mapping
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        
        # Create model - now using properly mapped parameters
        document = create_fibril_17_18_model(params, params_with_units)
        
        return document
    
    return model_runner

if __name__ == "__main__":
    # Test the model creation with both drug types
    runner = create_fibril_17_18_model_runner()
    
    print("\n*** Testing with GANTENERUMAB ***")
    doc_gant = runner("parameters/PK_Geerts.csv", drug_type="gantenerumab")
    print("Gantenerumab model validation complete")
    
    print("\n*** Testing with LECANEMAB ***")
    doc_lec = runner("parameters/PK_Geerts.csv", drug_type="lecanemab")
    print("Lecanemab model validation complete") 