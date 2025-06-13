"""
Module for modeling medium Aβ oligomer formation and dynamics (13-16mers).
This is a QSP component that models the formation and behavior of medium-sized Aβ oligomers.

The module includes:
- Oligomer formation from smaller species and monomers
- Oligomer dissociation to smaller species
- Plaque formation from oligomers
- Antibody binding to oligomers
- Microglia-mediated clearance of oligomers
- Transport and reflection coefficients for oligomers

Rate Constants:
This module uses rate constants extrapolated by K_rates_extrapolate.py, which calculates
forward and backward rates for oligomerization based on known experimental values for
small oligomers. The rates are extrapolated using Hill functions to capture the size-dependent
behavior of oligomer formation and dissociation. For these medium-sized oligomers (13-16mers),
the extrapolation is particularly important as direct experimental measurements are challenging.
"""

# deleted formation of next from current and monomer and formation of next from current and monomer plaque catalyzed
# deleted shrinkage of next oligomer to current oligomer and monomer
import libsbml
from pathlib import Path
import pandas as pd
import numpy as np
from K_rates_extrapolate import calculate_k_rates

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} in Oligomer 13-16 module ===")
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
    print("\nAntibody binding parameters for Oligomer 13-16:")
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

def create_oligomer_13_16_model(params, params_with_units):
    """Create a parameterized SBML model for the Geerts Oligomer 13-16 model
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
        
    Returns:
        SBML document
    """
    print("\nCreating Geerts Oligomer 13-16 model...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_Oligomer_13_16_Model")
    
    # Add a global dictionary to track used reaction IDs across all AB types
    created_reaction_ids = {}
    
    # Helper function to ensure unique reaction IDs
    def get_unique_reaction_id(base_id):
        """Create a unique reaction ID by incrementing counter if base ID already exists
        
        Args:
            base_id: Base ID for the reaction
        
        Returns:
            Unique reaction ID string
        """
        if base_id not in created_reaction_ids:
            created_reaction_ids[base_id] = 0
            return base_id
        else:
            created_reaction_ids[base_id] += 1
            return f"{base_id}_{created_reaction_ids[base_id]}"
    
    # Helper function to check if a species exists
    def check_species_exists(species_id):
        """Check if a species already exists in the model
        
        Args:
            species_id: ID of the species to check
            
        Returns:
            True if species exists, False otherwise
        """
        return model.getSpecies(species_id) is not None
    
    
    
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

    # Binding affinity constants (nanomole/litre)
    affinity = model.createUnitDefinition()
    affinity.setId("nanomole_per_litre")
    mole_unit = affinity.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(1.0)
    mole_unit.setScale(0)
    mole_unit.setMultiplier(1.0)
    litre_unit = affinity.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(-1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)

    # Per mole per hour unit
    per_mole_per_hour = model.createUnitDefinition()
    per_mole_per_hour.setId("per_nanomole_per_hour")
    
    # Mole unit (per mole)
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

    # Define common compartments
    compartments = {
        "comp_ISF_brain": params["VIS_brain"],
        "comp_microglia": params["V_microglia"],
    }

    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setUnits('litre')
        
    # Create dictionaries to store sinks and sources for each species
    sinks = {}
    sources = {}
    
    # Pre-create sinks for oligomer species that need them
    for ab_type in ["40", "42"]:
        for i in range(13, 17):
            # Create sink for each oligomer
            oligomer_species = f"AB{ab_type}_Oligomer{i:02d}"
            sinks[oligomer_species] = create_sink_for_species(oligomer_species, model)

    # Define parameters needed for oligomer 13-16 reactions
    oligomer_params = []
    
    # Add reflection coefficients for oligomers 13-16
    for j in range(13, 17):
        oligomer_params.extend([
            (f"sigma_ISF_ABeta40_oligomer{j:02d}", params[f"sigma_ISF_ABeta40_oligomer{j:02d}"]),
            (f"sigma_ISF_ABeta42_oligomer{j:02d}", params[f"sigma_ISF_ABeta42_oligomer{j:02d}"]),
        ])

    # Add plaque formation parameters
    for j in range(13, 17):
        oligomer_params.extend([
            (f"k_O{j}_Plaque_forty", params[f"k_O{j}_Plaque_forty"]),
            (f"k_O{j}_Plaque_fortytwo", params[f"k_O{j}_Plaque_fortytwo"]),
        ])
    
    # Add other non-oligomerization parameters
    oligomer_params.extend([
        # Plaque-driven parameters
        ("AB40_Plaque_Driven_Monomer_Addition_Vmax", params["AB40_Plaque_Driven_Monomer_Addition_Vmax"]),
        ("AB42_Plaque_Driven_Monomer_Addition_Vmax", params["AB42_Plaque_Driven_Monomer_Addition_Vmax"]),
        ("AB40_Plaque_Driven_Monomer_Addition_EC50", params["AB40_Plaque_Driven_Monomer_Addition_EC50"]),
        ("AB42_Plaque_Driven_Monomer_Addition_EC50", params["AB42_Plaque_Driven_Monomer_Addition_EC50"]),
        
        # Antibody binding parameters
        ("fta1", params["fta1"]),  # Antibody binding to oligomer
        
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
    print("\nCreating oligomer 13-16 parameters...")
    for param_id, value in oligomer_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Assign appropriate units based on parameter type
        if param_id.startswith('k_O'):
            if "_O" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")
            else:
                param.setUnits("per_hour")
        elif param_id.startswith('sigma_'):
            param.setUnits("dimensionless")  # Reflection coefficient
        elif param_id.startswith('Q_'):
            param.setUnits("litre_per_hour")  # Flow rate
        elif param_id.startswith('vol_'):
            param.setUnits("litre")  # Volume
        elif param_id.startswith('k_O'):
            param.setUnits("per_nanomole_per_hour")
        else:
            param.setUnits("dimensionless")

    # Now add the extrapolated rate parameters from calculate_k_rates
    print("\nAdding extrapolated rate parameters for oligomers 13-16...")
    for i in range(13, 17):
        # Forward rates (Oi-1 to Oi)
        if i > 1:
            for ab_type, suffix in [("40", "forty"), ("42", "fortytwo")]:
                rate_name = f"k_O{i-1}_O{i}_{suffix}"
                if rate_name in params:
                    param = model.createParameter()
                    param.setId(rate_name)
                    param.setValue(params[rate_name])
                    param.setConstant(True)
                    param.setUnits("per_nanomole_per_hour")
                    print(f"  Set {rate_name} = {params[rate_name]}")
        
        # Backward rates (Oi to Oi-1)
        if i > 1:
            for ab_type, suffix in [("40", "forty"), ("42", "fortytwo")]:
                rate_name = f"k_O{i}_O{i-1}_{suffix}"
                if rate_name in params:
                    param = model.createParameter()
                    param.setId(rate_name)
                    param.setValue(params[rate_name])
                    param.setConstant(True)
                    param.setUnits("per_hour")
                    print(f"  Set {rate_name} = {params[rate_name]}")
    
    # Define common oligomer species
    oligomer_species = []
    
    # Add oligomer species for AB40
    for i in range(13, 17):
        oligomer_species.append((f"AB40_Oligomer{i:02d}", "comp_ISF_brain", 0.0))
    
    # Add oligomer species for AB42
    for i in range(13, 17):
        oligomer_species.append((f"AB42_Oligomer{i:02d}", "comp_ISF_brain", 0.0))
    
    # Also include monomer and small oligomer species for reaction connectivity
    oligomer_species.extend([
        ("AB40_Monomer", "comp_ISF_brain", 0.0),
        ("AB42_Monomer", "comp_ISF_brain", 0.0),
        ("AB40_Oligomer12", "comp_ISF_brain", 0.0),
        ("AB42_Oligomer12", "comp_ISF_brain", 0.0),
        # Add antibody and plaque species
        ("Ab_t", "comp_ISF_brain", 0.0),
        ("AB40_Plaque_unbound", "comp_ISF_brain", 0.0),
        ("AB42_Plaque_unbound", "comp_ISF_brain", 0.0),
    ])
    
    # Add bound oligomer species for AB40 and AB42
    for i in range(13, 17):
        oligomer_species.extend([
            (f"AB40_Oligomer{i:02d}_Antibody_bound", "comp_ISF_brain", 0.0),
            (f"AB42_Oligomer{i:02d}_Antibody_bound", "comp_ISF_brain", 0.0),
        ])
    
    # Add microglia species
    oligomer_species.extend([
        ("Microglia_Hi_Fract", "comp_microglia", params["Microglia_Hi_Fract_0"]),
        ("Microglia_cell_count", "comp_microglia", params["Microglia_cell_count_0"]),
    ])
    
    # Create all species
    print("\nCreating oligomer 13-16 species with initial concentrations from parameters...")
    for species_id, compartment_id, initial_value in oligomer_species:
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
    
    # Add antibody-related species that might be missing
    antibody_species = [
        ("Ab_t", "comp_ISF_brain", params.get("Ab_t_0", 0.0)),
            ]
    
    # Create antibody compartment if it doesn't exist
    if model.getCompartment("comp_ISF_brain") is None:
        comp = model.createCompartment()
        comp.setId("comp_ISF_brain")
        comp.setConstant(True)
        comp.setSize(params["VIS_brain"])
        print("Created compartment: comp_ISF_brain")
    
    # Create antibody species if they don't exist
    for species_id, compartment_id, initial_value in antibody_species:
        if model.getSpecies(species_id) is None:
            species = model.createSpecies()
            species.setId(species_id)
            species.setCompartment(compartment_id)
            
            # Check for parameter in CSV first with proper naming
            param_name = f"{species_id}_0"
            if param_name in params:
                initial_conc = params[param_name]
                print(f"  Setting {species_id} initial concentration from CSV: {initial_conc}")
            else:
                initial_conc = initial_value
                print(f"  Parameter {param_name} not found, using default: {initial_value}")
                
            species.setInitialConcentration(initial_conc)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {species_id}")
    
    # Helper function to ensure sigma parameter exists
    def ensure_sigma_parameter(j, ab_type):
        """Ensure that sigma parameter for oligomer j exists
        
        Args:
            j: Oligomer size
            ab_type: AB type (40 or 42)
        
        Returns:
            Name of the sigma parameter
        """
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
        """Ensure that a rate parameter exists
        
        Args:
            param_id: Parameter ID
        
        Returns:
            Name of the parameter
        """
        # Create parameter if it doesn't exist
        if model.getParameter(param_id) is None:
            param = model.createParameter()
            param.setId(param_id)
            param.setValue(params[param_id])
            param.setConstant(True)
            
            # Determine units based on parameter type
            if param_id.startswith('k_O') and "_O" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")
            else:
                param.setUnits("per_hour")
        
        return param_id
    
    # Add helper functions to global variables
    global_vars = globals().copy()
    global_vars.update({
        'model': model, 
        'libsbml': libsbml, 
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids,
        'check_species_exists': check_species_exists,
        'sinks': sinks,
        'sources': sources,
        'create_sink_for_species': create_sink_for_species,
        'create_source_for_species': create_source_for_species,
        'ensure_sigma_parameter': ensure_sigma_parameter,
        'ensure_rate_parameter': ensure_rate_parameter
    })
    
    def create_oligomer_13_16_reactions(ab_type):
        """Create all reactions for oligomers 13-16 of a specific AB type (40 or 42)
        
        Args:
            ab_type: AB type (40 or 42)
        """
        print(f"\nCreating oligomer 13-16 reactions for AB{ab_type}...")
        suffix = "forty" if ab_type == "40" else "fortytwo"
        
        # We'll create chain reactions between oligomers sizes 12-16
        for j in range(13, 17):
            # Determine current oligomer and previous/next oligomers
            current_oligomer = f"AB{ab_type}_Oligomer{j:02d}"
            prev_oligomer = f"AB{ab_type}_Oligomer{j-1:02d}"
            next_oligomer = f"AB{ab_type}_Oligomer{j+1:02d}" if j < 16 else None
            
            # Define plaque formation parameter
            plaque_formation_param = f"k_O{j}_Plaque_{suffix}"
            
            # List to hold all reaction blocks for this oligomer
            oligomer_reaction_blocks = []
            
            # Ensure sigma parameter exists
            sigma_param = ensure_sigma_parameter(j, ab_type)
            
            # Ensure necessary parameters exist
            if j > 12:
                ensure_rate_parameter(f"k_O{j-1:d}_O{j:d}_{suffix}")
                ensure_rate_parameter(f"k_O{j:d}_O{j-1:d}_{suffix}")
            
            if j < 16:
                ensure_rate_parameter(f"k_O{j:d}_O{j+1:d}_{suffix}")
                ensure_rate_parameter(f"k_O{j+1:d}_O{j:d}_{suffix}")
            
            # Reaction 1: Formation from smaller oligomer + monomer
            formation_reaction = f'''
# Formation reaction for AB{ab_type}_Oligomer{j:02d} from AB{ab_type}_Oligomer{j-1:02d} and AB{ab_type}_Monomer
reaction_oligomer{j}_formation = model.createReaction()
reaction_oligomer{j}_formation.setId(get_unique_reaction_id("{current_oligomer}_formation"))
reaction_oligomer{j}_formation.setReversible(False)

# Reactant 1: Previous oligomer
reactant_prev_oligomer = reaction_oligomer{j}_formation.createReactant()
reactant_prev_oligomer.setSpecies("{prev_oligomer}")
reactant_prev_oligomer.setStoichiometry(1.0)
reactant_prev_oligomer.setConstant(True)

# Reactant 2: Monomer
reactant_monomer = reaction_oligomer{j}_formation.createReactant()
reactant_monomer.setSpecies("AB{ab_type}_Monomer")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Product: Current oligomer
product_oligomer = reaction_oligomer{j}_formation.createProduct()
product_oligomer.setSpecies("{current_oligomer}")
product_oligomer.setStoichiometry(1.0)
product_oligomer.setConstant(True)

# Kinetic law: Update to include in formula
klaw_formation = reaction_oligomer{j}_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_O{j-1:d}_O{j:d}_{suffix} * {prev_oligomer} * AB{ab_type}_Monomer * VIS_brain")
klaw_formation.setMath(math_ast)
'''
            oligomer_reaction_blocks.append(formation_reaction)
            
            # Reaction 2: Dissociation to smaller oligomer + monomer
            dissociation_reaction = f'''
# Dissociation reaction for AB{ab_type}_Oligomer{j:02d} to AB{ab_type}_Oligomer{j-1:02d} and AB{ab_type}_Monomer
reaction_oligomer{j}_dissociation = model.createReaction()
reaction_oligomer{j}_dissociation.setId(get_unique_reaction_id("{current_oligomer}_dissociation"))
reaction_oligomer{j}_dissociation.setReversible(False)

# Reactant: Current oligomer
reactant_oligomer = reaction_oligomer{j}_dissociation.createReactant()
reactant_oligomer.setSpecies("{current_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

# Product 1: Previous oligomer
product_prev_oligomer = reaction_oligomer{j}_dissociation.createProduct()
product_prev_oligomer.setSpecies("{prev_oligomer}")
product_prev_oligomer.setStoichiometry(1.0)
product_prev_oligomer.setConstant(True)

# Product 2: Monomer
product_monomer = reaction_oligomer{j}_dissociation.createProduct()
product_monomer.setSpecies("AB{ab_type}_Monomer")
product_monomer.setStoichiometry(1.0)
product_monomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_dissociation = reaction_oligomer{j}_dissociation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_O{j:d}_O{j-1:d}_{suffix} * {current_oligomer} * VIS_brain")
klaw_dissociation.setMath(math_ast)
'''
            oligomer_reaction_blocks.append(dissociation_reaction)
            
            
            
            # Plaque-related reactions
            plaque_reactions = []
            
            # Reaction: Plaque-catalyzed formation of current oligomer
            plaque_catalyzed_formation = f'''
# Plaque-catalyzed formation of {current_oligomer} from {prev_oligomer} and monomer
reaction_plaque_oligomer{j}_formation = model.createReaction()
reaction_plaque_oligomer{j}_formation.setId(get_unique_reaction_id("AB{ab_type}_plaque_oligomer{j:02d}_formation"))
reaction_plaque_oligomer{j}_formation.setReversible(False)

# Reactants: {prev_oligomer} and monomer
reactant_oligomer = reaction_plaque_oligomer{j}_formation.createReactant()
reactant_oligomer.setSpecies("{prev_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

reactant_monomer = reaction_plaque_oligomer{j}_formation.createReactant()
reactant_monomer.setSpecies("AB{ab_type}_Monomer")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Modifier: AB{ab_type}_Plaque_unbound (catalyst, not consumed)
modifier_plaque = reaction_plaque_oligomer{j}_formation.createModifier()
modifier_plaque.setSpecies("AB{ab_type}_Plaque_unbound")

# Product: {current_oligomer}
product_oligomer = reaction_plaque_oligomer{j}_formation.createProduct()
product_oligomer.setSpecies("{current_oligomer}")
product_oligomer.setStoichiometry(1.0)
product_oligomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_plaque_formation = reaction_plaque_oligomer{j}_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_O{j-1:d}_O{j:d}_{suffix} * AB{ab_type}_Plaque_Driven_Monomer_Addition_Vmax * {prev_oligomer} * AB{ab_type}_Monomer * (AB{ab_type}_Plaque_unbound / (AB{ab_type}_Plaque_unbound + AB{ab_type}_Plaque_Driven_Monomer_Addition_EC50)) * VIS_brain")
klaw_plaque_formation.setMath(math_ast)
'''
            plaque_reactions.append(plaque_catalyzed_formation)
            
            
            # Reaction: Formation of plaque from oligomer and monomer
            plaque_formation = f'''
# Formation of plaque from {current_oligomer} and monomer
reaction_oligomer_to_plaque = model.createReaction()
reaction_oligomer_to_plaque.setId(get_unique_reaction_id("{current_oligomer}_to_plaque"))
reaction_oligomer_to_plaque.setReversible(False)

# Reactants: {current_oligomer} and monomer
reactant_oligomer = reaction_oligomer_to_plaque.createReactant()
reactant_oligomer.setSpecies("{current_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

reactant_monomer = reaction_oligomer_to_plaque.createReactant()
reactant_monomer.setSpecies("AB{ab_type}_Monomer")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Product: Plaque
product_plaque = reaction_oligomer_to_plaque.createProduct()
product_plaque.setSpecies("AB{ab_type}_Plaque_unbound")
product_plaque.setStoichiometry(1.0)
product_plaque.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_oligomer_to_plaque = reaction_oligomer_to_plaque.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"{plaque_formation_param} * {current_oligomer} * AB{ab_type}_Monomer * VIS_brain")
klaw_oligomer_to_plaque.setMath(math_ast)
'''
            plaque_reactions.append(plaque_formation)
            
            # Add plaque reactions to the list
            oligomer_reaction_blocks.extend(plaque_reactions)
            
            # Clearance and binding reactions
            clearance_reactions = []
            
            # Reaction: Antibody binding to current oligomer
            antibody_binding = f'''
# Antibody binding to {current_oligomer}
reaction_antibody_binding = model.createReaction()
reaction_antibody_binding.setId(get_unique_reaction_id("{current_oligomer}_antibody_binding"))
reaction_antibody_binding.setReversible(False)

# Reactants: {current_oligomer} and Antibody
reactant_oligomer = reaction_antibody_binding.createReactant()
reactant_oligomer.setSpecies("{current_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

reactant_antibody = reaction_antibody_binding.createReactant()
reactant_antibody.setSpecies("Ab_t")
reactant_antibody.setStoichiometry(1.0)
reactant_antibody.setConstant(True)

# Product: Bound oligomer
product_bound = reaction_antibody_binding.createProduct()
product_bound.setSpecies(f"{current_oligomer}_Antibody_bound")
product_bound.setStoichiometry(1.0)
product_bound.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_antibody_binding = reaction_antibody_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta1 * {current_oligomer} * Ab_t * VIS_brain")
klaw_antibody_binding.setMath(math_ast)
'''
            clearance_reactions.append(antibody_binding)
            
            # Reaction: Microglia clearance of current oligomer
            microglia_clearance = f'''
# Microglia clearance of {current_oligomer}
reaction_microglia_clearance = model.createReaction()
reaction_microglia_clearance.setId(get_unique_reaction_id("{current_oligomer}_microglia_clearance"))
reaction_microglia_clearance.setReversible(False)

# Reactant: {current_oligomer}
reactant_oligomer = reaction_microglia_clearance.createReactant()
reactant_oligomer.setSpecies("{current_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

# Modifiers: Add microglia species as modifiers since they influence but aren't consumed
modifier_cell_count = reaction_microglia_clearance.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction_microglia_clearance.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Add Sink as product
product = reaction_microglia_clearance.createProduct()
product.setSpecies(sinks["{current_oligomer}"])
product.setConstant(True)
product.setStoichiometry(1.0)

# Kinetic law: Update to include VIS_brain in formula
klaw_microglia_clearance = reaction_microglia_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"Microglia_Vmax_{suffix} * {current_oligomer} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
klaw_microglia_clearance.setMath(math_ast)
            '''
            clearance_reactions.append(microglia_clearance)
            
            # Add the clearance reactions to oligomer_reaction_blocks
            oligomer_reaction_blocks.extend(clearance_reactions)
            
            # Execute each reaction block for this oligomer size
            for block in oligomer_reaction_blocks:
                if block:  # Skip empty blocks
                    exec(block, global_vars)
    
    # Create species and reactions for AB40
    create_oligomer_13_16_reactions("40")

    # Create species and reactions for AB42
    create_oligomer_13_16_reactions("42")

    # Log the number of reactions created
    print(f"Created {len(created_reaction_ids)} unique reactions in the oligomer 13-16 model")

    return document

def create_oligomer_13_16_model_runner():
    """Create a function that loads parameters and creates a parameterized model
        
    Returns:
        Function that takes a parameter CSV path and returns an SBML document
    """
    def model_runner(csv_path, drug_type="gantenerumab"):
        # Load parameters with drug-specific parameter mapping
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        
        # Create model with mapped parameters
        document = create_oligomer_13_16_model(params, params_with_units)
        
        return document
    
    return model_runner

def verify_extrapolated_rates(params):
    """Verify that extrapolated rate constants are included and make sense
    For oligomers 13-16, we need rates for these transitions to be available
    """
    expected_rates = [
        'k_O12_O13_forty', 'k_O12_O13_fortytwo',  # Forward: 12 -> 13
        'k_O13_O12_forty', 'k_O13_O12_fortytwo',  # Backward: 13 -> 12
        'k_O13_O14_forty', 'k_O13_O14_fortytwo',  # Forward: 13 -> 14
        'k_O14_O13_forty', 'k_O14_O13_fortytwo',  # Backward: 14 -> 13
        'k_O14_O15_forty', 'k_O14_O15_fortytwo',  # Forward: 14 -> 15
        'k_O15_O14_forty', 'k_O15_O14_fortytwo',  # Backward: 15 -> 14
        'k_O15_O16_forty', 'k_O15_O16_fortytwo',  # Forward: 15 -> 16
        'k_O16_O15_forty', 'k_O16_O15_fortytwo',  # Backward: 16 -> 15
        'k_O16_O17_forty', 'k_O16_O17_fortytwo',  # Forward: 16 -> 17
        'k_O17_O16_forty', 'k_O17_O16_fortytwo',  # Backward: 17 -> 16
    ]
    
    missing_rates = [rate for rate in expected_rates if rate not in params]
    
    if missing_rates:
        print("WARNING: Missing expected rate constants:")
        for rate in missing_rates:
            print(f"  - {rate}")
    else:
        print("All expected rate constants are present")
        
    # Print the values of key rates to verify they make sense
    print("\nKey rate constants:")
    for rate in expected_rates[:8]:  # Just show a few
        if rate in params:
            print(f"  {rate}: {params[rate]:.6e}")

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
    sink.setInitialConcentration(0.0)  # Fixed concentration
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
    source.setInitialConcentration(1.0)  # Fixed concentration
    source.setConstant(True)
    source.setHasOnlySubstanceUnits(False)
    source.setBoundaryCondition(True)  # Boundary species
    source.setSubstanceUnits("nanomole_per_litre")
    return source_id

if __name__ == "__main__":
    # Test the model creation with both drug types
    runner = create_oligomer_13_16_model_runner()
    
    print("\n*** Testing with GANTENERUMAB ***")
    doc_gant = runner("parameters/PK_Geerts.csv", drug_type="gantenerumab")
    print("Gantenerumab model validation complete")
    
    print("\n*** Testing with LECANEMAB ***")
    doc_lec = runner("parameters/PK_Geerts.csv", drug_type="lecanemab")
    print("Lecanemab model validation complete") 