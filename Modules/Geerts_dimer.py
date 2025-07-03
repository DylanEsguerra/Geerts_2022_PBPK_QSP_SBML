"""
Module for modeling Aβ dimer formation and dynamics.
This is a QSP component that models the formation and behavior of Aβ dimers (2-mers) from monomers.
The module includes:
- Dimer formation from monomers
- Dimer dissociation back to monomers
- Plaque-catalyzed dimer formation
- Antibody binding to dimers
"""

# deleted trimer formation from current and monomer plaque catalyzed
# deleted trimer dissociation
import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} ===")
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
    print("\nAntibody binding parameters:")
    for param in ['fta0', 'fta1', 'fta2', 'fta3']:
        if param in params:
            print(f"  {param}: {params[param]}")
        else:
            print(f"  {param}: NOT FOUND")
    
    return params, params_with_units

def create_dimer_model(params, params_with_units):
    """Create a parameterized SBML model for the Geerts model
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
        
    Returns:
        SBML document
    """
    print("\nCreating Geerts dimer model...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_Dimer_Model")
    
    # Add a global dictionary to track used reaction IDs across all AB types
    created_reaction_ids = {}
    
    # Helper function to ensure unique reaction IDs
    def get_unique_reaction_id(base_id):
        if base_id not in created_reaction_ids:
            created_reaction_ids[base_id] = 0
            return base_id
        else:
            created_reaction_ids[base_id] += 1
            return f"{base_id}_{created_reaction_ids[base_id]}"
    
    # Add get_unique_reaction_id to global variables used in exec
    global_vars = globals().copy()
    global_vars.update({
        'model': model, 
        'libsbml': libsbml, 
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids
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

    # Create aggregation-specific parameters with updated parameter names
    aggregation_params = [
        # Forward reaction rates for oligomerization - UPDATED NAMES
        ("k_M_O2_forty", params["k_M_O2_forty"]),  # AB40 monomer-monomer binding
        ("k_M_O2_fortytwo", params["k_M_O2_fortytwo"]),  # AB42 monomer-monomer binding
        ("k_O2_O3_forty", params["k_O2_O3_forty"]),  # AB40 dimer to trimer
        ("k_O2_O3_fortytwo", params["k_O2_O3_fortytwo"]),  # AB42 dimer to trimer
        
        # Backward reaction rates for oligomerization - UPDATED NAMES
        ("k_O2_M_forty", params["k_O2_M_forty"]),  # AB40 dimer dissociation
        ("k_O2_M_fortytwo", params["k_O2_M_fortytwo"]),  # AB42 dimer dissociation
        ("k_O3_O2_forty", params["k_O3_O2_forty"]),  # AB40 trimer dissociation
        ("k_O3_O2_fortytwo", params["k_O3_O2_fortytwo"]),  # AB42 trimer dissociation

        # Plaque-induced oligomerization parameters
        ("AB40_Plaque_Vmax", params["AB40_Plaque_Driven_Monomer_Addition_Vmax"]),  # Effect of plaque on AB40 oligomerization
        ("AB42_Plaque_Vmax", params["AB42_Plaque_Driven_Monomer_Addition_Vmax"]),  # Effect of plaque on AB42 oligomerization
        ("AB40_Plaque_EC50", params["AB40_Plaque_Driven_Monomer_Addition_EC50"]),  # Half-saturation for plaque effect on AB40
        ("AB42_Plaque_EC50", params["AB42_Plaque_Driven_Monomer_Addition_EC50"]),  # Half-saturation for plaque effect on AB42
        
        # Antibody binding parameters
        ("fta1", params["fta1"]),  # Antibody binding to AB40 oligomer
        
        # Transport and clearance parameters
        ("sigma_ISF_AB40_oligomer02", params["sigma_ISF_ABeta40_oligomer02"]),  # Reflection coefficient for AB40 dimer
        ("sigma_ISF_AB42_oligomer02", params["sigma_ISF_ABeta42_oligomer02"]),  # Reflection coefficient for AB42 dimer
        ("VIS_brain", params["VIS_brain"]),  # ISF volume
        
    ]

    # Add the parameters to the model
    print("\nCreating aggregation parameters...")
    for param_id, value in aggregation_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Assign appropriate units based on parameter type
        if param_id.startswith('k_M_'):
            param.setUnits("per_nanomole_per_hour")
        elif param_id.startswith('k_O'):
            if "_O" in param_id.split("k_")[1]:
                param.setUnits("per_nanomole_per_hour")
            else:
                param.setUnits("per_hour")
        elif param_id.startswith('AB40_Plaque_Driven_Monomer_Addition_'):
            param.setUnits("dimensionless")  # Enhancement factor
        elif param_id.startswith('sigma_'):
            param.setUnits("dimensionless")  # Reflection coefficient
        elif param_id.startswith('Q_'):
            param.setUnits("litre_per_hour")  # Flow rate
        elif param_id.startswith('vol_'):
            param.setUnits("litre")  # Volume
        elif param_id.startswith('fta'):
            param.setUnits("per_nanomole_per_hour")
        elif param_id.startswith('K_O'):
            param.setUnits("per_nanomole_per_hour")
        else:
            param.setUnits("dimensionless")


    # Create common compartments
    compartments = {
        "comp_ISF_brain": params["VIS_brain"],
    }

    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)
        comp.setUnits('litre')

    
    # Add the fibril species to the species list
    species = [
        # Monomer species 
        ("AB42_Monomer", "comp_ISF_brain", 0.0),
        ("AB40_Monomer", "comp_ISF_brain", 0.0),
        
        # Dimer species
        ("AB40_Oligomer02", "comp_ISF_brain", 0.0),
        ("AB42_Oligomer02", "comp_ISF_brain", 0.0),
        
        # Trimer species
        ("AB40_Oligomer03", "comp_ISF_brain", 0.0),
        ("AB42_Oligomer03", "comp_ISF_brain", 0.0),
        
        # Plaque species
        ("AB40_Plaque_unbound", "comp_ISF_brain", 0.0),
        ("AB42_Plaque_unbound", "comp_ISF_brain", 0.0),
        
        # Antibody species
        ("Ab_t", "comp_ISF_brain", 0.0),
        
        # Bound species
        ("AB40_Oligomer02_Antibody_bound", "comp_ISF_brain", 0.0),
        ("AB42_Oligomer02_Antibody_bound", "comp_ISF_brain", 0.0),
    
    ]
    

    

             # Create variable species
    for species_id, compartment_id, initial_value in species:
        spec = model.createSpecies()
        spec.setId(species_id)
        spec.setCompartment(compartment_id)
        spec.setInitialConcentration(initial_value)
        spec.setSubstanceUnits("nanomole_per_litre")
        spec.setHasOnlySubstanceUnits(False)
        spec.setBoundaryCondition(False)
        spec.setConstant(False)


    def create_dimer_reactions(ab_type):
        """Create all species and reactions for dimerization of a specific AB type (40 or 42)."""

        # Different parameter name suffix based on AB type
        suffix = "forty" if ab_type == "40" else "fortytwo"
        
        print(f'Creating dimer reactions for AB{ab_type}...')
        
        # Break down the equation into individual reactions
        reaction_blocks = [
            # Reaction 1: Dimer formation
            f'''
# 1. Formation of AB{ab_type} dimer from two monomers
reaction_dimer_formation = model.createReaction()
base_id = f"AB{ab_type}_dimer_formation"
reaction_dimer_formation.setId(get_unique_reaction_id(base_id))
reaction_dimer_formation.setReversible(False)

# Reactants: Two AB{ab_type} monomers
reactant_monomer1 = reaction_dimer_formation.createReactant()
reactant_monomer1.setSpecies("AB{ab_type}_Monomer")
reactant_monomer1.setStoichiometry(2.0)
reactant_monomer1.setConstant(True)

# Product: AB{ab_type} dimer
product_dimer = reaction_dimer_formation.createProduct()
product_dimer.setSpecies("AB{ab_type}_Oligomer02")
product_dimer.setStoichiometry(1.0)
product_dimer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_dimer_formation = reaction_dimer_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_M_O2_{suffix} * AB{ab_type}_Monomer * AB{ab_type}_Monomer * VIS_brain")
klaw_dimer_formation.setMath(math_ast)
            ''',
            
            # Reaction 2: Dimer dissociation
            f'''
# 2. Dissociation of AB{ab_type} dimer back to monomers
reaction_dimer_dissociation = model.createReaction()
base_id = f"AB{ab_type}_dimer_dissociation"
reaction_dimer_dissociation.setId(get_unique_reaction_id(base_id))
reaction_dimer_dissociation.setReversible(False)

# Reactant: AB{ab_type} dimer
reactant_dimer = reaction_dimer_dissociation.createReactant()
reactant_dimer.setSpecies("AB{ab_type}_Oligomer02")
reactant_dimer.setStoichiometry(1.0)
reactant_dimer.setConstant(True)

# Products: Two AB{ab_type} monomers
product_monomer = reaction_dimer_dissociation.createProduct()
product_monomer.setSpecies("AB{ab_type}_Monomer")
product_monomer.setStoichiometry(2.0)
product_monomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_dimer_dissociation = reaction_dimer_dissociation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_O2_M_{suffix} * AB{ab_type}_Oligomer02 * VIS_brain")
klaw_dimer_dissociation.setMath(math_ast)
            ''',
            
            
            # Reaction 5: Plaque-catalyzed dimer formation
            f'''
# 5. Plaque-catalyzed formation of AB{ab_type} dimer from monomers
reaction_plaque_dimer_formation = model.createReaction()
base_id = f"AB{ab_type}_Oligomer02_plaque_catalyzed_formation"
reaction_plaque_dimer_formation.setId(get_unique_reaction_id(base_id))
reaction_plaque_dimer_formation.setReversible(False)

# Reactants: Two AB{ab_type} monomers
reactant_monomer1 = reaction_plaque_dimer_formation.createReactant()
reactant_monomer1.setSpecies("AB{ab_type}_Monomer")
reactant_monomer1.setStoichiometry(2.0)
reactant_monomer1.setConstant(True)

# Modifier: AB{ab_type} Plaque (catalyst, not consumed)
modifier_plaque = reaction_plaque_dimer_formation.createModifier()
modifier_plaque.setSpecies("AB{ab_type}_Plaque_unbound")

# Product: AB{ab_type} dimer
product_dimer = reaction_plaque_dimer_formation.createProduct()
product_dimer.setSpecies("AB{ab_type}_Oligomer02")
product_dimer.setStoichiometry(1.0)
product_dimer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_plaque_dimer_formation = reaction_plaque_dimer_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"k_M_O2_{suffix} * AB{ab_type}_Plaque_Vmax * AB{ab_type}_Monomer * AB{ab_type}_Monomer * (AB{ab_type}_Plaque_unbound / (AB{ab_type}_Plaque_unbound + AB{ab_type}_Plaque_EC50)) * VIS_brain")
klaw_plaque_dimer_formation.setMath(math_ast)
            ''',
            
            
            
            # Reaction 7: Antibody binding
            f'''
# 7. Antibody binding to AB{ab_type} dimer
reaction_antibody_dimer_binding = model.createReaction()
base_id = f"AB{ab_type}_antibody_dimer_binding"
reaction_antibody_dimer_binding.setId(get_unique_reaction_id(base_id))
reaction_antibody_dimer_binding.setReversible(False)

# Reactants: AB{ab_type} dimer and Antibody
reactant_dimer = reaction_antibody_dimer_binding.createReactant()
reactant_dimer.setSpecies("AB{ab_type}_Oligomer02")
reactant_dimer.setStoichiometry(1.0)
reactant_dimer.setConstant(True)

reactant_antibody = reaction_antibody_dimer_binding.createReactant()
reactant_antibody.setSpecies("Ab_t")
reactant_antibody.setStoichiometry(1.0)
reactant_antibody.setConstant(True)

# Product: Bound dimer
product_bound = reaction_antibody_dimer_binding.createProduct()
product_bound.setSpecies("AB{ab_type}_Oligomer02_Antibody_bound")
product_bound.setStoichiometry(1.0)
product_bound.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_antibody_dimer_binding = reaction_antibody_dimer_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta1 * AB{ab_type}_Oligomer02 * Ab_t * VIS_brain")
klaw_antibody_dimer_binding.setMath(math_ast)
            '''
        ]

        # Execute each reaction block with access to the model variable
        for formatted_block in reaction_blocks:
            # Pass the global vars dictionary to make all required variables available in the executed code
            exec(formatted_block, global_vars)

    
    # Create species and reactions for AB40
    create_dimer_reactions("40")

    # Create species and reactions for AB42
    create_dimer_reactions("42")
    
    # Log the number of reactions created
    print(f"Created {len(created_reaction_ids)} unique reactions in the dimer model")
    
    return document

def create_dimer_model_runner():
    """Create a function that loads parameters and creates a parameterized model
        
    Returns:
        Function that takes a parameter CSV path and returns an SBML document
    """
    def model_runner(csv_path, drug_type="gantenerumab"):
        # Load parameters with drug-specific parameter mapping
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        
        # Create model - now using mapped parameters
        document = create_dimer_model(params, params_with_units)
        
        return document
    
    return model_runner

if __name__ == "__main__":
    # Test the model creation
    runner = create_dimer_model_runner()
    doc = runner("parameters/PK_Geerts.csv")
    print("Model validation complete") 