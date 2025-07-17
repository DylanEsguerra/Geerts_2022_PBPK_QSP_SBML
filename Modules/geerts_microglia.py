"""
Module for modeling microglia-mediated clearance of Aβ species.
This is a PBPK component that models the role of microglia in clearing Aβ species from the brain.
The module includes:
- Microglia activation and deactivation dynamics
- High and low activity states of microglia
- Microglia-mediated clearance of bound Aβ species
- Antibody-dependent microglial activation
- Microglia cell population dynamics
"""

# updated microglia equation 
import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to the CSV file
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        
    Returns:
        Tuple of (params, params_with_units)
    """
    print(f"Loading parameters from {csv_path}")
    df = pd.read_csv(csv_path)
    print("\nFirst few rows of parameter file:")
    print(df.head())
    
    params = dict(zip(df['name'], df['value']))
    params_with_units = dict(zip(df['name'], zip(df['value'], df['units'])))
    
    print("\nFirst few parameters loaded:")
    for i, (key, value) in enumerate(params.items()):
        if i < 5:  # Print first 5 parameters
            print(f"{key}: {value} ({params_with_units[key][1]})")
    
    return params, params_with_units

def create_microglia_model(params, params_with_units):
    """Create Geerts Microglia model using reactions instead of rate rules
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
    """
    print("\nCreating Geerts Microglia model with reactions...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    
    # Create model
    model = document.createModel()
    model.setId("Geerts_Microglia_Model")
    model.setName("Geerts Microglia Model")
    
    # Define units FIRST
    # Hour unit
    hour = model.createUnitDefinition()
    hour.setId("hour")
    hour_unit = hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setMultiplier(3600.0)  # seconds in an hour
    hour_unit.setScale(0)
    hour_unit.setExponent(1.0)

    # Add per_hour unit definition
    per_hour = model.createUnitDefinition()
    per_hour.setId("per_hour")
    hour_unit = per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # Add nanomole_per_hour unit definition
    nanomole_per_hour = model.createUnitDefinition()
    nanomole_per_hour.setId("nanomole_per_hour")
    mole_unit = nanomole_per_hour.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(1.0)
    mole_unit.setScale(-9)  # nano
    mole_unit.setMultiplier(1.0)
    hour_unit = nanomole_per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # Add nanomole unit definition
    nanomole = model.createUnitDefinition()
    nanomole.setId("nanomole")
    mole_unit = nanomole.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(1.0)
    mole_unit.setScale(-9)  # nano
    mole_unit.setMultiplier(1.0)

    # NOW set model time units
    model.setTimeUnits("hour")

    # Create common compartments
    compartments = {
        "comp_ISF_brain": params["VIS_brain"],  # ISF compartment for amyloid species
        "comp_microglia": params["V_microglia"],  # Microglia compartment for microglia species
    }

    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)
        comp.setUnits('litre')
    
    # Create dictionaries to store sinks and sources for each species
    sinks = {}
    sources = {}
    
    # Create microglia species in microglia compartment
    microglia_hi_fract = model.createSpecies()
    microglia_hi_fract.setId("Microglia_Hi_Fract")
    microglia_hi_fract.setName("Microglia High Activity Fraction")
    microglia_hi_fract.setCompartment("comp_microglia")
    microglia_hi_fract.setInitialConcentration(params["Microglia_Hi_Fract_0"])
    microglia_hi_fract.setConstant(False)
    microglia_hi_fract.setHasOnlySubstanceUnits(False)
    microglia_hi_fract.setBoundaryCondition(False)
    microglia_hi_fract.setSubstanceUnits("dimensionless")
    
    microglia_cell_count = model.createSpecies()
    microglia_cell_count.setId("Microglia_cell_count")
    microglia_cell_count.setName("Microglia Cell Count")
    microglia_cell_count.setCompartment("comp_microglia")
    microglia_cell_count.setInitialConcentration(params["Microglia_cell_count_0"])
    microglia_cell_count.setConstant(False)
    microglia_cell_count.setHasOnlySubstanceUnits(False)
    microglia_cell_count.setBoundaryCondition(False)
    microglia_cell_count.setSubstanceUnits("dimensionless")

    # Add antibody-bound species for oligomers and fibrils
    bound_species = []
    
    # Add bound oligomer species for AB40 and AB42 (sizes 2-16)
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            bound_species.append((
                f"{prefix}_Oligomer{i:02d}_Antibody_bound",
                "comp_ISF_brain",  # Changed to ISF compartment
                0.0
            ))
    
    # Add bound fibril species for AB40 and AB42 (sizes 17-24)
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            bound_species.append((
                f"{prefix}_Fibril{i:02d}_Antibody_bound",
                "comp_ISF_brain",  # Changed to ISF compartment
                0.0
            ))
    
    # Add bound plaque species
    bound_species.extend([
        ("AB40_Plaque_Antibody_bound", "comp_ISF_brain", 0.0),
        ("AB42_Plaque_Antibody_bound", "comp_ISF_brain", 0.0)
    ])
    
    # Pre-create sinks for all bound species
    for species_id, _, _ in bound_species:
        sinks[species_id] = create_sink_for_species(species_id, model)

    # Create bound species
    for species_id, compartment_id, initial_value in bound_species:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment(compartment_id)
        species.setInitialConcentration(initial_value)
        species.setConstant(False)
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setSubstanceUnits("nanomole_per_litre")
    
    # Create microglia clearance reactions for all bound species
    for species_id, _, _ in bound_species:
        # Determine AB type from species name
        ab_type = "40" if "AB40" in species_id else "42"
        suffix = "forty" if "AB40" in species_id else "fortytwo"
        
        # Create reaction
        reaction = model.createReaction()
        reaction.setId(f"{species_id}_microglia_clearance")
        reaction.setReversible(False)
        
        # Add reactant (bound species)
        reactant = reaction.createReactant()
        reactant.setSpecies(species_id)
        reactant.setStoichiometry(1.0)
        reactant.setConstant(True)
        
        # Add modifiers (microglia species)
        modifier_cell_count = reaction.createModifier()
        modifier_cell_count.setSpecies("Microglia_cell_count")
        
        modifier_hi_fract = reaction.createModifier()
        modifier_hi_fract.setSpecies("Microglia_Hi_Fract")
        
        # Create kinetic law - different formula for plaque vs other species
        kinetic_law = reaction.createKineticLaw()
        
        # Check if this is a plaque species
        if "Plaque" in species_id:
            # Use original formula for plaque scaled by 0.5
            math_ast = libsbml.parseL3Formula(
                f"0.5 * {species_id} * Microglia_cell_count * "
                f"(Microglia_Hi_Fract * Microglia_CL_high_mAb + "
                f"(1 - Microglia_Hi_Fract) * Microglia_CL_low_mAb) * VIS_brain"
            )
        else:
            #'''
            # Use new formula for oligomers and fibrils
            math_ast = libsbml.parseL3Formula(
                f"{species_id} * Microglia_cell_count * "
                f"(Microglia_Hi_Fract * Microglia_Hi_Lo_ratio * (Microglia_Vmax_{suffix}/(Microglia_EC50_{suffix} + {species_id})) + "
                f"(1 - Microglia_Hi_Fract) * (Microglia_Vmax_{suffix}/(Microglia_EC50_{suffix} + {species_id}))) * VIS_brain"
            )
            '''
            # original formula
            math_ast = libsbml.parseL3Formula(
                f"{species_id} * Microglia_cell_count * "
                f"(Microglia_Hi_Fract * Microglia_CL_high_mAb + "
                f"(1 - Microglia_Hi_Fract) * Microglia_CL_low_mAb) * VIS_brain"
            )
            '''
        
        kinetic_law.setMath(math_ast)
        
        # Add species-specific sink as product
        product = reaction.createProduct()
        product.setSpecies(sinks[species_id])
        product.setStoichiometry(1.0)
        product.setConstant(True)
    
    # Add parameters
    microglia_params = [
        ("gamma_fr", params["gamma_fr"]),
        ("gamma_MG", params["gamma_MG"]),
        ("Microglia_Hi_fract_max", params["Microglia_Hi_fract_max"]),
        ("Microglia_Hi_fract_base", params["Microglia_Hi_fract_base"]),
        ("Microglia_CL_high_mAb", params["Microglia_CL_high_mAb"]),
        ("Microglia_CL_low_mAb", params["Microglia_CL_low_mAb"]),
        ("Microglia_cells_max", params["Microglia_cells_max"]),
        ("Microglia_cells_max", params["Microglia_cells_max"]),
        ("Microglia_baseline", params["Microglia_baseline"]),
        ("SmaxUp", params["SmaxUp"]),
        ("EC50_Up", params["EC50_Up"]),
        ("SmaxPro", params["SmaxPro"]),
        ("EC50_pro", params["EC50_pro"]),
        # New parameters for updated clearance formula
        ("Microglia_Hi_Lo_ratio", params["Microglia_Hi_Lo_ratio"]),
        ("Microglia_Vmax_forty", params["Microglia_Vmax_forty"]),
        ("Microglia_Vmax_fortytwo", params["Microglia_Vmax_fortytwo"]),
        ("Microglia_EC50_forty", params["Microglia_EC50_forty"]),
        ("Microglia_EC50_fortytwo", params["Microglia_EC50_fortytwo"]),
    ]
    
    for param_id, value in microglia_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Set appropriate units
        if param_id.startswith('gamma'):
            param.setUnits("per_hour")
        elif param_id.startswith('EC50'):
            param.setUnits("nanomole_per_litre")
        elif param_id.startswith('Microglia_Vmax'):
            param.setUnits("nanomole_per_hour")
        elif param_id.startswith('Microglia_EC50'):
            param.setUnits("nanomole_per_litre")
        else:
            param.setUnits("dimensionless")
    
    # Change Anti_ABeta_bound_sum from a parameter to an assignment rule
    # First, create it as a parameter that will be assigned by the rule
    param = model.createParameter()
    param.setId("Anti_ABeta_bound_sum")
    param.setValue(0.0)  # Initial value doesn't matter since it will be assigned
    param.setConstant(False)  # Important: must be false to allow assignment
    param.setUnits("nanomole_per_litre")

    # Create the assignment rule to sum all bound species
    rule = model.createAssignmentRule()
    rule.setVariable("Anti_ABeta_bound_sum")

    # Build the sum expression for all bound species
    sum_terms = []
    # Add oligomer bound species
    for i in range(2, 17):
        for prefix in ['AB40', 'AB42']:
            sum_terms.append(f"{prefix}_Oligomer{i:02d}_Antibody_bound")

    # Add fibril bound species
    for i in range(17, 25):
        for prefix in ['AB40', 'AB42']:
            sum_terms.append(f"{prefix}_Fibril{i:02d}_Antibody_bound")

    # Add plaque bound species
    sum_terms.extend([
        "AB40_Plaque_Antibody_bound",
        "AB42_Plaque_Antibody_bound"
    ])

    # Create the math expression for the sum
    sum_expression = " + ".join(sum_terms)
    math_ast = libsbml.parseL3Formula(sum_expression)
    rule.setMath(math_ast)

    # Change rule names to be more specific and unique
    # Rate rule for Microglia Hi Fraction (Equation 69)
    hi_fract_rule = model.createRateRule()
    hi_fract_rule.setId("microglia_hi_fract_rate_rule")  # Add unique identifier
    hi_fract_rule.setVariable("Microglia_Hi_Fract")
    math_ast = libsbml.parseL3Formula(
        "(1 + SmaxUp * Anti_ABeta_bound_sum / (EC50_Up + Anti_ABeta_bound_sum)) * "
        "gamma_fr * Microglia_Hi_Fract * (Microglia_Hi_fract_max - Microglia_Hi_Fract) / "
        "(Microglia_Hi_fract_max - Microglia_Hi_fract_base)  - "
        "gamma_fr * Microglia_Hi_Fract"
    )
    hi_fract_rule.setMath(math_ast)

    # Rate rule for Microglia Cell Count (Equation 68)
    cell_count_rule = model.createRateRule()
    cell_count_rule.setId("microglia_cell_count_rate_rule")  # Add unique identifier
    cell_count_rule.setVariable("Microglia_cell_count")
    math_ast = libsbml.parseL3Formula(
        "(1 + SmaxPro * Anti_ABeta_bound_sum / (EC50_pro + Anti_ABeta_bound_sum)) * "
        "gamma_MG * Microglia_cells_max / (Microglia_cells_max - Microglia_baseline) * "
        "Microglia_cell_count * (Microglia_cells_max - Microglia_cell_count) / Microglia_cells_max - "
        "gamma_MG * Microglia_cell_count"
    )
    cell_count_rule.setMath(math_ast)

    # Add Microglia_cell_count_0 as a parameter if not already present
    param = model.createParameter()
    param.setId("Microglia_cell_count_0")
    param.setValue(params["Microglia_cell_count_0"])
    param.setConstant(True)
    param.setUnits("dimensionless")

    param = model.createParameter()
    param.setId("Microglia_Hi_Fract_0")
    param.setValue(params["Microglia_Hi_Fract_0"])
    param.setConstant(True)
    param.setUnits("dimensionless")
    
    return document

def write_model_to_file(document, output_file):
    """Write the SBML document to a file"""
    libsbml.writeSBMLToFile(document, output_file)
    print(f"Model written to {output_file}")

def run_microglia_model(csv_path, output_file="Geerts_Microglia_Model.xml", drug_type="gantenerumab"):
    """Run the microglia model with parameters from CSV file"""
    params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
    document = create_microglia_model(params, params_with_units)
    write_model_to_file(document, output_file)
    return document

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
    # Example usage
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "parameters/PK_Geerts.csv"
    
    run_microglia_model(csv_path)
