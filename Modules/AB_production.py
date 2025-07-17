"""
Module for modeling amyloid beta (Aβ) production in the brain.
This is a QSP component that models the production of Aβ monomers (both 40 and 42 amino acid variants)
from amyloid precursor protein (APP) through enzymatic cleavage.
The module includes:
- APP production and processing
- C99 fragment formation
- Aβ40 and Aβ42 monomer production
- IDE-mediated clearance of Aβ monomers

Just removed negative sign from clearance 
"""

import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path):
    """Load parameters from CSV file into dictionary with values and units"""
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

def create_geerts_monomer_production_model(params, params_with_units):
    """Create Geerts Monomer Production model using reactions instead of rate rules
    
    Args:
        params: Dictionary of parameters for the model (already properly scaled)
        params_with_units: Dictionary of parameters with units
    """
    print("\nCreating Geerts Monomer Production model with reactions...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    
    # Create model
    model = document.createModel()
    model.setId("Geerts_Monomer_Production_Model")
    model.setName("Geerts Monomer Production Model")
    
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

    # Get brain ISF volume
    VIS_brain = params["VIS_brain"]
    param = model.createParameter()
    param.setId("VIS_brain")
    param.setValue(VIS_brain)
    param.setConstant(True)
    
    # Create all required compartments
    compartments = {
        "comp_ISF_brain": VIS_brain
    }
    
    # Create compartments
    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setConstant(True)
        comp.setSize(size)
        comp.setUnits("litre")
    
    # Create species with more balanced initial conditions
    # APP - start with a reasonable concentration
    app = model.createSpecies()
    app.setId("APP")
    app.setName("Amyloid Precursor Protein")
    app.setCompartment("comp_ISF_brain")
    app.setInitialConcentration(0.0)
    app.setConstant(False)
    app.setHasOnlySubstanceUnits(False)
    app.setBoundaryCondition(False)
    app.setSubstanceUnits("nanomole_per_litre")
    
    # C99
    c99 = model.createSpecies()
    c99.setId("C99")
    c99.setName("C99 Fragment")
    c99.setCompartment("comp_ISF_brain")
    c99.setInitialConcentration(0.0)
    c99.setConstant(False)
    c99.setHasOnlySubstanceUnits(False)
    c99.setBoundaryCondition(False)
    c99.setSubstanceUnits("nanomole_per_litre")
    
    # AB40_monomer - match with existing AB40 in the model
    ab40 = model.createSpecies()
    ab40.setId("AB40_Monomer")
    ab40.setName("Amyloid Beta 40 Monomer")
    ab40.setCompartment("comp_ISF_brain")
    ab40.setInitialConcentration(0.0)  # Match with existing model
    ab40.setConstant(False)
    ab40.setHasOnlySubstanceUnits(False)
    ab40.setBoundaryCondition(False)
    ab40.setSubstanceUnits("nanomole_per_litre")
    
    # AB42_monomer - match with existing AB42 in the model
    ab42 = model.createSpecies()
    ab42.setId("AB42_Monomer")
    ab42.setName("Amyloid Beta 42 Monomer")
    ab42.setCompartment("comp_ISF_brain")
    ab42.setInitialConcentration(0.0)  # Match with existing model
    ab42.setConstant(False)
    ab42.setHasOnlySubstanceUnits(False)
    ab42.setBoundaryCondition(False)
    ab42.setSubstanceUnits("nanomole_per_litre")

    # Create dictionaries to store sinks and sources for each species
    sinks = {}
    sources = {}
    
    # Pre-create sinks and sources for required species
    sinks["C99"] = create_sink_for_species("C99", model)
    sinks["AB40_Monomer"] = create_sink_for_species("AB40_Monomer", model)
    sinks["AB42_Monomer"] = create_sink_for_species("AB42_Monomer", model)
    sources["APP"] = create_source_for_species("APP", model)

    # Add parameters
    common_params = [
        ("k_C99", params["k_C99"]),
        ("k_in_AB40", params["k_in_Ab40"]),
        ("k_in_AB42", params["k_in_Ab42"]),
        ("v_C99_clearance", params["v_C99"]),
        ("k_APP_production", params["k_APP_production"]),  
        ("k_out_AB40", params["k_out_Ab40"]),
        ("k_out_AB42", params["k_out_Ab42"]),
        # IDE clearance parameters
        ("IDE_conc", params["IDE_conc"]),
        ("AB40_IDE_Kcat_lin", params["AB40_IDE_Kcat_lin"]),
        ("AB40_IDE_Kcat_exp", params["AB40_IDE_Kcat_exp"]),
        ("AB40_IDE_Hill", params["AB40_IDE_Hill"]),
        ("AB40_IDE_IC50", params["AB40_IDE_IC50"]),
        ("AB42_IDE_Kcat_lin", params["AB42_IDE_Kcat_lin"]),
        ("AB42_IDE_Kcat_exp", params["AB42_IDE_Kcat_exp"]),
        ("AB42_IDE_Hill", params["AB42_IDE_Hill"]),
        ("AB42_IDE_IC50", params["AB42_IDE_IC50"]),
        ("Unit_removal_1", params["Unit_removal_1"]),
        # Exponential decay rates from data (converting from /year to /hour)
        ("exp_decline_rate_IDE_forty", params["exp_decline_rate_IDE_forty"]),  
        ("exp_decline_rate_IDE_fortytwo", params["exp_decline_rate_IDE_fortytwo"]),
        ("lin_decline_rate_IDE_forty", params["lin_decline_rate_IDE_forty"]),
        ("lin_decline_rate_IDE_fortytwo", params["lin_decline_rate_IDE_fortytwo"]),
        ("age_init", 0.0),                    # Initial age in years (at t=0)
        # Add reflection coefficients for monomers
        ("sigma_ISF_ABeta40_monomer", params["sigma_ISF_ABeta40_monomer"]),
        ("sigma_ISF_ABeta42_monomer", params["sigma_ISF_ABeta42_monomer"]),
    ]
    
    # No scaling logic here - parameters should already be properly scaled
    print(f"Using k_APP_production value: {params['k_APP_production']:.6e}")
    
    for param_id, value in common_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
   
        # More specific unit assignments based on parameter type
        if param_id.startswith('V'):  # Volumes
            param.setUnits("litre")
        elif param_id.startswith('k') or param_id.startswith('v'):
            param.setUnits("per_hour") # check this
        else:
            param.setUnits("dimensionless")
    
    # 1. APP to C99 conversion: APP -> C99
    reaction1 = model.createReaction()
    reaction1.setId("APP_to_C99")
    reaction1.setName("Conversion of APP to C99")
    reaction1.setReversible(False)
    
    # Reactant: APP
    reactant = reaction1.createReactant()
    reactant.setSpecies("APP")
    reactant.setConstant(False)
    reactant.setStoichiometry(1.0)
    
    # Product: C99
    product = reaction1.createProduct()
    product.setSpecies("C99")
    product.setConstant(False)
    product.setStoichiometry(1.0)
    
    # Kinetic law: kC99 * APP
    kinetic_law = reaction1.createKineticLaw()
    math_ast = libsbml.parseL3Formula("k_C99 * APP * VIS_brain")
    kinetic_law.setMath(math_ast)
    
    # 2. APP production from 
    reaction2 = model.createReaction()
    reaction2.setId("APP_production")
    reaction2.setName("Production of APP")
    reaction2.setReversible(False)
    
    # Add Source as reactant
    reactant = reaction2.createReactant()
    reactant.setSpecies(sources["APP"])
    reactant.setConstant(True)
    reactant.setStoichiometry(1.0)
    
    # Product: APP
    product = reaction2.createProduct()
    product.setSpecies("APP")
    product.setConstant(False)
    product.setStoichiometry(1.0)
    
    # Kinetic law: kAPP_Potter_1st_order, could add plasma leucine to this
    kinetic_law = reaction2.createKineticLaw()
    math_ast = libsbml.parseL3Formula("k_APP_production * VIS_brain")
    kinetic_law.setMath(math_ast)
    
    # 3. C99 to AB40_monomer conversion
    reaction3 = model.createReaction()
    reaction3.setId("C99_to_AB40")
    reaction3.setName("Production of AB40 monomer from C99")
    reaction3.setReversible(False)
    
    # Reactant: C99 
    reactant = reaction3.createReactant()
    reactant.setSpecies("C99")
    reactant.setConstant(False)
    reactant.setStoichiometry(1.0)
    
    # Product: AB40_monomer
    product = reaction3.createProduct()
    product.setSpecies("AB40_Monomer")
    product.setConstant(False)
    product.setStoichiometry(1.0)
    
    # Kinetic law: k_40_C99 * C99
    kinetic_law = reaction3.createKineticLaw()
    math_ast = libsbml.parseL3Formula("k_in_AB40 * C99 * VIS_brain")
    kinetic_law.setMath(math_ast)
    
    # 4. C99 to AB42_monomer conversion
    reaction4 = model.createReaction()
    reaction4.setId("C99_to_AB42")
    reaction4.setName("Production of AB42 monomer from C99")
    reaction4.setReversible(False)
    
    # Reactant: C99 
    reactant = reaction4.createReactant()
    reactant.setSpecies("C99")
    reactant.setConstant(False)
    reactant.setStoichiometry(1.0)
    
    # Product: AB42_monomer
    product = reaction4.createProduct()
    product.setSpecies("AB42_Monomer")
    product.setConstant(False)
    product.setStoichiometry(1.0)
    
    # Kinetic law: k_42_C99 * C99
    kinetic_law = reaction4.createKineticLaw()
    math_ast = libsbml.parseL3Formula("k_in_AB42 * C99 * VIS_brain")
    kinetic_law.setMath(math_ast)
    
    # 5. C99 clearance
    reaction5 = model.createReaction()
    reaction5.setId("C99_clearance")
    reaction5.setName("Clearance of C99")
    reaction5.setReversible(False)
    
    # Reactant: C99
    reactant = reaction5.createReactant()
    reactant.setSpecies("C99")
    reactant.setConstant(False)
    reactant.setStoichiometry(1.0)
    
    # Add Sink as product
    product = reaction5.createProduct()
    product.setSpecies(sinks["C99"])
    product.setConstant(True)
    product.setStoichiometry(1.0)
    
    # Kinetic law: vC99 * C99
    kinetic_law = reaction5.createKineticLaw()
    math_ast = libsbml.parseL3Formula("v_C99_clearance * C99 * VIS_brain")
    kinetic_law.setMath(math_ast)

    # Create rules for IDE efficiency calculation (time/age dependent)
    # Create IDE efficiency parameters 
    param40 = model.createParameter()
    param40.setId("CL_AB40_IDE")
    param40.setValue(params["AB40_IDE_Kcat_exp"])  # Initial value
    #param40.setValue(params["AB40_IDE_Kcat_lin"])
    param40.setConstant(False)
    
    param42 = model.createParameter()
    param42.setId("CL_AB42_IDE")
    param42.setValue(params["AB42_IDE_Kcat_exp"])  # Initial value
    #param42.setValue(params["AB42_IDE_Kcat_lin"])
    param42.setConstant(False)

    # AB40 IDE efficiency rule
    
    ide_rule40 = model.createRateRule()
    ide_rule40.setId("CL_AB40_IDE_rule")
    ide_rule40.setVariable("CL_AB40_IDE")
    ide_rule40.setMath(libsbml.parseL3Formula("-exp_decline_rate_IDE_forty*CL_AB40_IDE"))
    #ide_rule40.setMath(libsbml.parseL3Formula("AB40_IDE_Kcat_lin"))
    # AB42 IDE efficiency rule
    ide_rule42 = model.createRateRule()
    ide_rule42.setId("CL_AB42_IDE_rule")
    ide_rule42.setVariable("CL_AB42_IDE")
    ide_rule42.setMath(libsbml.parseL3Formula("-exp_decline_rate_IDE_fortytwo*CL_AB42_IDE"))
    #ide_rule42.setMath(libsbml.parseL3Formula("AB42_IDE_Kcat_lin"))
    
    # Add AB40 clearance reaction
    
    reaction6 = model.createReaction()
    reaction6.setId("IDE_AB40_clearance")
    reaction6.setName("Clearance of AB40 monomer")
    reaction6.setReversible(False)
    
    # Reactant: AB40_monomer
    reactant = reaction6.createReactant()
    reactant.setSpecies("AB40_Monomer")
    reactant.setConstant(False)
    reactant.setStoichiometry(1.0)
    
    # Add Sink as product
    product = reaction6.createProduct()
    product.setSpecies(sinks["AB40_Monomer"])
    product.setConstant(True)
    product.setStoichiometry(1.0)

    # Kinetic law for IDE clearance of AB40
    kinetic_law = reaction6.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(IDE_conc * CL_AB40_IDE * (pow(AB40_Monomer * Unit_removal_1, AB40_IDE_Hill) / (pow(AB40_Monomer * Unit_removal_1, AB40_IDE_Hill) + pow(AB40_IDE_IC50, AB40_IDE_Hill)))) * VIS_brain")
    kinetic_law.setMath(math_ast)
    
    # Add AB42 clearance reaction
    reaction7 = model.createReaction()
    reaction7.setId("IDE_AB42_clearance")
    reaction7.setName("Clearance of AB42 monomer")
    reaction7.setReversible(False)
    
    # Reactant: AB42_monomer
    reactant = reaction7.createReactant()
    reactant.setSpecies("AB42_Monomer")
    reactant.setConstant(False)
    reactant.setStoichiometry(1.0)
    
    # Add Sink as product
    product = reaction7.createProduct()
    product.setSpecies(sinks["AB42_Monomer"])
    product.setConstant(True)
    product.setStoichiometry(1.0)
    
    # Kinetic law for IDE clearance of AB42
    kinetic_law = reaction7.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(IDE_conc * CL_AB42_IDE * (pow(AB42_Monomer * Unit_removal_1, AB42_IDE_Hill) / (pow(AB42_Monomer * Unit_removal_1, AB42_IDE_Hill) + pow(AB42_IDE_IC50, AB42_IDE_Hill)))) * VIS_brain")
    kinetic_law.setMath(math_ast)
    
    
    return document

def write_model_to_file(document, output_file):
    """Write the SBML document to a file"""
    libsbml.writeSBMLToFile(document, output_file)
    print(f"Model written to {output_file}")

def run_ab_production_model(csv_path, output_file="Geerts_AB_Production_Model.xml"):
    """Run the AB production model with parameters from CSV file"""
    params, params_with_units = load_parameters(csv_path)
    document = create_geerts_monomer_production_model(params, params_with_units)
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
    
    run_ab_production_model(csv_path)
    