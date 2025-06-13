"""
Combined Master Model Generator

This script combines multiple individual SBML modules into a single comprehensive model that integrates:
1. Amyloid beta (Aβ) production and aggregation dynamics (QSP components)
2. Antibody pharmacokinetics and distribution (PBPK components)
3. Perivascular space (PVS) and microglial interactions

The combination process follows these steps:
1. Loads parameters from CSV using specialized loaders from each module
2. Creates individual SBML models for each component:
   - AB_production: Monomer production and clearance
   - Geerts_dimer: Dimer formation and dynamics
   - Geerts_Oligomer_3_12: Small oligomer formation (3-12mers)
   - Geerts_Oligomer_13_16: Medium oligomer formation (13-16mers)
   - Geerts_Fibril_17_18: Small fibril formation (17-18mers)
   - Geerts_Fibril_19_24: Large fibril and plaque formation (19-24mers)
   - Geerts_PBPK_mAb: Antibody pharmacokinetics
   - Geerts_PBPK_monomers: Monomer pharmacokinetics
   - Geerts_PVS_ARIA: Perivascular space dynamics
   - geerts_microglia: Microglial interactions

3. Combines all models into a single SBML document by:
   - Copying unit definitions from the first model
   - Transferring compartments, species, parameters, reactions, and rules
   - Handling duplicate components and reporting conflicts
   - Preserving all mathematical relationships and kinetic laws

4. Validates the combined model and saves it as an SBML file

The resulting model can be used for both single-dose and multi-dose simulations,
with support for different drug types (gantenerumab or lecanemab).
"""

import sys
from pathlib import Path
root_dir = Path(__file__).parents[3]  # Go up 3 levels to reach pbpk_sbml_jax
sys.path.append(str(root_dir))

import libsbml
import pandas as pd
from K_rates_extrapolate import calculate_k_rates

# Import specialized parameter loaders from all modules
# QSP Components (Amyloid Beta Kinetics)
from .AB_production import load_parameters as load_ab_parameters  # Amyloid precursor protein (APP) processing and Aβ production
from .Geerts_dimer import load_parameters as load_dimer_parameters  # Aβ dimer formation
from .Geerts_Oligomer_3_12 import load_parameters as load_oligomer_3_12_parameters  # Small oligomer formation (3-12mers)
from .Geerts_Oligomer_13_16 import load_parameters as load_oligomer_13_16_parameters  # Medium oligomer formation (13-16mers)
from .Geerts_Fibril_17_18 import load_parameters as load_fibril_17_18_parameters  # Fibril formation (17-18mers)
from .Geerts_Fibril_19_24 import load_parameters as load_fibril_19_24_parameters  # Large fibril formation (19-24mers)

# PBPK Components (Antibody Distribution)
from .Geerts_PBPK_mAb_2 import load_parameters as load_mab_parameters  # Monoclonal antibody pharmacokinetics
from .Geerts_PBPK_monomers import load_parameters as load_monomers_parameters  # Aβ monomer pharmacokinetics
from .Geerts_PVS_ARIA import load_parameters as load_pvs_parameters  # Perivascular space and ARIA modeling

# Import model creators
# QSP Components
from .AB_production import create_geerts_monomer_production_model
from .Geerts_dimer import create_dimer_model
from .Geerts_Oligomer_3_12 import create_oligomer_3_12_model
from .Geerts_Oligomer_13_16 import create_oligomer_13_16_model
from .Geerts_Fibril_17_18 import create_fibril_17_18_model
from .Geerts_Fibril_19_24 import create_fibril_19_24_model

# PBPK Components
from .Geerts_PBPK_mAb_2 import create_geerts_model
from .Geerts_PBPK_monomers import create_parameterized_model
from .geerts_microglia import create_microglia_model
from .Geerts_PVS_ARIA import create_pvs_aria_model

def check(value, message):
    """If 'value' is None, prints an error message constructed using
    'message' and then exits with status code 1. If 'value' is an integer,
    it assumes it is a libSBML return status code. If the code value is
    LIBSBML_OPERATION_SUCCESS, returns without further action; if it is not,
    prints an error message constructed using 'message' along with text from
    libSBML explaining the meaning of the code, and exits with status code 1.
    """
    if value is None:
        raise SystemExit('LibSBML returned a null value trying to ' + message + '.')
    elif type(value) is int:
        if value == libsbml.LIBSBML_OPERATION_SUCCESS:
            return
        else:
            err_msg = 'Error encountered trying to ' + message + '.' \
                    + 'LibSBML returned error code ' + str(value) + ': "' \
                    + libsbml.OperationReturnValue_toString(value).strip() + '"'
            raise SystemExit(err_msg)
    else:
        return

def load_parameters_from_csv(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file using specialized loaders from each module
    
    Args:
        csv_path: Path to the CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        
    Returns:
        Tuple of (params, params_with_units)
    """
    print(f"\n=== Loading parameters for Combined Master Model ({drug_type.upper()}) ===")
    print(f"Loading parameters from {csv_path}")
    
    # First, load parameters using the core loaders to establish baselines
    print("\n1. Loading core parameters...")
    dimer_params, dimer_params_with_units = load_dimer_parameters(csv_path, drug_type=drug_type)
    mab_params, mab_params_with_units = load_mab_parameters(csv_path, drug_type=drug_type)
    monomers_params, monomers_params_with_units = load_monomers_parameters(csv_path, drug_type=drug_type)
    pvs_params, pvs_params_with_units = load_pvs_parameters(csv_path, drug_type=drug_type)
    
    # Merge the parameter dictionaries, starting with the most comprehensive loaders
    params = dimer_params.copy()
    params_with_units = dimer_params_with_units.copy()
    
    # Add PBPK-specific parameters from mAb and monomers modules
    print("\n2. Merging PBPK-specific parameters...")
    for key, value in mab_params.items():
        if key not in params:
            params[key] = value
            if key in mab_params_with_units:
                params_with_units[key] = mab_params_with_units[key]
    
    for key, value in monomers_params.items():
        if key not in params:
            params[key] = value
            if key in monomers_params_with_units:
                params_with_units[key] = monomers_params_with_units[key]
    
    # Add PVS-specific parameters
    print("\n2a. Merging PVS-specific parameters...")
    for key, value in pvs_params.items():
        if key not in params:
            params[key] = value
            if key in pvs_params_with_units:
                params_with_units[key] = pvs_params_with_units[key]
    
    # Add extrapolated rates using the parameterized version
    print("\n3. Adding extrapolated rate constants...")
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
    
    for rate_name, rate_value in extrapolated_rates.items():
        params[rate_name] = rate_value
        params_with_units[rate_name] = (rate_value, "1/h")
    
    # Verify that all critical parameters are available
    print("\n4. Validating parameter completeness...")
    critical_parameter_groups = {
        "Antibody binding": ['fta0', 'fta1', 'fta2', 'fta3', 'fta4'],
        "PBPK core": ['Vcent', 'Vper', 'PK_CL', 'PK_CLd2', 'VIS_brain', 'kon_FcRn', 'koff_FcRn'],
        "Oligomerization": ['k_M_O2_forty', 'k_M_O2_fortytwo', 'k_O2_M_forty', 'k_O2_M_fortytwo'],
        "Initial conditions": ['AB40_Monomer_0', 'AB42_Monomer_0', 'AB40_Oligomer02_0', 'AB42_Oligomer02_0'],
        "Production": ['AB40Mu_systemic_synthesis_rate', 'AB42Mu_systemic_synthesis_rate', 'k_APP_production'],
        "PVS parameters": ['Q_PVS', 'sigma_PVS', 'V_PVS', 'sigma_ISF', 'PS_Ab_ISF_PVS']
    }
    
    # Check each group of critical parameters
    for group_name, param_list in critical_parameter_groups.items():
        print(f"\n{group_name} parameters:")
        for param in param_list:
            if param in params:
                print(f"  ✓ {param}: {params[param]}")
            else:
                print(f"  ✗ {param}: NOT FOUND - Model may not function correctly!")
    
    # Print summary statistics
    print(f"\nTotal parameters loaded: {len(params)}")
    
    return params, params_with_units

def create_combined_master_model(params, params_with_units, drug_type="gantenerumab"):
    """Create a master SBML model that combines all modules from both AB_Master_Model and Geerts_Master_Model
    
    Args:
        params: Dictionary of parameters for all models
        params_with_units: Dictionary of parameters with units
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        
    Returns:
        SBML document containing the combined model
    """
    print(f"\n=== Creating Combined Master Model for {drug_type.upper()} ===")
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Combined_Master_Model")
    model.setTimeUnits("hour")
    
    # Add explicit check for drug-specific parameters before creating models
    if drug_type == "lecanemab":
        # Verify the correct drug parameters are being used
        drug_specific_params = {
            'Vcent': params.get('Vcent'),
            'Vper': params.get('Vper'),
            'PK_CL': params.get('PK_CL'),
            'PK_CLd2': params.get('PK_CLd2'),
            'fta0': params.get('fta0')
        }
        print(f"\nConfirming drug-specific parameters for {drug_type}:")
        for key, value in drug_specific_params.items():
            print(f"  {key}: {value}")
    
    print("Creating AB_production model...")
    ab_production_doc = create_geerts_monomer_production_model(params, params_with_units)
    
    print("Creating Geerts_dimer model...")
    dimer_doc = create_dimer_model(params, params_with_units)
    
    print("Creating Geerts_Oligomer_3_12 model...")
    oligomer_3_12_doc = create_oligomer_3_12_model(params, params_with_units)
    
    print("Creating Geerts_Oligomer_13_16 model...")
    oligomer_13_16_doc = create_oligomer_13_16_model(params, params_with_units)

    print("Creating Geerts_Fibril_17_18 model...")
    fibril_17_18_doc = create_fibril_17_18_model(params, params_with_units)
    
    print("Creating Geerts_Fibril_19_24 model...")
    fibril_19_24_doc = create_fibril_19_24_model(params, params_with_units)
    
    print("Creating Geerts_PBPK_mAb model...")
    geerts_doc = create_geerts_model(params, params_with_units)
    
    print("Creating Geerts_PBPK_monomers model...")
    parameterized_doc = create_parameterized_model(params, params_with_units, drug_type=drug_type)
    
    print("Creating Microglia model...")
    microglia_doc = create_microglia_model(params, params_with_units)
    
    print("Creating PVS/ARIA model...")
    pvs_doc = create_pvs_aria_model(params, params_with_units)
    
    # Get all models
    ab_models = [
        ab_production_doc.getModel(),
        dimer_doc.getModel(),
        oligomer_3_12_doc.getModel(),
        oligomer_13_16_doc.getModel(),
        fibril_17_18_doc.getModel(),
        fibril_19_24_doc.getModel()
    ]
    ab_model_names = [
        "AB_production",
        "Geerts_dimer",
        "Geerts_Oligomer_3_12",
        "Geerts_Oligomer_13_16",
        "Geerts_Fibril_17_18",
        "Geerts_Fibril_19_24"
    ]
    
    geerts_models = [
        geerts_doc.getModel(),
        parameterized_doc.getModel(),
        microglia_doc.getModel(),
        pvs_doc.getModel()
    ]
    geerts_model_names = [
        "Geerts_PBPK_mAb",
        "Geerts_PBPK_monomers",
        "Microglia",
        "Geerts_PVS_ARIA"
    ]
    
    # Combine all models
    all_models = ab_models + geerts_models
    model_names = ab_model_names + geerts_model_names
    
    # Copy unit definitions from first model 
    for i in range(ab_models[0].getNumUnitDefinitions()):
        unit_def = ab_models[0].getUnitDefinition(i)
        if not model.getUnitDefinition(unit_def.getId()):
            new_unit_def = model.createUnitDefinition()
            new_unit_def.setId(unit_def.getId())
            for j in range(unit_def.getNumUnits()):
                unit = unit_def.getUnit(j)
                new_unit = new_unit_def.createUnit()
                new_unit.setKind(unit.getKind())
                new_unit.setExponent(unit.getExponent())
                new_unit.setScale(unit.getScale())
                new_unit.setMultiplier(unit.getMultiplier())

    # Track transferred components
    transferred_compartments = set()
    transferred_species = set()
    transferred_parameters = set()
    transferred_reactions = set()
    transferred_rules = set()
    
    # Track duplicate counts for reporting
    duplicate_reactions = []
    duplicate_species = []
    duplicate_compartments = []
    duplicate_parameters = []
    duplicate_rules = []

    # Transfer components from all models
    for model_idx, source_model in enumerate(all_models):
        model_name = model_names[model_idx]
        print(f"\nProcessing model: {model_name}")
        
        # Transfer compartments
        for i in range(source_model.getNumCompartments()):
            comp = source_model.getCompartment(i)
            comp_id = comp.getId()
            if comp_id not in transferred_compartments:
                transferred_compartments.add(comp_id)
                new_comp = model.createCompartment()
                new_comp.setId(comp_id)
                new_comp.setConstant(comp.getConstant())
                new_comp.setSize(comp.getSize())
                if comp.isSetUnits():
                    new_comp.setUnits(comp.getUnits())
            else:
                duplicate_compartments.append((comp_id, model_name))

        # Transfer parameters
        for i in range(source_model.getNumParameters()):
            param = source_model.getParameter(i)
            param_id = param.getId()
            if param_id not in transferred_parameters:
                transferred_parameters.add(param_id)
                new_param = model.createParameter()
                new_param.setId(param_id)
                new_param.setValue(param.getValue())
                new_param.setConstant(param.getConstant())
                if param.isSetUnits():
                    new_param.setUnits(param.getUnits())
            else:
                duplicate_parameters.append((param_id, model_name))

        # Transfer species
        for i in range(source_model.getNumSpecies()):
            species = source_model.getSpecies(i)
            species_id = species.getId()
            if species_id not in transferred_species:
                transferred_species.add(species_id)
                new_species = model.createSpecies()
                new_species.setId(species_id)
                new_species.setConstant(species.getConstant())
                new_species.setBoundaryCondition(species.getBoundaryCondition())
                new_species.setCompartment(species.getCompartment())
                new_species.setHasOnlySubstanceUnits(species.getHasOnlySubstanceUnits())
                if species.isSetInitialAmount():
                    new_species.setInitialAmount(species.getInitialAmount())
                if species.isSetInitialConcentration():
                    new_species.setInitialConcentration(species.getInitialConcentration())
                if species.isSetSubstanceUnits():
                    new_species.setSubstanceUnits(species.getSubstanceUnits())
                if species.isSetUnits():
                    new_species.setUnits(species.getUnits())
            else:
                duplicate_species.append((species_id, model_name))

        # Transfer reactions
        for i in range(source_model.getNumReactions()):
            reaction = source_model.getReaction(i)
            reaction_id = reaction.getId()
            if reaction_id not in transferred_reactions:
                transferred_reactions.add(reaction_id)
                new_reaction = model.createReaction()
                new_reaction.setId(reaction_id)
                new_reaction.setReversible(reaction.getReversible())
                
                # Transfer reactants
                for j in range(reaction.getNumReactants()):
                    reactant = reaction.getReactant(j)
                    new_reactant = new_reaction.createReactant()
                    new_reactant.setSpecies(reactant.getSpecies())
                    new_reactant.setStoichiometry(reactant.getStoichiometry())
                    new_reactant.setConstant(reactant.getConstant())
                
                # Transfer products
                for j in range(reaction.getNumProducts()):
                    product = reaction.getProduct(j)
                    new_product = new_reaction.createProduct()
                    new_product.setSpecies(product.getSpecies())
                    new_product.setStoichiometry(product.getStoichiometry())
                    new_product.setConstant(product.getConstant())
                
                # Transfer modifiers
                for j in range(reaction.getNumModifiers()):
                    modifier = reaction.getModifier(j)
                    new_modifier = new_reaction.createModifier()
                    new_modifier.setSpecies(modifier.getSpecies())
                
                # Transfer kinetic law
                if reaction.isSetKineticLaw():
                    kinetic = reaction.getKineticLaw()
                    new_kinetic = new_reaction.createKineticLaw()
                    new_kinetic.setMath(kinetic.getMath().deepCopy())
                    
                    # Transfer local parameters in the kinetic law
                    for j in range(kinetic.getNumLocalParameters()):
                        local_param = kinetic.getLocalParameter(j)
                        new_local_param = new_kinetic.createLocalParameter()
                        new_local_param.setId(local_param.getId())
                        new_local_param.setValue(local_param.getValue())
                        if local_param.isSetUnits():
                            new_local_param.setUnits(local_param.getUnits())
            else:
                duplicate_reactions.append((reaction_id, model_name))
        
        # Transfer rules (including assignment rules)
        for i in range(source_model.getNumRules()):
            rule = source_model.getRule(i)
            # Use the rule's ID if it has one, otherwise use the variable name
            rule_id = rule.getId() if rule.isSetId() else rule.getVariable()
            
            if rule_id not in transferred_rules:
                transferred_rules.add(rule_id)
                
                if rule.isAssignment():
                    new_rule = model.createAssignmentRule()
                    new_rule.setVariable(rule.getVariable())
                elif rule.isRate():
                    new_rule = model.createRateRule()
                    new_rule.setVariable(rule.getVariable())
                    if rule.isSetId():
                        new_rule.setId(rule.getId())  # Preserve the rule ID
                elif rule.isAlgebraic():
                    new_rule = model.createAlgebraicRule()
                
                new_rule.setMath(rule.getMath().deepCopy())
            else:
                duplicate_rules.append((rule_id, model_name))

    # Report duplicates
    if duplicate_reactions:
        print("\n=== DUPLICATE REACTIONS DETECTED ===")
        print("The following reaction IDs were found in multiple models:")
        for reaction_id, model_name in duplicate_reactions:
            print(f"  - '{reaction_id}' in {model_name}")
        print("Consider renaming these reactions to ensure all reactions are included in the master model.")
    
    
    if duplicate_rules:
        print("\n=== DUPLICATE RULES DETECTED ===")
        print("The following rule variables were found in multiple models:")
        for rule_id, model_name in duplicate_rules:
            print(f"  - '{rule_id}' in {model_name}")
        print("This is expected for shared rules between models.")
    
    print(f"\nCombined master model created with:")
    print(f"  - {model.getNumCompartments()} compartments")
    print(f"  - {model.getNumSpecies()} species")
    print(f"  - {model.getNumParameters()} parameters")
    print(f"  - {model.getNumReactions()} reactions")
    print(f"  - {model.getNumRules()} rules")

    # After transferring all components, check critical parameters in the final model
    # This will help identify if values were overwritten during model combination
    print("\nVerifying critical parameters in the final model:")
    critical_params = ['Vcent', 'Vper', 'PK_CL', 'PK_CLd2', 'fta0']
    for param_id in critical_params:
        param = model.getParameter(param_id)
        if param:
            print(f"  {param_id}: {param.getValue()}")
        else:
            print(f"  {param_id}: NOT FOUND IN MODEL")

    return document

def save_model(document, filename):
    """Save the SBML document to a file"""
    libsbml.writeSBMLToFile(document, filename)

def main():
    """Create and save the combined master model"""
    print("\n=== Starting Combined Master Model generation ===")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create combined master model")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], 
                      default="gantenerumab", help="Drug type to simulate")
    args = parser.parse_args()
    
    drug_type = args.drug
    
    # Update path to correct location
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"combined_master_model_{drug_type}.xml"
    
    # Load parameters
    params_path = "parameters/PK_Geerts.csv"
    print(f"Loading parameters from: {params_path}")
    
    # Check if parameter file exists before proceeding
    param_file = Path(params_path)
    if not param_file.is_file():
        print(f"ERROR: Parameter file not found at {param_file.absolute()}")
        print(f"Current working directory: {Path.cwd()}")
        print("Please ensure the parameter file exists at the correct location.")
        sys.exit(1)
    
    # Load parameters
    params, params_with_units = load_parameters_from_csv(
        params_path,
        drug_type=drug_type
    )
    
    # Create the combined master model
    document = create_combined_master_model(
        params,
        params_with_units,
        drug_type=drug_type
    )
    
    # Check for errors
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        # Save the model
        save_model(document, str(output_path))
        print(f"Combined master model for {drug_type.upper()} saved successfully to {output_path}!")

if __name__ == "__main__":
    main() 