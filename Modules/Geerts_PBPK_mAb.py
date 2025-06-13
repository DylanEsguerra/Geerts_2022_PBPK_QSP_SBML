"""
Module for modeling monoclonal antibody (mAb) pharmacokinetics in the brain.
This is a PBPK component that models the distribution and clearance of therapeutic antibodies.
The module includes:
- Antibody distribution across blood-brain barrier (BBB)
- Antibody distribution across blood-CSF barrier (BCSFB)
- FcRn-mediated antibody recycling
- Antibody binding to Aβ species
- Clearance mechanisms for antibody-Aβ complexes
"""

# Geerts Model with Reactions
import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file
    
    Args:
        csv_path: Path to the CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
        
    Returns:
        Tuple of (params, params_with_units)
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} ===")
    print(f"Loading parameters from {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    
    # Create parameter dictionaries
    params = {}
    params_with_units = {}
    
    # Load all parameters from CSV
    for _, row in df.iterrows():
        name = row['name']
        value = float(row['value'])
        unit = row['units'] if 'units' in row else None
        params[name] = value
        params_with_units[name] = (value, unit) if unit else value
    
    
    # Handle drug-specific parameter mapping
    is_lecanemab = drug_type.lower() == "lecanemab"
    
    # Map drug-specific parameters
    param_mapping = {
        'Vcent': 'Lec_Vcent' if is_lecanemab else 'Gant_Vcent',
        'Vper': 'Lec_Vper' if is_lecanemab else 'Gant_Vper',
        'PK_CL': 'Lec_CL' if is_lecanemab else 'Gant_CL',
        'PK_CLd2': 'Lec_CLd2' if is_lecanemab else 'Gant_CLd2',
        'PK_SC_ka': 'Lec_SC_ka' if is_lecanemab else 'Gant_SC_ka',
        'PK_SC_bio': 'Lec_SC_bio' if is_lecanemab else 'Gant_SC_bio',
    }
    
    # Apply parameter mapping
    for generic_name, specific_name in param_mapping.items():
        if specific_name in params:
            params[generic_name] = params[specific_name]
            if specific_name in params_with_units:
                params_with_units[generic_name] = params_with_units[specific_name]
    
    
    return params, params_with_units

def create_geerts_model(params, params_with_units):
    """Create Geerts PK model using reactions instead of rate rules
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
    """
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_PBPK_mAb_Model")
    model.setTimeUnits("hour")
    
    # Add dosing schedule parameters
    input_cent_param = model.createParameter()
    input_cent_param.setId("InputCent")
    input_cent_param.setValue(0.0)
    input_cent_param.setConstant(False)
    input_cent_param.setUnits("nanomole_per_hour")
    
    input_sc_param = model.createParameter()
    input_sc_param.setId("InputSC")
    input_sc_param.setValue(0.0)
    input_sc_param.setConstant(False)
    input_sc_param.setUnits("nanomole_per_hour")
    
    # Add parameters for dose amounts - use the values from params
    iv_dose_amount_param = model.createParameter()
    iv_dose_amount_param.setId("IV_DoseAmount")
    iv_dose_amount_param.setValue(params.get('IV_DoseAmount', 0.0))
    iv_dose_amount_param.setConstant(False)
    iv_dose_amount_param.setUnits("nanomole")
    
    sc_dose_amount_param = model.createParameter()
    sc_dose_amount_param.setId("SC_DoseAmount")
    sc_dose_amount_param.setValue(params.get('SC_DoseAmount', 0.0))
    sc_dose_amount_param.setConstant(False)
    sc_dose_amount_param.setUnits("nanomole")
    
    # Add parameters for number of doses
    iv_num_doses_param = model.createParameter()
    iv_num_doses_param.setId("IV_NumDoses")
    iv_num_doses_param.setValue(params.get('IV_NumDoses', 0))
    iv_num_doses_param.setConstant(False)
    iv_num_doses_param.setUnits("dimensionless")
    
    sc_num_doses_param = model.createParameter()
    sc_num_doses_param.setId("SC_NumDoses")
    sc_num_doses_param.setValue(params.get('SC_NumDoses', 0))
    sc_num_doses_param.setConstant(False)
    sc_num_doses_param.setUnits("dimensionless")
    
    # Add parameters for dose duration - REMOVE THE SCALE FACTOR FOR TIME PARAMETERS
    iv_dose_duration_param = model.createParameter()
    iv_dose_duration_param.setId("IV_DoseDuration")
    iv_dose_duration_param.setValue(1.0)  # No scaling for time parameters
    iv_dose_duration_param.setConstant(True)
    iv_dose_duration_param.setUnits("hour")
    
    sc_dose_duration_param = model.createParameter()
    sc_dose_duration_param.setId("SC_DoseDuration")
    sc_dose_duration_param.setValue(1.0)  # No scaling for time parameters
    sc_dose_duration_param.setConstant(True)
    sc_dose_duration_param.setUnits("hour")
    
    # Add parameters for dosing intervals - REMOVE THE SCALE FACTOR FOR TIME PARAMETERS
    iv_dose_interval_param = model.createParameter()
    iv_dose_interval_param.setId("IV_DoseInterval")
    iv_dose_interval_param.setValue(params.get('IV_DoseInterval', 336.0))
    iv_dose_interval_param.setConstant(True)
    iv_dose_interval_param.setUnits("hour")
    
    sc_dose_interval_param = model.createParameter()
    sc_dose_interval_param.setId("SC_DoseInterval")
    sc_dose_interval_param.setValue(params.get('SC_DoseInterval', 672.0))
    sc_dose_interval_param.setConstant(True)
    sc_dose_interval_param.setUnits("hour")
    
    # Add parameters for maximum dosing time - REMOVE THE SCALE FACTOR FOR TIME PARAMETERS
    max_dosing_time_param = model.createParameter()
    max_dosing_time_param.setId("MaxDosingTime")
    max_dosing_time_param.setValue(params.get('MaxDosingTime', 13140.0))
    max_dosing_time_param.setConstant(True)
    max_dosing_time_param.setUnits("hour")
    
    # Initialize compartments (all are constant)
    compartments = {
        "BBB_Bound": params["VBBB_brain"],
        "BBB_Unbound": params["VBBB_brain"],
        "FcRn_free_BBB": params["VBBB_brain"],
        "BCSFB_Bound": params["V_BCSFB_brain"],
        "BCSFB_Unbound": params["V_BCSFB_brain"],
        "FcRn_free_BCSFB": params["V_BCSFB_brain"],
        "Brain_plasma": params["Vp_brain"],
        "Central_compartment": params["Vcent"],
        "CSF_CM": params["V_CM_brain"],
        "CSF_LV": params["V_LV_brain"],
        "CSF_SAS": params["V_SAS_brain"],
        "CSF_TFV": params["V_TFV_brain"],
        "Peripheral_compartment": params["Vper"],
        "ISF_brain": params["VIS_brain"],
        "SubCut_absorption_compartment": params["V_SubCut"]
    }

    # Create compartments
    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(f"comp_{comp_id}")  # Add prefix to avoid conflicts
        comp.setConstant(True)
        comp.setSize(size)
        comp.setUnits("litre")

    # Create dictionaries to store sinks and sources for each species
    sinks = {}
    sources = {}
    
    # Pre-create sinks and sources for required species
    sources["PK_central"] = create_source_for_species("PK_central", model)
    sources["SubCut_absorption"] = create_source_for_species("SubCut_absorption", model)
    
    sinks["PK_central"] = create_sink_for_species("PK_central", model)
    sinks["SubCut_absorption"] = create_sink_for_species("SubCut_absorption", model)
    sinks["PK_BBB_unbound_brain"] = create_sink_for_species("PK_BBB_unbound_brain", model)
    sinks["PK_BCSFB_unbound_brain"] = create_sink_for_species("PK_BCSFB_unbound_brain", model)

    # Create all parameters
    print("\nCreating parameters...")
    required_params = [
        # Core volumes (always keep)
        ("VBBB_brain", params["VBBB_brain"]),
        ("V_BCSFB_brain", params["V_BCSFB_brain"]),
        ("Vp_brain", params["Vp_brain"]),
        ("Vcent", params["Vcent"]),
        ("V_CM_brain", params["V_CM_brain"]),
        ("V_LV_brain", params["V_LV_brain"]),
        ("V_SAS_brain", params["V_SAS_brain"]),
        ("V_TFV_brain", params["V_TFV_brain"]),
        ("Vper", params["Vper"]),
        ("V_SubCut", params["V_SubCut"]),
        ("VIS_brain", params["VIS_brain"]),
        ("VES_brain", params["VES_brain"]),
        
        # Flow rates
        ("Q_p_brain", params["Q_p_brain"]),
        ("Q_CSF_brain", params["Q_CSF_brain"]),
        ("Q_ISF_brain", params["Q_ISF_brain"]),
        ("Q_PVS", params["Q_PVS"]),
        ("L_brain", params["L_brain"]),
        
        # Global kinetic parameters
        ("kon_FcRn", params["kon_FcRn"]),       # FcRn binding rate
        ("koff_FcRn", params["koff_FcRn"]),     # FcRn unbinding rate
        ("kdeg", params["kdeg"]),               # Degradation rate
        ("FR", params["FR"]),                   # Global recycling fraction
        ("f_BBB", params["f_BBB"]),
        ("f_LV", params["f_LV"]),
        ("f_BCSFB", params["f_BCSFB"]),
        ("PK_SC_ka", params["PK_SC_ka"]),
        ("PK_SC_bio", params["PK_SC_bio"]),
        ("PK_CL", params["PK_CL"]),
        ("PK_CLd2", params["PK_CLd2"]),
        ("InputCent", params["InputCent"]),
        ("InputSC", params["InputSC"]),
        ("FcRn_free_BBB_0", params["FcRn_free_BBB_0"]),
        ("FcRn_free_BCSFB_0", params["FcRn_free_BCSFB_0"]),
        
        # Reflection coefficients
        ("sigma_V_brain_ISF", params["sigma_V_brain_ISF"]),
        ("sigma_V_BCSFB", params["sigma_V_BCSFB"]),
        ("sigma_L_brain_ISF", params["sigma_L_brain_ISF"]),
        ("sigma_L_SAS", params["sigma_L_SAS"]),
        
        # Clearance
        ("CLup_brain", params["CLup_brain"]),
    ]
    
    for param_id, value in required_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)  # No scaling for volumes, rates, etc.
        
        param.setConstant(True)
        
        # More specific unit assignments based on parameter type
        if param_id.startswith('V'):  # Volumes
            param.setUnits("litre")
        elif param_id.startswith('C'):  # Concentrations
            param.setUnits("nanomole")  # Initial condition units are mole, SBML will convert to molarity
        elif param_id.startswith('Q') or param_id.startswith('L'):  # Flows
            param.setUnits("litre_per_hour")
        elif param_id.startswith('k') and param_id.endswith('FcRn'):  # FcRn binding/unbinding rates
            if param_id.startswith('kon'):
                param.setUnits("per_nanomole_per_hour")  # For second-order reaction
            else:
                param.setUnits("per_hour")  # For first-order reaction
        elif param_id.startswith('kdeg'):  # Degradation rates
            param.setUnits("per_hour")
        elif param_id.startswith('sigma'):  # Reflection coefficients
            param.setUnits("dimensionless")
        elif param_id.startswith('f_') or param_id == 'FR':  # Fractions
            param.setUnits("dimensionless")
        elif param_id.startswith('CLup'):  # Clearance
            param.setUnits("per_hour")  # Changed to per_hour (L/h/L = h^-1)
        elif param_id.startswith('FcRn_free'):  # Free FcRn concentrations
            param.setUnits("nanomole")  # Initial condition units are mole, SBML will convert to molarity
        elif param_id.startswith('PK_SC_'):  # SubCut parameters
            param.setUnits("per_hour")
        else:
            # Use units from params_with_units if available
            if param_id in params_with_units:
                param.setUnits(params_with_units[param_id][1])
            else:
                param.setUnits("dimensionless")

    # Create species
    species = [
        ("PK_BBB_unbound_brain", "BBB_Unbound", params["PK_BBB_unbound_brain_0"]),
        ("PK_BBB_bound_brain", "BBB_Bound", params["PK_BBB_bound_brain_0"]),
        ("PK_p_brain", "Brain_plasma", params["PK_p_brain_0"]),
        ("Ab_t", "ISF_brain", params["Ab_t_0"]),
        ("PK_BCSFB_unbound_brain", "BCSFB_Unbound", params["PK_BCSFB_unbound_brain_0"]),
        ("PK_BCSFB_bound_brain", "BCSFB_Bound", params["PK_BCSFB_bound_brain_0"]),
        ("PK_LV_brain", "CSF_LV", params["PK_LV_brain_0"]),
        ("PK_TFV_brain", "CSF_TFV", params["PK_TFV_brain_0"]),
        ("PK_CM_brain", "CSF_CM", params["PK_CM_brain_0"]),
        ("PK_SAS_brain", "CSF_SAS", params["PK_SAS_brain_0"]),
        ("PK_central", "Central_compartment", params["PK_central_0"]),
        ("PK_per", "Peripheral_compartment", params["PK_per_0"]),
        ("SubCut_absorption", "SubCut_absorption_compartment", params["SubCut_absorption_0"]),
    ]
    
    # Create variable species
    for species_id, compartment_id, initial_value in species:
        spec = model.createSpecies()
        spec.setId(species_id)
        spec.setCompartment(f"comp_{compartment_id}")  # Use new compartment IDs
        spec.setInitialConcentration(initial_value)
        spec.setSubstanceUnits("nanomole_per_litre")
        spec.setHasOnlySubstanceUnits(False)
        spec.setBoundaryCondition(False)
        spec.setConstant(False)

    # create species for free FcRn
    fcrn_bbb_free = model.createSpecies()
    fcrn_bbb_free.setId("FcRn_free_BBB")
    fcrn_bbb_free.setCompartment("comp_BBB_Bound")  # Same compartment as bound species
    fcrn_bbb_free.setInitialConcentration(params["FcRn_free_BBB_0"])
    fcrn_bbb_free.setSubstanceUnits("nanomole_per_litre")
    fcrn_bbb_free.setHasOnlySubstanceUnits(False)
    fcrn_bbb_free.setBoundaryCondition(False)
    fcrn_bbb_free.setConstant(False)

    # Similar for BCSFB
    fcrn_bcsfb_free = model.createSpecies()
    fcrn_bcsfb_free.setId("FcRn_free_BCSFB")
    fcrn_bcsfb_free.setCompartment("comp_BCSFB_Bound")
    fcrn_bcsfb_free.setInitialConcentration(params["FcRn_free_BCSFB_0"])
    fcrn_bcsfb_free.setSubstanceUnits("nanomole_per_litre")
    fcrn_bcsfb_free.setHasOnlySubstanceUnits(False)
    fcrn_bcsfb_free.setBoundaryCondition(False)
    fcrn_bcsfb_free.setConstant(False)

    # Now create reactions instead of rate rules
    
    # 1. SubCut absorption processes
    
    # Input to SubCut compartment
    reaction_input_to_subcut = model.createReaction()
    reaction_input_to_subcut.setId("input_to_subcut")
    reaction_input_to_subcut.setReversible(False)
    
    # Add Source as reactant
    reactant_source = reaction_input_to_subcut.createReactant()
    reactant_source.setSpecies(sources["SubCut_absorption"])
    reactant_source.setStoichiometry(1.0)
    reactant_source.setConstant(True)
    
    product_subcut = reaction_input_to_subcut.createProduct()
    product_subcut.setSpecies("SubCut_absorption")
    product_subcut.setStoichiometry(1.0)
    product_subcut.setConstant(True)
    
    # Kinetic law: InputSC
    klaw_input_to_subcut = reaction_input_to_subcut.createKineticLaw()
    math_ast = libsbml.parseL3Formula("InputSC")
    klaw_input_to_subcut.setMath(math_ast)
    
    # SubCut absorption to central compartment
    reaction_subcut_cent = model.createReaction()
    reaction_subcut_cent.setId("subcut_to_central")
    reaction_subcut_cent.setReversible(False)
    
    reactant_sc = reaction_subcut_cent.createReactant()
    reactant_sc.setSpecies("SubCut_absorption")
    reactant_sc.setStoichiometry(1.0)
    reactant_sc.setConstant(True)
    
    product_central = reaction_subcut_cent.createProduct()
    product_central.setSpecies("PK_central")
    product_central.setStoichiometry(1.0)
    product_central.setConstant(True)
    
    # Kinetic law for subcutaneous absorption
    klaw_subcut_cent = reaction_subcut_cent.createKineticLaw()
    math_ast = libsbml.parseL3Formula("PK_SC_ka * PK_SC_bio * SubCut_absorption")
    klaw_subcut_cent.setMath(math_ast)
    
    # SubCut elimination (non-bioavailable fraction)
    reaction_sc_elimination = model.createReaction()
    reaction_sc_elimination.setId("subcut_elimination")
    reaction_sc_elimination.setReversible(False)
    
    reactant_sc_elim = reaction_sc_elimination.createReactant()
    reactant_sc_elim.setSpecies("SubCut_absorption")
    reactant_sc_elim.setStoichiometry(1.0)
    reactant_sc_elim.setConstant(True)
    
    # Add Sink as product
    product_sink = reaction_sc_elimination.createProduct()
    product_sink.setSpecies(sinks["SubCut_absorption"])
    product_sink.setStoichiometry(1.0)
    product_sink.setConstant(True)
    
    # Kinetic law: PK_SC_ka * (1 - PK_SC_bio) * SubCut_absorption_compartment
    klaw_sc_elim = reaction_sc_elimination.createKineticLaw()
    math_ast = libsbml.parseL3Formula("PK_SC_ka * (1 - PK_SC_bio) * SubCut_absorption")
    klaw_sc_elim.setMath(math_ast)
    
    # 2. Central compartment processes
    
    # Input to central compartment
    reaction_input_central = model.createReaction()
    reaction_input_central.setId("input_to_central")
    reaction_input_central.setReversible(False)
    
    # Add Source as reactant
    reactant_source = reaction_input_central.createReactant()
    reactant_source.setSpecies(sources["PK_central"])
    reactant_source.setStoichiometry(1.0)
    reactant_source.setConstant(True)
    
    product_central_input = reaction_input_central.createProduct()
    product_central_input.setSpecies("PK_central")
    product_central_input.setStoichiometry(1.0)
    product_central_input.setConstant(True)
    
    # Kinetic law: InputCent
    klaw_input_central = reaction_input_central.createKineticLaw()
    math_ast = libsbml.parseL3Formula("InputCent")
    klaw_input_central.setMath(math_ast)
    
    # Central to peripheral compartment
    reaction_central_to_per = model.createReaction()
    reaction_central_to_per.setId("central_to_peripheral")
    reaction_central_to_per.setReversible(False)
    
    reactant_central = reaction_central_to_per.createReactant()
    reactant_central.setSpecies("PK_central")
    reactant_central.setStoichiometry(1.0)
    reactant_central.setConstant(True)
    
    product_per = reaction_central_to_per.createProduct()
    product_per.setSpecies("PK_per")
    product_per.setStoichiometry(1.0)
    product_per.setConstant(True)
    
    # Kinetic law: (PK_CLd2/Vcent) * PK_central
    klaw_central_to_per = reaction_central_to_per.createKineticLaw()
    math_ast = libsbml.parseL3Formula("PK_CLd2 * PK_central")
    klaw_central_to_per.setMath(math_ast)
    
    # Peripheral to central compartment
    reaction_per_to_central = model.createReaction()
    reaction_per_to_central.setId("peripheral_to_central")
    reaction_per_to_central.setReversible(False)
    
    reactant_per = reaction_per_to_central.createReactant()
    reactant_per.setSpecies("PK_per")
    reactant_per.setStoichiometry(1.0)
    reactant_per.setConstant(True)
    
    product_central_from_per = reaction_per_to_central.createProduct()
    product_central_from_per.setSpecies("PK_central")
    product_central_from_per.setStoichiometry(1.0)
    product_central_from_per.setConstant(True)
    
    # Kinetic law: (PK_CLd2/Vper) * PK_per
    klaw_per_to_central = reaction_per_to_central.createKineticLaw()
    math_ast = libsbml.parseL3Formula("PK_CLd2 * PK_per")
    klaw_per_to_central.setMath(math_ast)
    
    # Central compartment clearance
    reaction_central_clearance = model.createReaction()
    reaction_central_clearance.setId("central_clearance")
    reaction_central_clearance.setReversible(False)
    
    reactant_central_cl = reaction_central_clearance.createReactant()
    reactant_central_cl.setSpecies("PK_central")
    reactant_central_cl.setStoichiometry(1.0)
    reactant_central_cl.setConstant(True)
    
    # Add Sink as product
    product_sink = reaction_central_clearance.createProduct()
    product_sink.setSpecies(sinks["PK_central"])
    product_sink.setStoichiometry(1.0)
    product_sink.setConstant(True)
    
    # Kinetic law: (PK_CL/Vcent) * PK_central
    klaw_central_cl = reaction_central_clearance.createKineticLaw()
    math_ast = libsbml.parseL3Formula("PK_CL * PK_central")
    klaw_central_cl.setMath(math_ast)
    
    # 3. Central to brain plasma transport
    reaction_central_to_brain = model.createReaction()
    reaction_central_to_brain.setId("central_to_brain_plasma")
    reaction_central_to_brain.setReversible(False)
    
    reactant_central_brain = reaction_central_to_brain.createReactant()
    reactant_central_brain.setSpecies("PK_central")
    reactant_central_brain.setStoichiometry(1.0)
    reactant_central_brain.setConstant(True)
    
    product_brain = reaction_central_to_brain.createProduct()
    product_brain.setSpecies("PK_p_brain")
    product_brain.setStoichiometry(1.0)
    product_brain.setConstant(True)
    
    # Kinetic law: (Q_p_brain/Vcent) * PK_central
    klaw_central_to_brain = reaction_central_to_brain.createKineticLaw()
    math_ast = libsbml.parseL3Formula("Q_p_brain * PK_central")
    klaw_central_to_brain.setMath(math_ast)

    # 5. CSF flow affecting central compartment (SAS reflection)
    reaction_sas_central = model.createReaction()
    reaction_sas_central.setId("sas_flow_central")
    reaction_sas_central.setReversible(False)

    reactant_sas = reaction_sas_central.createReactant()
    reactant_sas.setSpecies("PK_SAS_brain")
    reactant_sas.setStoichiometry(1.0)
    reactant_sas.setConstant(True)

    product_central_sas = reaction_sas_central.createProduct()
    product_central_sas.setSpecies("PK_central")
    product_central_sas.setStoichiometry(1.0)
    product_central_sas.setConstant(True)

    # Kinetic law: (1 - sigma_L_SAS) * Q_CSF_brain * PK_SAS_brain
    klaw_sas_central = reaction_sas_central.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1 - sigma_L_SAS) * Q_CSF_brain * PK_SAS_brain")
    klaw_sas_central.setMath(math_ast)
    
    # 4. Brain plasma processes

    # Transport from brain plasma back to central compartment
    reaction_brain_to_central = model.createReaction()
    reaction_brain_to_central.setId("brain_plasma_to_central")
    reaction_brain_to_central.setReversible(False)

    reactant_brain = reaction_brain_to_central.createReactant()
    reactant_brain.setSpecies("PK_p_brain")
    reactant_brain.setStoichiometry(1.0)
    reactant_brain.setConstant(True)

    product_central_from_brain = reaction_brain_to_central.createProduct()
    product_central_from_brain.setSpecies("PK_central")
    product_central_from_brain.setStoichiometry(1.0)
    product_central_from_brain.setConstant(True)

    # Kinetic law: (Q_p_brain - L_brain) * PK_p_brain
    klaw_brain_to_central = reaction_brain_to_central.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(Q_p_brain - L_brain) * PK_p_brain")
    klaw_brain_to_central.setMath(math_ast)



    # 5. BBB Unbound processes

    # Uptake from ISF to BBB unbound
    reaction_isf_to_bbb = model.createReaction()
    reaction_isf_to_bbb.setId("isf_to_bbb_unbound")
    reaction_isf_to_bbb.setReversible(False)

    reactant_isf = reaction_isf_to_bbb.createReactant()
    reactant_isf.setSpecies("Ab_t")
    reactant_isf.setStoichiometry(1.0)
    reactant_isf.setConstant(True)

    product_bbb_unbound = reaction_isf_to_bbb.createProduct()
    product_bbb_unbound.setSpecies("PK_BBB_unbound_brain")
    product_bbb_unbound.setStoichiometry(1.0)
    product_bbb_unbound.setConstant(True)

    # Kinetic law: CLup_brain * f_BBB * VES_brain * Ab_t
    klaw_isf_to_bbb = reaction_isf_to_bbb.createKineticLaw()
    math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * Ab_t")
    klaw_isf_to_bbb.setMath(math_ast)

    # Uptake from brain plasma to BBB unbound
    reaction_plasma_to_bbb = model.createReaction()
    reaction_plasma_to_bbb.setId("plasma_to_bbb_unbound")
    reaction_plasma_to_bbb.setReversible(False)

    reactant_plasma_bbb = reaction_plasma_to_bbb.createReactant()
    reactant_plasma_bbb.setSpecies("PK_p_brain")
    reactant_plasma_bbb.setStoichiometry(1.0)
    reactant_plasma_bbb.setConstant(True)

    product_bbb_unbound_from_plasma = reaction_plasma_to_bbb.createProduct()
    product_bbb_unbound_from_plasma.setSpecies("PK_BBB_unbound_brain")
    product_bbb_unbound_from_plasma.setStoichiometry(1.0)
    product_bbb_unbound_from_plasma.setConstant(True)

    # Kinetic law: CLup_brain * f_BBB * VES_brain * PK_p_brain # Could be part of reaction Uptake from brain plasma to BBB/BCSFB
    klaw_plasma_to_bbb = reaction_plasma_to_bbb.createKineticLaw()
    math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * PK_p_brain")
    klaw_plasma_to_bbb.setMath(math_ast)

    # Degradation of BBB unbound
    reaction_bbb_unbound_degradation = model.createReaction()
    reaction_bbb_unbound_degradation.setId("bbb_unbound_degradation")
    reaction_bbb_unbound_degradation.setReversible(False)

    reactant_bbb_unbound_deg = reaction_bbb_unbound_degradation.createReactant()
    reactant_bbb_unbound_deg.setSpecies("PK_BBB_unbound_brain")
    reactant_bbb_unbound_deg.setStoichiometry(1.0)
    reactant_bbb_unbound_deg.setConstant(True)

    # Add Sink as product
    product_sink = reaction_bbb_unbound_degradation.createProduct()
    product_sink.setSpecies(sinks["PK_BBB_unbound_brain"])
    product_sink.setStoichiometry(1.0)
    product_sink.setConstant(True)

    # Kinetic law: kdeg * PK_BBB_unbound_brain * VBBB_brain
    klaw_bbb_unbound_deg = reaction_bbb_unbound_degradation.createKineticLaw()
    math_ast = libsbml.parseL3Formula("kdeg * PK_BBB_unbound_brain * VBBB_brain")
    klaw_bbb_unbound_deg.setMath(math_ast)

    # 6. BBB FcRn binding processes

    # Binding of BBB unbound to FcRn
    reaction_bbb_fcrn_binding = model.createReaction()
    reaction_bbb_fcrn_binding.setId("bbb_fcrn_binding")
    reaction_bbb_fcrn_binding.setReversible(False)

    reactant_bbb_unbound_bind = reaction_bbb_fcrn_binding.createReactant()
    reactant_bbb_unbound_bind.setSpecies("PK_BBB_unbound_brain")
    reactant_bbb_unbound_bind.setStoichiometry(1.0)
    reactant_bbb_unbound_bind.setConstant(True)

    reactant_fcrn_free = reaction_bbb_fcrn_binding.createReactant()
    reactant_fcrn_free.setSpecies("FcRn_free_BBB")
    reactant_fcrn_free.setStoichiometry(1.0)
    reactant_fcrn_free.setConstant(True)

    
    product_bbb_bound = reaction_bbb_fcrn_binding.createProduct()
    product_bbb_bound.setSpecies("PK_BBB_bound_brain")
    product_bbb_bound.setStoichiometry(1.0)
    product_bbb_bound.setConstant(True)

    # Kinetic law: kon_FcRn * PK_BBB_unbound_brain * FcRn_free_BBB * VBBB_brain
    klaw_bbb_binding = reaction_bbb_fcrn_binding.createKineticLaw()    
    math_ast = libsbml.parseL3Formula("kon_FcRn * PK_BBB_unbound_brain * FcRn_free_BBB * VBBB_brain")
    klaw_bbb_binding.setMath(math_ast)

    # Unbinding of BBB bound from FcRn
    reaction_bbb_fcrn_unbinding = model.createReaction()
    reaction_bbb_fcrn_unbinding.setId("bbb_fcrn_unbinding")
    reaction_bbb_fcrn_unbinding.setReversible(False)

    reactant_bbb_bound_unbind = reaction_bbb_fcrn_unbinding.createReactant()
    reactant_bbb_bound_unbind.setSpecies("PK_BBB_bound_brain")
    reactant_bbb_bound_unbind.setStoichiometry(1.0)
    reactant_bbb_bound_unbind.setConstant(True)

    product_bbb_unbound = reaction_bbb_fcrn_unbinding.createProduct()
    product_bbb_unbound.setSpecies("PK_BBB_unbound_brain")
    product_bbb_unbound.setStoichiometry(1.0)
    product_bbb_unbound.setConstant(True)

    product_fcrn_free = reaction_bbb_fcrn_unbinding.createProduct()
    product_fcrn_free.setSpecies("FcRn_free_BBB")
    product_fcrn_free.setStoichiometry(1.0)
    product_fcrn_free.setConstant(True)

    

    # Kinetic law: koff_FcRn * PK_BBB_bound_brain * VBBB_brain
    klaw_bbb_unbinding = reaction_bbb_fcrn_unbinding.createKineticLaw()
    math_ast = libsbml.parseL3Formula("koff_FcRn * PK_BBB_bound_brain * VBBB_brain")
    klaw_bbb_unbinding.setMath(math_ast)

    # 7. BBB Bound processes


    reaction_bbb_to_brain_plasma = model.createReaction()
    reaction_bbb_to_brain_plasma.setId("bbb_to_brain_plasma")
    reaction_bbb_to_brain_plasma.setReversible(False)

    reactant_bbb_bound_brain_plasma = reaction_bbb_to_brain_plasma.createReactant()
    reactant_bbb_bound_brain_plasma.setSpecies("PK_BBB_bound_brain")
    reactant_bbb_bound_brain_plasma.setStoichiometry(1.0)
    reactant_bbb_bound_brain_plasma.setConstant(True)

    product_brain_plasma_from_bbb = reaction_bbb_to_brain_plasma.createProduct()
    product_brain_plasma_from_bbb.setSpecies("PK_p_brain")
    product_brain_plasma_from_bbb.setStoichiometry(1.0)
    product_brain_plasma_from_bbb.setConstant(True)

    product_fcrn_free_from_bbb = reaction_bbb_to_brain_plasma.createProduct()
    product_fcrn_free_from_bbb.setSpecies("FcRn_free_BBB")
    product_fcrn_free_from_bbb.setStoichiometry(1.0)
    product_fcrn_free_from_bbb.setConstant(True)

    # Kinetic law: CLup_brain * f_BBB * VES_brain * FR * PK_BBB_bound_brain
    klaw_bbb_to_brain_plasma = reaction_bbb_to_brain_plasma.createKineticLaw()
    math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * FR * PK_BBB_bound_brain")
    klaw_bbb_to_brain_plasma.setMath(math_ast)



    # Transcytosis of BBB bound to ISF (non-recycled fraction)
    reaction_bbb_to_isf = model.createReaction()
    reaction_bbb_to_isf.setId("bbb_bound_to_isf")
    reaction_bbb_to_isf.setReversible(False)

    reactant_bbb_bound_isf = reaction_bbb_to_isf.createReactant()
    reactant_bbb_bound_isf.setSpecies("PK_BBB_bound_brain")
    reactant_bbb_bound_isf.setStoichiometry(1.0)
    reactant_bbb_bound_isf.setConstant(True)

    product_isf = reaction_bbb_to_isf.createProduct()
    product_isf.setSpecies("Ab_t")
    product_isf.setStoichiometry(1.0)
    product_isf.setConstant(True)

    product_fcrn_free_from_bbb = reaction_bbb_to_isf.createProduct()
    product_fcrn_free_from_bbb.setSpecies("FcRn_free_BBB")
    product_fcrn_free_from_bbb.setStoichiometry(1.0)
    product_fcrn_free_from_bbb.setConstant(True)

    
    # Kinetic law: CLup_brain * f_BBB * VES_brain * (1-FR) * PK_BBB_bound_brain
    klaw_bbb_to_isf = reaction_bbb_to_isf.createKineticLaw()
    math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * (1-FR) * PK_BBB_bound_brain")
    klaw_bbb_to_isf.setMath(math_ast)

    # 8. BCSFB Unbound processes

    # Uptake from TFV to BCSFB unbound
    reaction_tfv_to_bcsfb = model.createReaction()
    reaction_tfv_to_bcsfb.setId("tfv_to_bcsfb_unbound")
    reaction_tfv_to_bcsfb.setReversible(False)

    reactant_tfv = reaction_tfv_to_bcsfb.createReactant()
    reactant_tfv.setSpecies("PK_TFV_brain")
    reactant_tfv.setStoichiometry(1.0)
    reactant_tfv.setConstant(True)

    product_bcsfb_unbound = reaction_tfv_to_bcsfb.createProduct()
    product_bcsfb_unbound.setSpecies("PK_BCSFB_unbound_brain")
    product_bcsfb_unbound.setStoichiometry(1.0)
    product_bcsfb_unbound.setConstant(True)

    # Kinetic law: (1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * PK_TFV_brain
    klaw_tfv_to_bcsfb = reaction_tfv_to_bcsfb.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * PK_TFV_brain")
    klaw_tfv_to_bcsfb.setMath(math_ast)

    # Uptake from LV to BCSFB unbound
    reaction_lv_to_bcsfb = model.createReaction()
    reaction_lv_to_bcsfb.setId("lv_to_bcsfb_unbound")
    reaction_lv_to_bcsfb.setReversible(False)

    reactant_lv = reaction_lv_to_bcsfb.createReactant()
    reactant_lv.setSpecies("PK_LV_brain")
    reactant_lv.setStoichiometry(1.0)
    reactant_lv.setConstant(True)

    product_bcsfb_unbound_from_lv = reaction_lv_to_bcsfb.createProduct()
    product_bcsfb_unbound_from_lv.setSpecies("PK_BCSFB_unbound_brain")
    product_bcsfb_unbound_from_lv.setStoichiometry(1.0)
    product_bcsfb_unbound_from_lv.setConstant(True)

    # Kinetic law: f_LV * CLup_brain * (1 - f_BBB) * VES_brain * PK_LV_brain
    klaw_lv_to_bcsfb = reaction_lv_to_bcsfb.createKineticLaw()
    math_ast = libsbml.parseL3Formula("f_LV * CLup_brain * (1 - f_BBB) * VES_brain * PK_LV_brain")
    klaw_lv_to_bcsfb.setMath(math_ast)

    # Uptake from brain plasma to BCSFB unbound
    reaction_plasma_to_bcsfb = model.createReaction()
    reaction_plasma_to_bcsfb.setId("plasma_to_bcsfb_unbound")
    reaction_plasma_to_bcsfb.setReversible(False)

    reactant_plasma_bcsfb = reaction_plasma_to_bcsfb.createReactant()
    reactant_plasma_bcsfb.setSpecies("PK_p_brain")
    reactant_plasma_bcsfb.setStoichiometry(1.0)
    reactant_plasma_bcsfb.setConstant(True)

    product_bcsfb_unbound_from_plasma = reaction_plasma_to_bcsfb.createProduct()
    product_bcsfb_unbound_from_plasma.setSpecies("PK_BCSFB_unbound_brain")
    product_bcsfb_unbound_from_plasma.setStoichiometry(1.0)
    product_bcsfb_unbound_from_plasma.setConstant(True)

    # Kinetic law: CLup_brain * f_BCSFB * VES_brain * PK_p_brain
    klaw_plasma_to_bcsfb = reaction_plasma_to_bcsfb.createKineticLaw()
    math_ast = libsbml.parseL3Formula("CLup_brain * f_BCSFB * VES_brain * PK_p_brain") # Could be part of reaction Uptake from brain plasma to BBB/BCSFB
    klaw_plasma_to_bcsfb.setMath(math_ast)

    # Degradation of BCSFB unbound
    reaction_bcsfb_unbound_degradation = model.createReaction()
    reaction_bcsfb_unbound_degradation.setId("bcsfb_unbound_degradation")
    reaction_bcsfb_unbound_degradation.setReversible(False)

    reactant_bcsfb_unbound_deg = reaction_bcsfb_unbound_degradation.createReactant()
    reactant_bcsfb_unbound_deg.setSpecies("PK_BCSFB_unbound_brain")
    reactant_bcsfb_unbound_deg.setStoichiometry(1.0)
    reactant_bcsfb_unbound_deg.setConstant(True)

    # Add Sink as product
    product_sink = reaction_bcsfb_unbound_degradation.createProduct()
    product_sink.setSpecies(sinks["PK_BCSFB_unbound_brain"])
    product_sink.setStoichiometry(1.0)
    product_sink.setConstant(True)

    # Kinetic law: kdeg * PK_BCSFB_unbound_brain * V_BCSFB_brain
    klaw_bcsfb_unbound_deg = reaction_bcsfb_unbound_degradation.createKineticLaw()
    math_ast = libsbml.parseL3Formula("kdeg * PK_BCSFB_unbound_brain * V_BCSFB_brain")
    klaw_bcsfb_unbound_deg.setMath(math_ast)

    # 9. BCSFB FcRn binding processes

    # Binding of BCSFB unbound to FcRn
    reaction_bcsfb_fcrn_binding = model.createReaction()
    reaction_bcsfb_fcrn_binding.setId("bcsfb_fcrn_binding")
    reaction_bcsfb_fcrn_binding.setReversible(False)

    reactant_bcsfb_unbound_bind = reaction_bcsfb_fcrn_binding.createReactant()
    reactant_bcsfb_unbound_bind.setSpecies("PK_BCSFB_unbound_brain")
    reactant_bcsfb_unbound_bind.setStoichiometry(1.0)
    reactant_bcsfb_unbound_bind.setConstant(True)

    reactant_fcrn_free_bind = reaction_bcsfb_fcrn_binding.createReactant()
    reactant_fcrn_free_bind.setSpecies("FcRn_free_BCSFB")
    reactant_fcrn_free_bind.setStoichiometry(1.0)
    reactant_fcrn_free_bind.setConstant(True)

    product_bcsfb_bound = reaction_bcsfb_fcrn_binding.createProduct()
    product_bcsfb_bound.setSpecies("PK_BCSFB_bound_brain")
    product_bcsfb_bound.setStoichiometry(1.0)
    product_bcsfb_bound.setConstant(True)

    # Kinetic law: kon_FcRn * PK_BCSFB_unbound_brain * FcRn_free_BCSFB * V_BCSFB_brain
    klaw_bcsfb_binding = reaction_bcsfb_fcrn_binding.createKineticLaw()
    math_ast = libsbml.parseL3Formula("kon_FcRn * PK_BCSFB_unbound_brain * FcRn_free_BCSFB * V_BCSFB_brain")
    klaw_bcsfb_binding.setMath(math_ast)

    # Unbinding of BCSFB bound from FcRn - this is a special case as it affects multiple species
    # We'll model this as a single reaction that affects all relevant species
    reaction_bcsfb_fcrn_unbinding = model.createReaction()
    reaction_bcsfb_fcrn_unbinding.setId("bcsfb_fcrn_unbinding")
    reaction_bcsfb_fcrn_unbinding.setReversible(False)

    reactant_bcsfb_bound_unbind = reaction_bcsfb_fcrn_unbinding.createReactant()
    reactant_bcsfb_bound_unbind.setSpecies("PK_BCSFB_bound_brain")
    reactant_bcsfb_bound_unbind.setStoichiometry(1.0)
    reactant_bcsfb_bound_unbind.setConstant(True)

    product_bcsfb_unbound = reaction_bcsfb_fcrn_unbinding.createProduct()
    product_bcsfb_unbound.setSpecies("PK_BCSFB_unbound_brain")
    product_bcsfb_unbound.setStoichiometry(1.0)
    product_bcsfb_unbound.setConstant(True)

    product_fcrn_free = reaction_bcsfb_fcrn_unbinding.createProduct()
    product_fcrn_free.setSpecies("FcRn_free_BCSFB")
    product_fcrn_free.setStoichiometry(1.0)
    product_fcrn_free.setConstant(True)

    # Kinetic law: koff_FcRn * PK_BCSFB_bound_brain
    klaw_bcsfb_unbinding = reaction_bcsfb_fcrn_unbinding.createKineticLaw()
    math_ast = libsbml.parseL3Formula("koff_FcRn * PK_BCSFB_bound_brain * V_BCSFB_brain")
    klaw_bcsfb_unbinding.setMath(math_ast)

    # 10. BCSFB Bound processes

    reaction_bcsfb_to_bbb_fcrn = model.createReaction()
    reaction_bcsfb_to_bbb_fcrn.setId("bcsfb_to_brain_plasma")
    reaction_bcsfb_to_bbb_fcrn.setReversible(False)

    reactant_bcsfb_bound_bbb = reaction_bcsfb_to_bbb_fcrn.createReactant()
    reactant_bcsfb_bound_bbb.setSpecies("PK_BCSFB_bound_brain")
    reactant_bcsfb_bound_bbb.setStoichiometry(1.0)
    reactant_bcsfb_bound_bbb.setConstant(True)

    product_brain_plasma_from_bcsfb = reaction_bcsfb_to_bbb_fcrn.createProduct()
    product_brain_plasma_from_bcsfb.setSpecies("PK_p_brain")
    product_brain_plasma_from_bcsfb.setStoichiometry(1.0)
    product_brain_plasma_from_bcsfb.setConstant(True)

    product_fcrn_free_from_bcsfb = reaction_bcsfb_to_bbb_fcrn.createProduct()
    product_fcrn_free_from_bcsfb.setSpecies("FcRn_free_BCSFB")
    product_fcrn_free_from_bcsfb.setStoichiometry(1.0)
    product_fcrn_free_from_bcsfb.setConstant(True)


    # Kinetic law: CLup_brain * f_BBB * VES_brain * FR * PK_BCSFB_bound_brain
    klaw_bcsfb_to_bbb_fcrn = reaction_bcsfb_to_bbb_fcrn.createKineticLaw()
    math_ast = libsbml.parseL3Formula("CLup_brain * (1-f_BBB) * VES_brain * FR * PK_BCSFB_bound_brain") 
    klaw_bcsfb_to_bbb_fcrn.setMath(math_ast)


    # Transcytosis of BCSFB bound to LV (non-recycled fraction)
    reaction_bcsfb_to_lv = model.createReaction()
    reaction_bcsfb_to_lv.setId("bcsfb_bound_to_lv")
    reaction_bcsfb_to_lv.setReversible(False)

    reactant_bcsfb_bound_lv = reaction_bcsfb_to_lv.createReactant()
    reactant_bcsfb_bound_lv.setSpecies("PK_BCSFB_bound_brain")
    reactant_bcsfb_bound_lv.setStoichiometry(1.0)
    reactant_bcsfb_bound_lv.setConstant(True)

    product_lv = reaction_bcsfb_to_lv.createProduct()
    product_lv.setSpecies("PK_LV_brain")
    product_lv.setStoichiometry(1.0)
    product_lv.setConstant(True)

    product_fcrn_free_from_bcsfb = reaction_bcsfb_to_lv.createProduct()
    product_fcrn_free_from_bcsfb.setSpecies("FcRn_free_BCSFB")
    product_fcrn_free_from_bcsfb.setStoichiometry(1.0)
    product_fcrn_free_from_bcsfb.setConstant(True)

    
    # Kinetic law: f_LV * CLup_brain * (1-f_BBB) * VES_brain * (1-FR) * PK_BCSFB_bound_brain
    klaw_bcsfb_to_lv = reaction_bcsfb_to_lv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("f_LV * CLup_brain * (1-f_BBB) * VES_brain * (1-FR) * PK_BCSFB_bound_brain")
    klaw_bcsfb_to_lv.setMath(math_ast)

    # Transcytosis of BCSFB bound to TFV (non-recycled fraction)
    reaction_bcsfb_to_tfv = model.createReaction()
    reaction_bcsfb_to_tfv.setId("bcsfb_bound_to_tfv")
    reaction_bcsfb_to_tfv.setReversible(False)

    reactant_bcsfb_bound_tfv = reaction_bcsfb_to_tfv.createReactant()
    reactant_bcsfb_bound_tfv.setSpecies("PK_BCSFB_bound_brain")
    reactant_bcsfb_bound_tfv.setStoichiometry(1.0)
    reactant_bcsfb_bound_tfv.setConstant(True)

    product_tfv = reaction_bcsfb_to_tfv.createProduct()
    product_tfv.setSpecies("PK_TFV_brain")
    product_tfv.setStoichiometry(1.0)
    product_tfv.setConstant(True)

    product_fcrn_free_from_bcsfb = reaction_bcsfb_to_tfv.createProduct()
    product_fcrn_free_from_bcsfb.setSpecies("FcRn_free_BCSFB")
    product_fcrn_free_from_bcsfb.setStoichiometry(1.0)
    product_fcrn_free_from_bcsfb.setConstant(True)

    

    # Kinetic law: (1-f_LV) * CLup_brain * (1-f_BBB) * VES_brain * (1-FR) * PK_BCSFB_bound_brain
    klaw_bcsfb_to_tfv = reaction_bcsfb_to_tfv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1-f_LV) * CLup_brain * (1-f_BBB) * VES_brain * (1-FR) * PK_BCSFB_bound_brain")
    klaw_bcsfb_to_tfv.setMath(math_ast)

    # 11. CSF flow processes

    # LV to TFV flow
    reaction_lv_to_tfv = model.createReaction()
    reaction_lv_to_tfv.setId("lv_to_tfv_flow")
    reaction_lv_to_tfv.setReversible(False)

    reactant_lv_flow = reaction_lv_to_tfv.createReactant()
    reactant_lv_flow.setSpecies("PK_LV_brain")
    reactant_lv_flow.setStoichiometry(1.0)
    reactant_lv_flow.setConstant(True)

    product_tfv_from_lv = reaction_lv_to_tfv.createProduct()
    product_tfv_from_lv.setSpecies("PK_TFV_brain")
    product_tfv_from_lv.setStoichiometry(1.0)
    product_tfv_from_lv.setConstant(True)

    # Kinetic law: f_LV * (Q_CSF_brain + Q_ISF_brain) * PK_LV_brain
    klaw_lv_to_tfv = reaction_lv_to_tfv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("f_LV * (Q_CSF_brain + Q_ISF_brain) * PK_LV_brain")
    klaw_lv_to_tfv.setMath(math_ast)

    # ISF to LV flow
    reaction_isf_to_lv = model.createReaction()
    reaction_isf_to_lv.setId("isf_to_lv_flow")
    reaction_isf_to_lv.setReversible(False)

    reactant_isf_lv = reaction_isf_to_lv.createReactant()
    reactant_isf_lv.setSpecies("Ab_t")
    reactant_isf_lv.setStoichiometry(1.0)
    reactant_isf_lv.setConstant(True)

    product_lv_from_isf = reaction_isf_to_lv.createProduct()
    product_lv_from_isf.setSpecies("PK_LV_brain")
    product_lv_from_isf.setStoichiometry(1.0)
    product_lv_from_isf.setConstant(True)

    # Kinetic law: f_LV * Q_ISF_brain * Ab_t
    klaw_isf_to_lv = reaction_isf_to_lv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("f_LV * Q_ISF_brain * Ab_t")
    klaw_isf_to_lv.setMath(math_ast)

    # Brain plasma to LV flow (reflection)
    reaction_plasma_to_lv = model.createReaction()
    reaction_plasma_to_lv.setId("plasma_to_lv_flow")
    reaction_plasma_to_lv.setReversible(False)

    reactant_plasma_lv = reaction_plasma_to_lv.createReactant()
    reactant_plasma_lv.setSpecies("PK_p_brain")
    reactant_plasma_lv.setStoichiometry(1.0)
    reactant_plasma_lv.setConstant(True)

    product_lv_from_plasma = reaction_plasma_to_lv.createProduct()
    product_lv_from_plasma.setSpecies("PK_LV_brain")
    product_lv_from_plasma.setStoichiometry(1.0)
    product_lv_from_plasma.setConstant(True)

    # Kinetic law: (1-sigma_V_BCSFB) * f_LV * Q_CSF_brain * PK_p_brain
    klaw_plasma_to_lv = reaction_plasma_to_lv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1-sigma_V_BCSFB) * f_LV * Q_CSF_brain * PK_p_brain")
    klaw_plasma_to_lv.setMath(math_ast)

    # 12. TFV to CM flow
    reaction_tfv_to_cm = model.createReaction()
    reaction_tfv_to_cm.setId("tfv_to_cm_flow")
    reaction_tfv_to_cm.setReversible(False)

    reactant_tfv_flow = reaction_tfv_to_cm.createReactant()
    reactant_tfv_flow.setSpecies("PK_TFV_brain")
    reactant_tfv_flow.setStoichiometry(1.0)
    reactant_tfv_flow.setConstant(True)

    product_cm_from_tfv = reaction_tfv_to_cm.createProduct()
    product_cm_from_tfv.setSpecies("PK_CM_brain")
    product_cm_from_tfv.setStoichiometry(1.0)
    product_cm_from_tfv.setConstant(True)

    # Kinetic law: (Q_CSF_brain + Q_ISF_brain) * PK_TFV_brain
    klaw_tfv_to_cm = reaction_tfv_to_cm.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(Q_CSF_brain + Q_ISF_brain) * PK_TFV_brain")
    klaw_tfv_to_cm.setMath(math_ast)

    # Brain plasma to TFV flow (reflection)
    reaction_plasma_to_tfv = model.createReaction()
    reaction_plasma_to_tfv.setId("plasma_to_tfv_flow")
    reaction_plasma_to_tfv.setReversible(False)

    reactant_plasma_tfv = reaction_plasma_to_tfv.createReactant()
    reactant_plasma_tfv.setSpecies("PK_p_brain")
    reactant_plasma_tfv.setStoichiometry(1.0)
    reactant_plasma_tfv.setConstant(True)

    product_tfv_from_plasma = reaction_plasma_to_tfv.createProduct()
    product_tfv_from_plasma.setSpecies("PK_TFV_brain")
    product_tfv_from_plasma.setStoichiometry(1.0)
    product_tfv_from_plasma.setConstant(True)

    # Kinetic law: (1-sigma_V_brain_ISF) * (1 - f_LV) * Q_ISF_brain * PK_p_brain
    klaw_plasma_to_tfv = reaction_plasma_to_tfv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1-sigma_V_brain_ISF) * (1 - f_LV) * Q_ISF_brain * PK_p_brain")
    klaw_plasma_to_tfv.setMath(math_ast)


    # 13. CM to SAS flow
    reaction_cm_to_sas = model.createReaction()
    reaction_cm_to_sas.setId("cm_to_sas_flow")
    reaction_cm_to_sas.setReversible(False)

    reactant_cm_flow = reaction_cm_to_sas.createReactant()
    reactant_cm_flow.setSpecies("PK_CM_brain")
    reactant_cm_flow.setStoichiometry(1.0)
    reactant_cm_flow.setConstant(True)

    product_sas_from_cm = reaction_cm_to_sas.createProduct()
    product_sas_from_cm.setSpecies("PK_SAS_brain")
    product_sas_from_cm.setStoichiometry(1.0)
    product_sas_from_cm.setConstant(True)

    # Kinetic law: (Q_CSF_brain + Q_ISF_brain) * PK_CM_brain
    klaw_cm_to_sas = reaction_cm_to_sas.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(Q_CSF_brain + Q_ISF_brain) * PK_CM_brain")
    klaw_cm_to_sas.setMath(math_ast)

    # 14. SAS outflow processes

    # SAS to ISF flow
    reaction_sas_to_isf = model.createReaction()
    reaction_sas_to_isf.setId("sas_to_isf_flow")
    reaction_sas_to_isf.setReversible(False)

    reactant_sas_isf = reaction_sas_to_isf.createReactant()
    reactant_sas_isf.setSpecies("PK_SAS_brain")
    reactant_sas_isf.setStoichiometry(1.0)
    reactant_sas_isf.setConstant(True)

    product_isf_from_sas = reaction_sas_to_isf.createProduct()
    product_isf_from_sas.setSpecies("Ab_t")
    product_isf_from_sas.setStoichiometry(1.0)
    product_isf_from_sas.setConstant(True)

    # Kinetic law: Q_ISF_brain * PK_SAS_brain
    klaw_sas_to_isf = reaction_sas_to_isf.createKineticLaw()
    math_ast = libsbml.parseL3Formula("Q_ISF_brain * PK_SAS_brain")
    klaw_sas_to_isf.setMath(math_ast)

    
    # 15. ISF outflow processes

    # ISF to TFV flow
    reaction_isf_to_tfv = model.createReaction()
    reaction_isf_to_tfv.setId("isf_to_tfv_flow")
    reaction_isf_to_tfv.setReversible(False)

    reactant_isf_tfv = reaction_isf_to_tfv.createReactant()
    reactant_isf_tfv.setSpecies("Ab_t")
    reactant_isf_tfv.setStoichiometry(1.0)
    reactant_isf_tfv.setConstant(True)

    product_tfv_from_isf = reaction_isf_to_tfv.createProduct()
    product_tfv_from_isf.setSpecies("PK_TFV_brain")
    product_tfv_from_isf.setStoichiometry(1.0)
    product_tfv_from_isf.setConstant(True)

    # Kinetic law: (1-f_LV) * Q_ISF_brain * Ab_t
    klaw_isf_to_tfv = reaction_isf_to_tfv.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1-f_LV) * Q_ISF_brain * Ab_t")
    klaw_isf_to_tfv.setMath(math_ast)

    # ISF to central compartment (reflection)
    reaction_isf_to_central = model.createReaction()
    reaction_isf_to_central.setId("isf_to_central_reflection")
    reaction_isf_to_central.setReversible(False)

    reactant_isf_central = reaction_isf_to_central.createReactant()
    reactant_isf_central.setSpecies("Ab_t")
    reactant_isf_central.setStoichiometry(1.0)
    reactant_isf_central.setConstant(True)

    product_central_from_isf = reaction_isf_to_central.createProduct()
    product_central_from_isf.setSpecies("PK_central")
    product_central_from_isf.setStoichiometry(1.0)
    product_central_from_isf.setConstant(True)

    # Kinetic law: (1-sigma_L_brain_ISF) * Q_ISF_brain * Ab_t
    klaw_isf_to_central = reaction_isf_to_central.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1-sigma_L_brain_ISF) * (Q_ISF_brain - Q_PVS) * Ab_t")
    klaw_isf_to_central.setMath(math_ast)

    # 16. Brain plasma to ISF (reflection)
    reaction_plasma_to_isf = model.createReaction()
    reaction_plasma_to_isf.setId("plasma_to_isf_reflection")
    reaction_plasma_to_isf.setReversible(False)

    reactant_plasma_isf = reaction_plasma_to_isf.createReactant()
    reactant_plasma_isf.setSpecies("PK_p_brain")
    reactant_plasma_isf.setStoichiometry(1.0)
    reactant_plasma_isf.setConstant(True)

    product_isf_from_plasma = reaction_plasma_to_isf.createProduct()
    product_isf_from_plasma.setSpecies("Ab_t")
    product_isf_from_plasma.setStoichiometry(1.0)
    product_isf_from_plasma.setConstant(True)

    # Kinetic law: (1-sigma_V_brain_ISF) * Q_ISF_brain * PK_p_brain
    klaw_plasma_to_isf = reaction_plasma_to_isf.createKineticLaw()
    math_ast = libsbml.parseL3Formula("(1-sigma_V_brain_ISF) * Q_ISF_brain * PK_p_brain")
    klaw_plasma_to_isf.setMath(math_ast)
    

    # Add assignment rules for InputCent and InputSC
    # These rules will implement the dosing schedule using piecewise functions
    
    # For IV dosing (InputCent)
    iv_rule = model.createAssignmentRule()
    iv_rule.setId("input_cent_assignment_rule")  # Add unique identifier
    iv_rule.setVariable("InputCent")
    
    # Create a more scalable function for IV dosing using modulo arithmetic
    # This will work for any number of doses
    iv_math_formula = """
    piecewise(
        IV_DoseAmount / IV_DoseDuration,
        (time < MaxDosingTime) && 
        (floor(time / IV_DoseInterval) < IV_NumDoses) && 
        ((time - floor(time / IV_DoseInterval) * IV_DoseInterval) < IV_DoseDuration),
        0
    )
    """
    iv_math_ast = libsbml.parseL3Formula(iv_math_formula)
    iv_rule.setMath(iv_math_ast)
    
    # For SC dosing (InputSC)
    sc_rule = model.createAssignmentRule()
    sc_rule.setId("input_sc_assignment_rule")  # Add unique identifier
    sc_rule.setVariable("InputSC")
    
    # Create a more scalable function for SC dosing using modulo arithmetic
    sc_math_formula = """
    piecewise(
        SC_DoseAmount / SC_DoseDuration,
        (time < MaxDosingTime) && 
        (floor(time / SC_DoseInterval) < SC_NumDoses) && 
        ((time - floor(time / SC_DoseInterval) * SC_DoseInterval) < SC_DoseDuration),
        0
    )
    """
    sc_math_ast = libsbml.parseL3Formula(sc_math_formula)
    sc_rule.setMath(sc_math_ast)
    
    # Add unit definitions for nanomolar if needed
    # Nanomole per liter (concentration)
    nano_conc = model.createUnitDefinition()
    nano_conc.setId("nanomole_per_litre")
    nano_unit = nano_conc.createUnit()
    nano_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nano_unit.setScale(-9)  # nano
    nano_unit.setExponent(1)
    nano_unit.setMultiplier(1.0)  # Add missing multiplier
    litre_unit = nano_conc.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(-1)
    litre_unit.setScale(0)  # Add missing scale
    litre_unit.setMultiplier(1.0)  # Add missing multiplier
    
    # Nanomole (amount)
    nano_amount = model.createUnitDefinition()
    nano_amount.setId("nanomole")
    nano_unit = nano_amount.createUnit()
    nano_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nano_unit.setScale(-9)  # nano
    nano_unit.setExponent(1)
    nano_unit.setMultiplier(1.0)  # Add missing multiplier
    
    # Nanomole per hour (flow)
    nano_flow = model.createUnitDefinition()
    nano_flow.setId("nanomole_per_hour")
    nano_unit = nano_flow.createUnit()
    nano_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nano_unit.setScale(-9)  # nano
    nano_unit.setExponent(1)
    nano_unit.setMultiplier(1.0)  # Add missing multiplier
    hour_unit = nano_flow.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setScale(0)  # Add missing scale
    hour_unit.setMultiplier(3600)  # seconds in an hour
    hour_unit.setExponent(-1)
    
    # Per nanomole per hour (for 2nd order rate constants)
    per_nano = model.createUnitDefinition()
    per_nano.setId("per_nanomole_per_hour")
    nano_unit = per_nano.createUnit()
    nano_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nano_unit.setScale(-9)  # nano
    nano_unit.setExponent(-1)
    nano_unit.setMultiplier(1.0)  # Add missing multiplier
    hour_unit = per_nano.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setScale(0)  # Add missing scale
    hour_unit.setMultiplier(3600)  # seconds in an hour
    hour_unit.setExponent(-1)
    
    
    return document

def save_model(document, filename):
    """Save SBML model to file with validation"""
    # Check for errors
    if document.getNumErrors() > 0:
        print("\nValidation errors:")
        document.printErrors()
        return False
        
    # Additional validation
    print("\nValidating SBML model...")
    print(f"Number of compartments: {document.getModel().getNumCompartments()}")
    print(f"Number of species: {document.getModel().getNumSpecies()}")
    print(f"Number of parameters: {document.getModel().getNumParameters()}")
    print(f"Number of reactions: {document.getModel().getNumReactions()}")
    
    # Save the file
    result = libsbml.writeSBMLToFile(document, filename)
    if result:
        print(f"\nGeerts PK model saved successfully to {filename}!")
        return True
    else:
        print(f"\nError: Unable to save SBML file to {filename}")
        return False

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
    sink.setCompartment("comp_Central_compartment")
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
    source.setCompartment("comp_Central_compartment")
    source.setInitialConcentration(1.0)  # Fixed concentration
    source.setConstant(True)
    source.setHasOnlySubstanceUnits(False)
    source.setBoundaryCondition(True)  # Boundary species
    source.setSubstanceUnits("nanomole_per_litre")
    return source_id

def main():
    # Load parameters
    params_path = Path("parameters/PK_Geerts.csv")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found at {params_path}")
    
    # Default to lecanemab for testing
    drug_type = "lecanemab"
    params, params_with_units = load_parameters(params_path, drug_type=drug_type)
    
    # Configure dosing parameters based on drug type
    is_lecanemab = drug_type.lower() == "lecanemab"
    
    if is_lecanemab:
        # Lecanemab: IV dosing (10 mg/kg, assuming 70kg patient)
        params['IV_DoseAmount'] = 1550.50 * params['Vcent']  # nM
        params['IV_NumDoses'] = 1  # Single dose for testing
        params['IV_DoseDuration'] = 1.0  # 1 hour
        params['IV_DoseInterval'] = 336.0  # 2 weeks
        params['SC_NumDoses'] = 0  # No SC doses for lecanemab
    else:
        # Gantenerumab: SC dosing (300 mg)
        params['SC_DoseAmount'] = 2050.6  # nM
        params['SC_NumDoses'] = 1  # Single dose for testing
        params['SC_DoseDuration'] = 1.0  # 1 hour
        params['SC_DoseInterval'] = 672.0  # 4 weeks
        params['IV_NumDoses'] = 0  # No IV doses for gantenerumab
    
    # Create output directory
    output_dir = Path("generated/sbml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "geerts_pk_reactions_model.xml"
    
    # Create and save model with nanomolar units
    document = create_geerts_model(params, params_with_units)
    if document.getNumErrors() != 0:
        print("Validation errors:")
        document.printErrors()
    else:
        save_model(document, str(output_path))
        print(f"Geerts PK reactions model saved successfully to {output_path}!")

if __name__ == "__main__":
    main()