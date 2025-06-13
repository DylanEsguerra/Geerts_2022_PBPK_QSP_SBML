"""
Module for modeling Aβ monomer pharmacokinetics in the brain.
This is a PBPK component that models the distribution and clearance of Aβ monomers.
The module includes:
- Monomer distribution across blood-brain barrier (BBB)
- Monomer distribution across blood-CSF barrier (BCSFB)
- Monomer clearance through various pathways
- Monomer production and degradation
- Transport and reflection coefficients for monomers
- Includes monomer_bound micorglia clearance
"""

# removed volume from synthesis rate
# No volume on central antibbody binding 
# changed reactant to product in CLup FcRn 
import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file
    
    Args:
        csv_path: Path to CSV file
        drug_type: Either "lecanemab" or "gantenerumab"
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create parameter dictionaries
    params = {}
    params_with_units = {}
    
    # Load all parameters from CSV
    print("\nLoading raw parameters from CSV:")
    for _, row in df.iterrows():
        name = row['name']
        value = float(row['value'])
        unit = row['units'] if 'units' in row else None
        params[name] = value
        params_with_units[name] = (value, unit) if unit else value
        if name in ['Gant_fta0', 'Lec_fta0']:  # Debug print for fta0 parameters
            print(f"Found {name} = {value} {unit}")
    
    # Handle drug-specific parameter mapping
    is_lecanemab = drug_type.lower() == "lecanemab"
    print(f"\nDrug type: {'lecanemab' if is_lecanemab else 'gantenerumab'}")
    
    # Map drug-specific parameters
    param_mapping = {
        'Vcent': 'Lec_Vcent' if is_lecanemab else 'Gant_Vcent',
        'Vper': 'Lec_Vper' if is_lecanemab else 'Gant_Vper',
        'fta0': 'Lec_fta0' if is_lecanemab else 'Gant_fta0',
        'PK_SC_ka': 'Lec_SC_ka' if is_lecanemab else 'Gant_SC_ka',
        'PK_SC_bio': 'Lec_SC_bio' if is_lecanemab else 'Gant_SC_bio',
        'PK_CL': 'Lec_CL' if is_lecanemab else 'Gant_CL',
        'PK_CLd2': 'Lec_CLd2' if is_lecanemab else 'Gant_CLd2',
    }
    
    # Add debug prints
    print("\nDrug-specific parameter mapping:")
    for generic_name, specific_name in param_mapping.items():
        if specific_name in params:
            params[generic_name] = params[specific_name]
            if specific_name in params_with_units:
                params_with_units[generic_name] = params_with_units[specific_name]
            print(f"  Success! Mapped {specific_name} -> {generic_name}: {params[generic_name]}")
        else:
            print(f"  Warning: {specific_name} not found in parameters")

    return params, params_with_units

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
    sink.setCompartment("comp_Central")
    sink.setInitialConcentration(1.0)  # Fixed concentration
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
    source.setCompartment("comp_Central")
    source.setInitialConcentration(1.0)  # Fixed concentration
    source.setConstant(True)
    source.setHasOnlySubstanceUnits(False)
    source.setBoundaryCondition(True)  # Boundary species
    source.setSubstanceUnits("nanomole_per_litre")
    return source_id

def create_parameterized_model(params, params_with_units, drug_type="gantenerumab"):
    """Create parameterized Geerts model for monomers
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print("\nCreating parameterized Geerts model for monomers...")
    print("Number of parameters loaded:", len(params))
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_PBPK_monomers_Model")
    model.setTimeUnits("hour")
    
    # Add unit definitions
    # Mole per liter (concentration)
    nanomole_conc = model.createUnitDefinition()
    nanomole_conc.setId("nanomole_per_litre")
    
    # Add nanomole unit
    nanomole_unit = nanomole_conc.createUnit()
    nanomole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nanomole_unit.setScale(-9)  # nano = 10^-9
    nanomole_unit.setExponent(1.0)
    nanomole_unit.setMultiplier(1.0)
    
    # Add litre unit
    litre_unit = nanomole_conc.createUnit()  # Use nanomole_conc instead of mole_conc
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(-1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)
    
    # Mole (amount)
    nanomole_amount = model.createUnitDefinition()
    nanomole_amount.setId("nanomole")
    nanomole_unit = nanomole_amount.createUnit()
    nanomole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nanomole_unit.setExponent(1)
    nanomole_unit.setScale(0)
    nanomole_unit.setMultiplier(1.0)
    
    # Mole per hour (flow)
    mole_flow = model.createUnitDefinition()
    mole_flow.setId("mole_per_hour")
    mole_unit = mole_flow.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(1)
    mole_unit.setScale(0)
    mole_unit.setMultiplier(1.0)
    hour_unit = mole_flow.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(3600)
    hour_unit.setExponent(-1)
    
    # Per nanomole per hour (for 2nd order rate constants)
    per_nanomole = model.createUnitDefinition()
    per_nanomole.setId("per_nanomole_per_hour")
    nanomole_unit = per_nanomole.createUnit()
    nanomole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    nanomole_unit.setExponent(-1)
    nanomole_unit.setScale(0)
    nanomole_unit.setMultiplier(1.0)
    hour_unit = per_nanomole.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(3600)
    hour_unit.setExponent(-1)

    # Add litre_per_hour unit definition
    litre_per_hour = model.createUnitDefinition()
    litre_per_hour.setId("litre_per_hour")
    litre_unit = litre_per_hour.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)
    hour_unit = litre_per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

 

    # Create compartments
    print("\nCreating compartments...")
    compartments = {
        # PK compartments
        "comp_Central": params["Vcent"],
        "comp_Peripheral": params["Vper"],
        "comp_SubCut_absorption_compartment": params["V_SubCut"],
        "comp_Brain_plasma": params["Vp_brain"],
        "comp_BBB_Unbound": params["VBBB_brain"],
        "comp_BBB_Bound": params["VBBB_brain"],
        "comp_BCSFB_Unbound": params["V_BCSFB_brain"],
        "comp_BCSFB_Bound": params["V_BCSFB_brain"],
        "comp_CSF_LV": params["V_LV_brain"],
        "comp_CSF_TFV": params["V_TFV_brain"],
        "comp_CSF_CM": params["V_CM_brain"],
        "comp_CSF_SAS": params["V_SAS_brain"],
        "comp_ISF_brain": params["VIS_brain"],
        "comp_ES_brain": params["VES_brain"],
        
        # Monomer compartments
        "comp_AB40_monomer": params["VIS_brain"],
        "comp_AB42_monomer": params["VIS_brain"],
        
        # AB40Mu compartments
        "comp_AB40Mu_Central": params["V_AB40Mu_Central"],
        "comp_AB40Mu_Peripheral": params["V_AB40Mu_Peripheral"],
        "comp_AB40Mu_Brain_Plasma": params["Vp_brain"],
        "comp_AB40Mu_BCSFB": params["V_BCSFB_brain"],
        "comp_AB40Mu_BBB": params["VBBB_brain"],
        "comp_AB40Mu_LV": params["V_LV_brain"],
        "comp_AB40Mu_TFV": params["V_TFV_brain"],
        "comp_AB40Mu_CM": params["V_CM_brain"],
        "comp_AB40Mu_SAS": params["V_SAS_brain"],
        
        # AB40Mb compartments
        "comp_AB40Mb_Central": params["Vcent"],
        "comp_AB40Mb_Peripheral": params["Vper"],
        "comp_AB40Mb_Brain_Plasma": params["Vp_brain"],
        "comp_AB40Mb_BCSFB_Unbound": params["V_BCSFB_brain"],
        "comp_AB40Mb_BCSFB_Bound": params["V_BCSFB_brain"],
        "comp_AB40Mb_BBB_Unbound": params["VBBB_brain"],
        "comp_AB40Mb_BBB_Bound": params["VBBB_brain"],
        "comp_AB40_monomer_antibody_bound": params["VIS_brain"],
        "comp_AB40Mb_LV": params["V_LV_brain"],
        "comp_AB40Mb_TFV": params["V_TFV_brain"],
        "comp_AB40Mb_CM": params["V_CM_brain"],
        "comp_AB40Mb_SAS": params["V_SAS_brain"],

        # AB42Mu compartments
        "comp_AB42Mu_Central": params["V_AB42Mu_Central"],
        "comp_AB42Mu_Peripheral": params["V_AB42Mu_Peripheral"],
        "comp_AB42Mu_Brain_Plasma": params["Vp_brain"],
        "comp_AB42Mu_BCSFB": params["V_BCSFB_brain"],
        "comp_AB42Mu_BBB": params["VBBB_brain"],
        "comp_AB42Mu_LV": params["V_LV_brain"],
        "comp_AB42Mu_TFV": params["V_TFV_brain"],
        "comp_AB42Mu_CM": params["V_CM_brain"],
        "comp_AB42Mu_SAS": params["V_SAS_brain"],

        # AB42Mb compartments
        "comp_AB42Mb_Central": params["Vcent"],
        "comp_AB42Mb_Peripheral": params["Vper"],
        "comp_AB42Mb_Brain_Plasma": params["Vp_brain"],
        "comp_AB42Mb_BCSFB_Unbound": params["V_BCSFB_brain"],
        "comp_AB42Mb_BCSFB_Bound": params["V_BCSFB_brain"],
        "comp_AB42Mb_BBB_Unbound": params["VBBB_brain"],
        "comp_AB42Mb_BBB_Bound": params["VBBB_brain"],
        "comp_AB42_monomer_antibody_bound": params["VIS_brain"],
        "comp_AB42Mb_LV": params["V_LV_brain"],
        "comp_AB42Mb_TFV": params["V_TFV_brain"],
        "comp_AB42Mb_CM": params["V_CM_brain"],
        "comp_AB42Mb_SAS": params["V_SAS_brain"],

        # Microglia compartment
        "comp_microglia": params.get("V_microglia", 1.0)  # Default to 1.0 if not specified
    }

    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)
        comp.setUnits('litre')

    # Create all parameters
    print("\nCreating parameters...")
    common_params = [
        # Microglia parameters
        #("Microglia_cell_count", params["Microglia_cell_count"]),
        ("Microglia_CL_high_AB40", params["Microglia_CL_high_AB40"]),
        ("Microglia_CL_low_AB40", params["Microglia_CL_low_AB40"]),
        ("Microglia_CL_high_AB42", params["Microglia_CL_high_AB42"]),
        ("Microglia_CL_low_AB42", params["Microglia_CL_low_AB42"]),
        #("Microglia_Hi_Fract", params["Microglia_Hi_Fract"]),

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
        ("FcRn_free_BBB_0", params["FcRn_free_BBB_0"]),
        ("FcRn_free_BCSFB_0", params["FcRn_free_BCSFB_0"]),
        
        # Reflection coefficients
        ("sigma_V_brain_ISF", params["sigma_V_brain_ISF"]),
        ("sigma_V_BCSFB", params["sigma_V_BCSFB"]),
        ("sigma_L_brain_ISF", params["sigma_L_brain_ISF"]),
        ("sigma_L_SAS", params["sigma_L_SAS"]),
        
        # Clearance
        ("CLup_brain", params["CLup_brain"]),
        ("CLup_brain_AB40Mu", params["CLup_brain_AB40Mu"]),
        ("CLup_brain_AB42Mu", params["CLup_brain_AB42Mu"]),

        # Monomer binding parameters
        ("fta0", params["fta0"]),
        
        # AB40Mu specific parameters
        ("AB40Mu_systemic_synthesis_rate", params["AB40Mu_systemic_synthesis_rate"]),
        ("sigma_V_BCSFB_AB40Mu", params["sigma_V_BCSFB_AB40Mu"]),
        ("sigma_V_ISF_AB40Mu", params["sigma_V_ISF_AB40Mu"]),
        ("sigma_L_SAS_AB40Mu", params["sigma_L_SAS_AB40Mu"]),
        ("sigma_L_ISF_central_AB40Mu", params["sigma_L_ISF_central_AB40Mu"]),
        ("kdeg_AB40Mu", params["kdeg_AB40Mu"]),


        ("V_AB40Mu_Central", params["V_AB40Mu_Central"]),
        ("V_AB40Mu_Peripheral", params["V_AB40Mu_Peripheral"]),

         # AB42Mu specific parameters
        ("AB42Mu_systemic_synthesis_rate", params["AB42Mu_systemic_synthesis_rate"]),       
        ("sigma_V_BCSFB_AB42Mu", params["sigma_V_BCSFB_AB42Mu"]),
        ("sigma_V_ISF_AB42Mu", params["sigma_V_ISF_AB42Mu"]),
        ("sigma_L_SAS_AB42Mu", params["sigma_L_SAS_AB42Mu"]),
        ("sigma_L_ISF_central_AB42Mu", params["sigma_L_ISF_central_AB42Mu"]),
        ("kdeg_AB42Mu", params["kdeg_AB42Mu"]),


        ("V_AB42Mu_Central", params["V_AB42Mu_Central"]),
        ("V_AB42Mu_Peripheral", params["V_AB42Mu_Peripheral"]),
    ]

    for param_id, value in common_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Set appropriate units
        if param_id.startswith('gamma'):
            param.setUnits("per_hour")
        elif param_id.startswith('EC50'):
            param.setUnits("nanomole_per_litre")
        elif param_id.startswith('V_'):
            param.setUnits("litre")
        elif param_id.startswith('Q_') or param_id.startswith('CL'):
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

    # Add clearance parameters
    clearance_params = [
        ("AB40MuCL", params["AB40MuCL"]),
        ("AB40MuCLd2", params["AB40MuCLd2"]),
        ("AB42MuCL", params["AB42MuCL"]),
        ("AB42MuCLd2", params["AB42MuCLd2"]),
    ]
    
    for param_id, value in clearance_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        param.setUnits("litre_per_hour")  # Use the new unit definition

    # Create species
    species = [
        # PK species
        ("PK_BBB_unbound_brain", "comp_BBB_Unbound", params["PK_BBB_unbound_brain_0"]),
        ("PK_BBB_bound_brain", "comp_BBB_Bound", params["PK_BBB_bound_brain_0"]),
        ("PK_p_brain", "comp_Brain_plasma", params["PK_p_brain_0"]),
        ("Ab_t", "comp_ISF_brain", params["Ab_t_0"]),
        ("PK_BCSFB_unbound_brain", "comp_BCSFB_Unbound", params["PK_BCSFB_unbound_brain_0"]),
        ("PK_BCSFB_bound_brain", "comp_BCSFB_Bound", params["PK_BCSFB_bound_brain_0"]),
        ("PK_LV_brain", "comp_CSF_LV", params["PK_LV_brain_0"]),
        ("PK_TFV_brain", "comp_CSF_TFV", params["PK_TFV_brain_0"]),
        ("PK_CM_brain", "comp_CSF_CM", params["PK_CM_brain_0"]),
        ("PK_SAS_brain", "comp_CSF_SAS", params["PK_SAS_brain_0"]),
        ("PK_central", "comp_Central", params["PK_central_0"]),
        ("PK_per", "comp_Peripheral", params["PK_per_0"]),
        ("SubCut_absorption", "comp_SubCut_absorption_compartment", params["SubCut_absorption_0"]),

        # Microglia species
        ("Microglia_Hi_Fract", "comp_microglia", params.get("Microglia_Hi_Fract_0", 0.0)),
        ("Microglia_cell_count", "comp_microglia", params.get("Microglia_cell_count_0", 0.0)),

        # Monomer species 
        ("AB42_Monomer", "comp_AB42_monomer", params["AB42_Monomer_0"]),
        ("AB40_Monomer", "comp_AB40_monomer", params["AB40_Monomer_0"]),
        # Monomer species - AB40 Unbound
        ("AB40Mu_Central", "comp_AB40Mu_Central", params["AB40Mu_Central_0"]),
        ("AB40Mu_Peripheral", "comp_AB40Mu_Peripheral", params["AB40Mu_Peripheral_0"]),
        ("AB40Mu_Brain_Plasma", "comp_AB40Mu_Brain_Plasma", params["AB40Mu_Brain_Plasma_0"]),
        ("AB40Mu_BCSFB", "comp_AB40Mu_BCSFB", params["AB40Mu_BCSFB_0"]),
        ("AB40Mu_BBB", "comp_AB40Mu_BBB", params["AB40Mu_BBB_0"]),
        ("AB40Mu_LV", "comp_AB40Mu_LV", params["AB40Mu_LV_0"]),
        ("AB40Mu_TFV", "comp_AB40Mu_TFV", params["AB40Mu_TFV_0"]),
        ("AB40Mu_CM", "comp_AB40Mu_CM", params["AB40Mu_CM_0"]),
        ("AB40Mu_SAS", "comp_AB40Mu_SAS", params["AB40Mu_SAS_0"]),
        
        # AB40 Bound to antibody
        ("AB40Mb_Central", "comp_AB40Mb_Central", params["AB40Mb_Central_0"]),
        ("AB40Mb_Peripheral", "comp_AB40Mb_Peripheral", params["AB40Mb_Peripheral_0"]),
        ("AB40Mb_Brain_Plasma", "comp_AB40Mb_Brain_Plasma", params["AB40Mb_Brain_Plasma_0"]),
        ("AB40Mb_BCSFB_Unbound", "comp_AB40Mb_BCSFB_Unbound", params["AB40Mb_BCSFB_Unbound_0"]),
        ("AB40Mb_BCSFB_Bound", "comp_AB40Mb_BCSFB_Bound", params["AB40Mb_BCSFB_Bound_0"]),
        ("AB40Mb_BBB_Unbound", "comp_AB40Mb_BBB_Unbound", params["AB40Mb_BBB_Unbound_0"]),
        ("AB40Mb_BBB_Bound", "comp_AB40Mb_BBB_Bound", params["AB40Mb_BBB_Bound_0"]),
        ("AB40_monomer_antibody_bound", "comp_AB40_monomer_antibody_bound", params["AB40_monomer_antibody_bound_0"]),
        ("AB40Mb_LV", "comp_AB40Mb_LV", params["AB40Mb_LV_0"]),
        ("AB40Mb_TFV", "comp_AB40Mb_TFV", params["AB40Mb_TFV_0"]),
        ("AB40Mb_CM", "comp_AB40Mb_CM", params["AB40Mb_CM_0"]),
        ("AB40Mb_SAS", "comp_AB40Mb_SAS", params["AB40Mb_SAS_0"]),
        
        # AB42 Unbound
        ("AB42Mu_Central", "comp_AB42Mu_Central", params["AB42Mu_Central_0"]),
        ("AB42Mu_Peripheral", "comp_AB42Mu_Peripheral", params["AB42Mu_Peripheral_0"]),
        ("AB42Mu_Brain_Plasma", "comp_AB42Mu_Brain_Plasma", params["AB42Mu_Brain_Plasma_0"]),
        ("AB42Mu_BCSFB", "comp_AB42Mu_BCSFB", params["AB42Mu_BCSFB_0"]),
        ("AB42Mu_BBB", "comp_AB42Mu_BBB", params["AB42Mu_BBB_0"]),
        ("AB42Mu_LV", "comp_AB42Mu_LV", params["AB42Mu_LV_0"]),
        ("AB42Mu_TFV", "comp_AB42Mu_TFV", params["AB42Mu_TFV_0"]),
        ("AB42Mu_CM", "comp_AB42Mu_CM", params["AB42Mu_CM_0"]),
        ("AB42Mu_SAS", "comp_AB42Mu_SAS", params["AB42Mu_SAS_0"]),
        
        # AB42 Bound to antibody
        ("AB42Mb_Central", "comp_AB42Mb_Central", params["AB42Mb_Central_0"]),
        ("AB42Mb_Peripheral", "comp_AB42Mb_Peripheral", params["AB42Mb_Peripheral_0"]),
        ("AB42Mb_Brain_Plasma", "comp_AB42Mb_Brain_Plasma", params["AB42Mb_Brain_Plasma_0"]),
        ("AB42Mb_BCSFB_Unbound", "comp_AB42Mb_BCSFB_Unbound", params["AB42Mb_BCSFB_Unbound_0"]),
        ("AB42Mb_BCSFB_Bound", "comp_AB42Mb_BCSFB_Bound", params["AB42Mb_BCSFB_Bound_0"]),
        ("AB42Mb_BBB_Unbound", "comp_AB42Mb_BBB_Unbound", params["AB42Mb_BBB_Unbound_0"]),
        ("AB42Mb_BBB_Bound", "comp_AB42Mb_BBB_Bound", params["AB42Mb_BBB_Bound_0"]),
        ("AB42_monomer_antibody_bound", "comp_AB42_monomer_antibody_bound", params["AB42_monomer_antibody_bound_0"]),
        ("AB42Mb_LV", "comp_AB42Mb_LV", params["AB42Mb_LV_0"]),
        ("AB42Mb_TFV", "comp_AB42Mb_TFV", params["AB42Mb_TFV_0"]),
        ("AB42Mb_CM", "comp_AB42Mb_CM", params["AB42Mb_CM_0"]),
        ("AB42Mb_SAS", "comp_AB42Mb_SAS", params["AB42Mb_SAS_0"]),

        
    ]

    for species_id, compartment_id, initial_value in species:
        spec = model.createSpecies()
        spec.setId(species_id)
        spec.setCompartment(compartment_id)
        spec.setInitialConcentration(initial_value)
        spec.setSubstanceUnits("nanomole_per_litre")
        spec.setHasOnlySubstanceUnits(False)  # Using concentrations
        spec.setBoundaryCondition(False)
        spec.setConstant(False)

     # Instead of parameters, create species for free FcRn
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

    # Create dictionaries to hold sink and source species
    sinks = {}
    sources = {}
    
    # Pre-create all sinks and sources needed for both AB types
    for ab_type in ["40", "42"]:
        # Use dictionary keys to store the species IDs
        sources[f"AB{ab_type}Mu_Central"] = create_source_for_species(f"AB{ab_type}Mu_Central", model)
        sinks[f"AB{ab_type}Mu_Central"] = create_sink_for_species(f"AB{ab_type}Mu_Central", model)
        sinks[f"AB{ab_type}Mu_BCSFB"] = create_sink_for_species(f"AB{ab_type}Mu_BCSFB", model)
        sinks[f"AB{ab_type}Mu_BBB"] = create_sink_for_species(f"AB{ab_type}Mu_BBB", model)
        sinks[f"AB{ab_type}Mb_Central"] = create_sink_for_species(f"AB{ab_type}Mb_Central", model)
        sinks[f"AB{ab_type}Mb_BCSFB_Unbound"] = create_sink_for_species(f"AB{ab_type}Mb_BCSFB_Unbound", model)
        sinks[f"AB{ab_type}Mb_BBB_Unbound"] = create_sink_for_species(f"AB{ab_type}Mb_BBB_Unbound", model)
        sinks[f"AB{ab_type}_monomer_antibody_bound"] = create_sink_for_species(f"AB{ab_type}_monomer_antibody_bound", model)

    def create_reactions(ab_type):
        """Create all species and reactions for a specific AB type (40 or 42)."""

        # Create reactions for this AB type
        # Define all reactions for this AB type
        print(f'Creating reactions for AB{ab_type}...')
        
        reaction_blocks = [
            # Reaction block 1
            f'''
# 1. Binding of antibody (PK_central) to AB{ab_type} monomer in central compartment
reaction_ab{ab_type}_binding = model.createReaction()
reaction_ab{ab_type}_binding.setId("ab{ab_type}_monomer_antibody_binding_central")
reaction_ab{ab_type}_binding.setReversible(False)

# Reactants: AB{ab_type}Mu_Central and PK_central
reactant_ab{ab_type} = reaction_ab{ab_type}_binding.createReactant()
reactant_ab{ab_type}.setSpecies("AB{ab_type}Mu_Central")
reactant_ab{ab_type}.setStoichiometry(1.0)
reactant_ab{ab_type}.setConstant(True)

reactant_pk = reaction_ab{ab_type}_binding.createReactant()
reactant_pk.setSpecies("PK_central")
reactant_pk.setStoichiometry(1.0)
reactant_pk.setConstant(True)

# Product: AB{ab_type}Mb_Central (antibody-bound AB{ab_type})
product_ab{ab_type}_bound = reaction_ab{ab_type}_binding.createProduct()
product_ab{ab_type}_bound.setSpecies("AB{ab_type}Mb_Central")
product_ab{ab_type}_bound.setStoichiometry(1.0)
product_ab{ab_type}_bound.setConstant(True)

# Kinetic law:  fta0 * AB{ab_type}Mu_Central * PK_central
klaw_ab{ab_type}_binding = reaction_ab{ab_type}_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_Central * PK_central")
klaw_ab{ab_type}_binding.setMath(math_ast)
            ''',
            # Reaction block 2
            f'''
# 3. Binding of antibody (PK_p_brain) to AB{ab_type} monomer in brain plasma
reaction_ab{ab_type}_binding_brain = model.createReaction()
reaction_ab{ab_type}_binding_brain.setId("ab{ab_type}_monomer_antibody_binding_brain_plasma")
reaction_ab{ab_type}_binding_brain.setReversible(False)

# Reactants: AB{ab_type}Mu_Brain_Plasma and PK_p_brain
reactant_ab{ab_type}_brain = reaction_ab{ab_type}_binding_brain.createReactant()
reactant_ab{ab_type}_brain.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}_brain.setStoichiometry(1.0)
reactant_ab{ab_type}_brain.setConstant(True)

reactant_pk_brain = reaction_ab{ab_type}_binding_brain.createReactant()
reactant_pk_brain.setSpecies("PK_p_brain")
reactant_pk_brain.setStoichiometry(1.0)
reactant_pk_brain.setConstant(True)

# Product: AB{ab_type}Mb_Brain_Plasma (antibody-bound AB{ab_type} in brain plasma)
product_ab{ab_type}_bound_brain = reaction_ab{ab_type}_binding_brain.createProduct()
product_ab{ab_type}_bound_brain.setSpecies("AB{ab_type}Mb_Brain_Plasma")
product_ab{ab_type}_bound_brain.setStoichiometry(1.0)
product_ab{ab_type}_bound_brain.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_Brain_Plasma * PK_p_brain
klaw_ab{ab_type}_binding_brain = reaction_ab{ab_type}_binding_brain.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_Brain_Plasma * PK_p_brain * Vp_brain")
klaw_ab{ab_type}_binding_brain.setMath(math_ast)
            ''',
            # Reaction block 3
            f'''
# 5. Binding of antibody (PK_BCSFB_unbound_brain) to AB{ab_type} monomer in BCSFB
reaction_ab{ab_type}_binding_bcsfb = model.createReaction()
reaction_ab{ab_type}_binding_bcsfb.setId("ab{ab_type}_monomer_antibody_binding_bcsfb")
reaction_ab{ab_type}_binding_bcsfb.setReversible(False)

# Reactants: AB{ab_type}Mu_BCSFB and PK_BCSFB_unbound_brain
reactant_ab{ab_type}_bcsfb = reaction_ab{ab_type}_binding_bcsfb.createReactant()
reactant_ab{ab_type}_bcsfb.setSpecies("AB{ab_type}Mu_BCSFB")
reactant_ab{ab_type}_bcsfb.setStoichiometry(1.0)
reactant_ab{ab_type}_bcsfb.setConstant(True)

reactant_pk_bcsfb = reaction_ab{ab_type}_binding_bcsfb.createReactant()
reactant_pk_bcsfb.setSpecies("PK_BCSFB_unbound_brain")
reactant_pk_bcsfb.setStoichiometry(1.0)
reactant_pk_bcsfb.setConstant(True)

# Product: AB{ab_type}Mb_BCSFB_Unbound (antibody-bound AB{ab_type} in BCSFB)
product_ab{ab_type}_bound_bcsfb = reaction_ab{ab_type}_binding_bcsfb.createProduct()
product_ab{ab_type}_bound_bcsfb.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
product_ab{ab_type}_bound_bcsfb.setStoichiometry(1.0)
product_ab{ab_type}_bound_bcsfb.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_BCSFB * PK_BCSFB_unbound_brain
klaw_ab{ab_type}_binding_bcsfb = reaction_ab{ab_type}_binding_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_BCSFB * PK_BCSFB_unbound_brain * V_BCSFB_brain")
klaw_ab{ab_type}_binding_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 4
            f'''
# 7. Binding of antibody (PK_BBB_unbound_brain) to AB{ab_type} monomer in BBB
reaction_ab{ab_type}_binding_bbb = model.createReaction()
reaction_ab{ab_type}_binding_bbb.setId("ab{ab_type}_monomer_antibody_binding_bbb")
reaction_ab{ab_type}_binding_bbb.setReversible(False)

# Reactants: AB{ab_type}Mu_BBB and PK_BBB_unbound_brain
reactant_ab{ab_type}_bbb = reaction_ab{ab_type}_binding_bbb.createReactant()
reactant_ab{ab_type}_bbb.setSpecies("AB{ab_type}Mu_BBB")
reactant_ab{ab_type}_bbb.setStoichiometry(1.0)
reactant_ab{ab_type}_bbb.setConstant(True)

reactant_pk_bbb = reaction_ab{ab_type}_binding_bbb.createReactant()
reactant_pk_bbb.setSpecies("PK_BBB_unbound_brain")
reactant_pk_bbb.setStoichiometry(1.0)
reactant_pk_bbb.setConstant(True)

# Product: AB{ab_type}Mb_BBB_Unbound (antibody-bound AB{ab_type} in BBB)
product_ab{ab_type}_bound_bbb = reaction_ab{ab_type}_binding_bbb.createProduct()
product_ab{ab_type}_bound_bbb.setSpecies("AB{ab_type}Mb_BBB_Unbound")
product_ab{ab_type}_bound_bbb.setStoichiometry(1.0)
product_ab{ab_type}_bound_bbb.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_BBB * PK_BBB_unbound_brain
klaw_ab{ab_type}_binding_bbb = reaction_ab{ab_type}_binding_bbb.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_BBB * PK_BBB_unbound_brain * VBBB_brain")
klaw_ab{ab_type}_binding_bbb.setMath(math_ast)
            ''',
            # Reaction block 4.5 - ISF binding
            f'''
# 9. Binding of AB{ab_type}_monomer to antibody in brain ISF 
reaction_ab{ab_type}_isf_binding = model.createReaction()
reaction_ab{ab_type}_isf_binding.setId("ab{ab_type}_monomer_antibody_binding_isf")
reaction_ab{ab_type}_isf_binding.setReversible(False)

# Reactants: Ab_t (antibody in ISF)
reactant_ab{ab_type}_isf = reaction_ab{ab_type}_isf_binding.createReactant()
reactant_ab{ab_type}_isf.setSpecies("Ab_t")
reactant_ab{ab_type}_isf.setStoichiometry(1.0)
reactant_ab{ab_type}_isf.setConstant(True)

# Reactant: AB{ab_type}_monomer (monomer in ISF)
reactant_ab{ab_type}_monomer_isf = reaction_ab{ab_type}_isf_binding.createReactant()
reactant_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
reactant_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
reactant_ab{ab_type}_monomer_isf.setConstant(True)

# Product: AB{ab_type}_monomer_antibody_bound
product_ab{ab_type}_isf_bound = reaction_ab{ab_type}_isf_binding.createProduct()
product_ab{ab_type}_isf_bound.setSpecies("AB{ab_type}_monomer_antibody_bound")
product_ab{ab_type}_isf_bound.setStoichiometry(1.0)
product_ab{ab_type}_isf_bound.setConstant(True)

# Kinetic law: fta0 * Ab_t * AB{ab_type}_monomer
klaw_ab{ab_type}_isf_binding = reaction_ab{ab_type}_isf_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * Ab_t * AB{ab_type}_Monomer * VIS_brain")
klaw_ab{ab_type}_isf_binding.setMath(math_ast)
            ''',
            # Reaction block 5
            f'''
# 10. Binding of antibody (PK_LV_brain) to AB{ab_type} monomer in CSF LV
reaction_ab{ab_type}_binding_lv = model.createReaction()
reaction_ab{ab_type}_binding_lv.setId("ab{ab_type}_monomer_antibody_binding_lv")
reaction_ab{ab_type}_binding_lv.setReversible(False)

# Reactants: AB{ab_type}Mu_LV and PK_LV_brain
reactant_ab{ab_type}_lv = reaction_ab{ab_type}_binding_lv.createReactant()
reactant_ab{ab_type}_lv.setSpecies("AB{ab_type}Mu_LV")
reactant_ab{ab_type}_lv.setStoichiometry(1.0)
reactant_ab{ab_type}_lv.setConstant(True)

reactant_pk_lv = reaction_ab{ab_type}_binding_lv.createReactant()
reactant_pk_lv.setSpecies("PK_LV_brain")
reactant_pk_lv.setStoichiometry(1.0)
reactant_pk_lv.setConstant(True)

# Product: AB{ab_type}Mb_LV (antibody-bound AB{ab_type} in LV)
product_ab{ab_type}_bound_lv = reaction_ab{ab_type}_binding_lv.createProduct()
product_ab{ab_type}_bound_lv.setSpecies("AB{ab_type}Mb_LV")
product_ab{ab_type}_bound_lv.setStoichiometry(1.0)
product_ab{ab_type}_bound_lv.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_LV * PK_LV_brain
klaw_ab{ab_type}_binding_lv = reaction_ab{ab_type}_binding_lv.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_LV * PK_LV_brain * V_LV_brain")
klaw_ab{ab_type}_binding_lv.setMath(math_ast)
            ''',
            # Reaction block 6
            f'''
# 12. Binding of antibody (PK_TFV_brain) to AB{ab_type} monomer in CSF TFV
reaction_ab{ab_type}_binding_tfv = model.createReaction()
reaction_ab{ab_type}_binding_tfv.setId("ab{ab_type}_monomer_antibody_binding_tfv")
reaction_ab{ab_type}_binding_tfv.setReversible(False)

# Reactants: AB{ab_type}Mu_TFV and PK_TFV_brain
reactant_ab{ab_type}_tfv = reaction_ab{ab_type}_binding_tfv.createReactant()
reactant_ab{ab_type}_tfv.setSpecies("AB{ab_type}Mu_TFV")
reactant_ab{ab_type}_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}_tfv.setConstant(True)

reactant_pk_tfv = reaction_ab{ab_type}_binding_tfv.createReactant()
reactant_pk_tfv.setSpecies("PK_TFV_brain")
reactant_pk_tfv.setStoichiometry(1.0)
reactant_pk_tfv.setConstant(True)

# Product: AB{ab_type}Mb_TFV (antibody-bound AB{ab_type} in TFV)
product_ab{ab_type}_bound_tfv = reaction_ab{ab_type}_binding_tfv.createProduct()
product_ab{ab_type}_bound_tfv.setSpecies("AB{ab_type}Mb_TFV")
product_ab{ab_type}_bound_tfv.setStoichiometry(1.0)
product_ab{ab_type}_bound_tfv.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_TFV * PK_TFV_brain
klaw_ab{ab_type}_binding_tfv = reaction_ab{ab_type}_binding_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_TFV * PK_TFV_brain * V_TFV_brain")
klaw_ab{ab_type}_binding_tfv.setMath(math_ast)
            ''',
            # Reaction block 7
            f'''
# 14. Binding of antibody (PK_CM_brain) to AB{ab_type} monomer in CSF CM
reaction_ab{ab_type}_binding_cm = model.createReaction()
reaction_ab{ab_type}_binding_cm.setId("ab{ab_type}_monomer_antibody_binding_cm")
reaction_ab{ab_type}_binding_cm.setReversible(False)

# Reactants: AB{ab_type}Mu_CM and PK_CM_brain
reactant_ab{ab_type}_cm = reaction_ab{ab_type}_binding_cm.createReactant()
reactant_ab{ab_type}_cm.setSpecies("AB{ab_type}Mu_CM")
reactant_ab{ab_type}_cm.setStoichiometry(1.0)
reactant_ab{ab_type}_cm.setConstant(True)

reactant_pk_cm = reaction_ab{ab_type}_binding_cm.createReactant()
reactant_pk_cm.setSpecies("PK_CM_brain")
reactant_pk_cm.setStoichiometry(1.0)
reactant_pk_cm.setConstant(True)

# Product: AB{ab_type}Mb_CM (antibody-bound AB{ab_type} in CM)
product_ab{ab_type}_bound_cm = reaction_ab{ab_type}_binding_cm.createProduct()
product_ab{ab_type}_bound_cm.setSpecies("AB{ab_type}Mb_CM")
product_ab{ab_type}_bound_cm.setStoichiometry(1.0)
product_ab{ab_type}_bound_cm.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_CM * PK_CM_brain
klaw_ab{ab_type}_binding_cm = reaction_ab{ab_type}_binding_cm.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_CM * PK_CM_brain * V_CM_brain")
klaw_ab{ab_type}_binding_cm.setMath(math_ast)
            ''',
            # Reaction block 8
            f'''
# 16. Binding of antibody (PK_SAS_brain) to AB{ab_type} monomer in CSF SAS
reaction_ab{ab_type}_binding_sas = model.createReaction()
reaction_ab{ab_type}_binding_sas.setId("ab{ab_type}_monomer_antibody_binding_sas")
reaction_ab{ab_type}_binding_sas.setReversible(False)

# Reactants: AB{ab_type}Mu_SAS and PK_SAS_brain
reactant_ab{ab_type}_sas = reaction_ab{ab_type}_binding_sas.createReactant()
reactant_ab{ab_type}_sas.setSpecies("AB{ab_type}Mu_SAS")
reactant_ab{ab_type}_sas.setStoichiometry(1.0)
reactant_ab{ab_type}_sas.setConstant(True)

reactant_pk_sas = reaction_ab{ab_type}_binding_sas.createReactant()
reactant_pk_sas.setSpecies("PK_SAS_brain")
reactant_pk_sas.setStoichiometry(1.0)
reactant_pk_sas.setConstant(True)

# Product: AB{ab_type}Mb_SAS (antibody-bound AB{ab_type} in SAS)
product_ab{ab_type}_bound_sas = reaction_ab{ab_type}_binding_sas.createProduct()
product_ab{ab_type}_bound_sas.setSpecies("AB{ab_type}Mb_SAS")
product_ab{ab_type}_bound_sas.setStoichiometry(1.0)
product_ab{ab_type}_bound_sas.setConstant(True)

# Kinetic law: fta0 * AB{ab_type}Mu_SAS * PK_SAS_brain
klaw_ab{ab_type}_binding_sas = reaction_ab{ab_type}_binding_sas.createKineticLaw()
math_ast = libsbml.parseL3Formula("fta0 * AB{ab_type}Mu_SAS * PK_SAS_brain * V_SAS_brain")
klaw_ab{ab_type}_binding_sas.setMath(math_ast)
            ''',
            # Reaction block 9
            f'''
# 1. Systemic synthesis of AB{ab_type}Mu in central compartment
reaction_ab{ab_type}mu_synthesis = model.createReaction()
reaction_ab{ab_type}mu_synthesis.setId("ab{ab_type}mu_systemic_synthesis")
reaction_ab{ab_type}mu_synthesis.setReversible(False)

# Add Source as reactant for synthesis 
reactant_source = reaction_ab{ab_type}mu_synthesis.createReactant()
reactant_source.setSpecies(sources[f"AB{ab_type}Mu_Central"])
reactant_source.setStoichiometry(1.0)
reactant_source.setConstant(True)

# Product: AB{ab_type}Mu_Central
product_ab{ab_type}mu = reaction_ab{ab_type}mu_synthesis.createProduct()
product_ab{ab_type}mu.setSpecies("AB{ab_type}Mu_Central")
product_ab{ab_type}mu.setStoichiometry(1.0)
product_ab{ab_type}mu.setConstant(True)

# Kinetic law: AB{ab_type}Mu_systemic_synthesis_rate
klaw_ab{ab_type}mu_synthesis = reaction_ab{ab_type}mu_synthesis.createKineticLaw()
math_ast = libsbml.parseL3Formula("AB{ab_type}Mu_systemic_synthesis_rate ")
klaw_ab{ab_type}mu_synthesis.setMath(math_ast)
            ''',
            # Reaction block 10
            f'''
# 2. Clearance of AB{ab_type}Mu from central compartment
reaction_ab{ab_type}mu_clearance = model.createReaction()
reaction_ab{ab_type}mu_clearance.setId("ab{ab_type}mu_central_clearance")
reaction_ab{ab_type}mu_clearance.setReversible(False)

# Reactant: AB{ab_type}Mu_Central
reactant_ab{ab_type}mu = reaction_ab{ab_type}mu_clearance.createReactant()
reactant_ab{ab_type}mu.setSpecies("AB{ab_type}Mu_Central")
reactant_ab{ab_type}mu.setStoichiometry(1.0)
reactant_ab{ab_type}mu.setConstant(True)

# Add Sink as product for clearance
product_sink = reaction_ab{ab_type}mu_clearance.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}Mu_Central"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)

# Kinetic law: (AB{ab_type}MuCL/AB{ab_type}MuV_cent) * AB{ab_type}Mu_Central
klaw_ab{ab_type}mu_clearance = reaction_ab{ab_type}mu_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula("(AB{ab_type}MuCL) * AB{ab_type}Mu_Central")
klaw_ab{ab_type}mu_clearance.setMath(math_ast)
            ''',
            # Reaction block 11
            f'''
# 3. Distribution of AB{ab_type}Mu from central to peripheral compartment
reaction_ab{ab_type}mu_cent_to_per = model.createReaction()
reaction_ab{ab_type}mu_cent_to_per.setId("ab{ab_type}mu_central_to_peripheral")
reaction_ab{ab_type}mu_cent_to_per.setReversible(False)

# Reactant: AB{ab_type}Mu_Central
reactant_ab{ab_type}mu_cent = reaction_ab{ab_type}mu_cent_to_per.createReactant()
reactant_ab{ab_type}mu_cent.setSpecies("AB{ab_type}Mu_Central")
reactant_ab{ab_type}mu_cent.setStoichiometry(1.0)
reactant_ab{ab_type}mu_cent.setConstant(True)

# Product: AB{ab_type}Mu_Peripheral
product_ab{ab_type}mu_per = reaction_ab{ab_type}mu_cent_to_per.createProduct()
product_ab{ab_type}mu_per.setSpecies("AB{ab_type}Mu_Peripheral")
product_ab{ab_type}mu_per.setStoichiometry(1.0)
product_ab{ab_type}mu_per.setConstant(True)

# Kinetic law: (AB{ab_type}MuCLd2/AB{ab_type}MuV_cent) * AB{ab_type}Mu_Central
klaw_ab{ab_type}mu_cent_to_per = reaction_ab{ab_type}mu_cent_to_per.createKineticLaw()
math_ast = libsbml.parseL3Formula("(AB{ab_type}MuCLd2) * AB{ab_type}Mu_Central")
klaw_ab{ab_type}mu_cent_to_per.setMath(math_ast)
            ''',
            # Reaction block 12
            f'''
# 4. Distribution of AB{ab_type}Mu from peripheral to central compartment
reaction_ab{ab_type}mu_per_to_cent = model.createReaction()
reaction_ab{ab_type}mu_per_to_cent.setId("ab{ab_type}mu_peripheral_to_central")
reaction_ab{ab_type}mu_per_to_cent.setReversible(False)

# Reactant: AB{ab_type}Mu_Peripheral
reactant_ab{ab_type}mu_per = reaction_ab{ab_type}mu_per_to_cent.createReactant()
reactant_ab{ab_type}mu_per.setSpecies("AB{ab_type}Mu_Peripheral")
reactant_ab{ab_type}mu_per.setStoichiometry(1.0)
reactant_ab{ab_type}mu_per.setConstant(True)

# Product: AB{ab_type}Mu_Central
product_ab{ab_type}mu_cent = reaction_ab{ab_type}mu_per_to_cent.createProduct()
product_ab{ab_type}mu_cent.setSpecies("AB{ab_type}Mu_Central")
product_ab{ab_type}mu_cent.setStoichiometry(1.0)
product_ab{ab_type}mu_cent.setConstant(True)

# Kinetic law: (AB{ab_type}MuCLd2/AB{ab_type}MuV_per) * AB{ab_type}Mu_Peripheral
klaw_ab{ab_type}mu_per_to_cent = reaction_ab{ab_type}mu_per_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(AB{ab_type}MuCLd2) * AB{ab_type}Mu_Peripheral")
klaw_ab{ab_type}mu_per_to_cent.setMath(math_ast)
            ''',
            # Reaction block 13
            f'''
# 5. Flow of AB{ab_type}Mu from central compartment to brain plasma
reaction_ab{ab_type}mu_cent_to_brain = model.createReaction()
reaction_ab{ab_type}mu_cent_to_brain.setId("ab{ab_type}mu_central_to_brain_plasma")
reaction_ab{ab_type}mu_cent_to_brain.setReversible(False)

# Reactant: AB{ab_type}Mu_Central
reactant_ab{ab_type}mu_cent = reaction_ab{ab_type}mu_cent_to_brain.createReactant()
reactant_ab{ab_type}mu_cent.setSpecies("AB{ab_type}Mu_Central")
reactant_ab{ab_type}mu_cent.setStoichiometry(1.0)
reactant_ab{ab_type}mu_cent.setConstant(True)

# Product: AB{ab_type}Mu_Brain_Plasma
product_ab{ab_type}mu_brain = reaction_ab{ab_type}mu_cent_to_brain.createProduct()
product_ab{ab_type}mu_brain.setSpecies("AB{ab_type}Mu_Brain_Plasma")
product_ab{ab_type}mu_brain.setStoichiometry(1.0)
product_ab{ab_type}mu_brain.setConstant(True)

# Kinetic law: Q_brain_plasma * (AB{ab_type}Mu_Central)
klaw_ab{ab_type}mu_cent_to_brain = reaction_ab{ab_type}mu_cent_to_brain.createKineticLaw()
math_ast = libsbml.parseL3Formula("Q_p_brain * (AB{ab_type}Mu_Central)")
klaw_ab{ab_type}mu_cent_to_brain.setMath(math_ast)
            ''',
            # Reaction block 14
            f'''
# 6. Flow of AB{ab_type}Mu from brain plasma to central compartment
reaction_ab{ab_type}mu_brain_to_cent = model.createReaction()
reaction_ab{ab_type}mu_brain_to_cent.setId("ab{ab_type}mu_brain_plasma_to_central")
reaction_ab{ab_type}mu_brain_to_cent.setReversible(False)

# Reactant: AB{ab_type}Mu_Brain_Plasma
reactant_ab{ab_type}mu_brain = reaction_ab{ab_type}mu_brain_to_cent.createReactant()
reactant_ab{ab_type}mu_brain.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}mu_brain.setStoichiometry(1.0)
reactant_ab{ab_type}mu_brain.setConstant(True)

# Product: AB{ab_type}Mu_Central
product_ab{ab_type}mu_cent = reaction_ab{ab_type}mu_brain_to_cent.createProduct()
product_ab{ab_type}mu_cent.setSpecies("AB{ab_type}Mu_Central")
product_ab{ab_type}mu_cent.setStoichiometry(1.0)
product_ab{ab_type}mu_cent.setConstant(True)

# Kinetic law: (Q_p_brain - L_brain) * AB{ab_type}Mu_Brain_Plasma
klaw_ab{ab_type}mu_brain_to_cent = reaction_ab{ab_type}mu_brain_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(Q_p_brain - L_brain) * AB{ab_type}Mu_Brain_Plasma")
klaw_ab{ab_type}mu_brain_to_cent.setMath(math_ast)
            ''',
            # Reaction block 15
            f'''
# 7. Flow of AB{ab_type}Mu from CSF SAS to central compartment
reaction_ab{ab_type}mu_sas_to_cent = model.createReaction()
reaction_ab{ab_type}mu_sas_to_cent.setId("ab{ab_type}mu_csf_sas_to_central")
reaction_ab{ab_type}mu_sas_to_cent.setReversible(False)

# Reactant: AB{ab_type}Mu_SAS
reactant_ab{ab_type}mu_sas = reaction_ab{ab_type}mu_sas_to_cent.createReactant()
reactant_ab{ab_type}mu_sas.setSpecies("AB{ab_type}Mu_SAS")
reactant_ab{ab_type}mu_sas.setStoichiometry(1.0)
reactant_ab{ab_type}mu_sas.setConstant(True)

# Product: AB{ab_type}Mu_Central
product_ab{ab_type}mu_cent = reaction_ab{ab_type}mu_sas_to_cent.createProduct()
product_ab{ab_type}mu_cent.setSpecies("AB{ab_type}Mu_Central")
product_ab{ab_type}mu_cent.setStoichiometry(1.0)
product_ab{ab_type}mu_cent.setConstant(True)

# Kinetic law: (1 - sigma_L_SAS_AB{ab_type}Mu) * Q_CSF_brain * AB{ab_type}Mu_SAS
klaw_ab{ab_type}mu_sas_to_cent = reaction_ab{ab_type}mu_sas_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - sigma_L_SAS_AB{ab_type}Mu) * Q_CSF_brain * AB{ab_type}Mu_SAS")
klaw_ab{ab_type}mu_sas_to_cent.setMath(math_ast)
            ''',
            # Reaction block 16
            f'''
# 8. Flow of AB{ab_type}Mu from ISF to central compartment
reaction_ab{ab_type}mu_isf_to_cent = model.createReaction()
reaction_ab{ab_type}mu_isf_to_cent.setId("ab{ab_type}mu_isf_to_central")
reaction_ab{ab_type}mu_isf_to_cent.setReversible(False)

# Reactant:(AB{ab_type}_monomer in ISF) 
reactant_ab{ab_type}_monomer_isf = reaction_ab{ab_type}mu_isf_to_cent.createReactant()
reactant_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
reactant_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
reactant_ab{ab_type}_monomer_isf.setConstant(True)

# Product: AB{ab_type}Mu_Central
product_ab{ab_type}mu_cent = reaction_ab{ab_type}mu_isf_to_cent.createProduct()
product_ab{ab_type}mu_cent.setSpecies("AB{ab_type}Mu_Central")
product_ab{ab_type}mu_cent.setStoichiometry(1.0)
product_ab{ab_type}mu_cent.setConstant(True)

# Kinetic law: (1 - sigma_L_ISF_central_AB{ab_type}Mu) * Q_ISF_brain * AB{ab_type}_monomer
klaw_ab{ab_type}mu_isf_to_cent = reaction_ab{ab_type}mu_isf_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - sigma_L_ISF_central_AB{ab_type}Mu) * Q_ISF_brain * AB{ab_type}_Monomer")
klaw_ab{ab_type}mu_isf_to_cent.setMath(math_ast)
            ''',
            # Reaction block 17
            f'''
# 1. Flow of AB{ab_type}Mu from brain plasma to CSF through BCSFB - LV compartment
reaction_ab{ab_type}mu_brain_to_csf_lv = model.createReaction()
reaction_ab{ab_type}mu_brain_to_csf_lv.setId("ab{ab_type}mu_brain_plasma_to_csf_lv")
reaction_ab{ab_type}mu_brain_to_csf_lv.setReversible(False)

# Reactant: AB{ab_type}Mu_Brain_Plasma
reactant_ab{ab_type}mu_brain_lv = reaction_ab{ab_type}mu_brain_to_csf_lv.createReactant()
reactant_ab{ab_type}mu_brain_lv.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}mu_brain_lv.setStoichiometry(1.0)
reactant_ab{ab_type}mu_brain_lv.setConstant(True)

# Product: AB{ab_type}Mu_CSF_LV
product_ab{ab_type}mu_csf_lv = reaction_ab{ab_type}mu_brain_to_csf_lv.createProduct()
product_ab{ab_type}mu_csf_lv.setSpecies("AB{ab_type}Mu_LV")
product_ab{ab_type}mu_csf_lv.setStoichiometry(1.0)
product_ab{ab_type}mu_csf_lv.setConstant(True)

# Kinetic law: f_LV * (1 - sigma_V_BCSFB_AB{ab_type}Mu) * Q_CSF_brain * AB{ab_type}Mu_Brain_Plasma
klaw_ab{ab_type}mu_brain_to_csf_lv = reaction_ab{ab_type}mu_brain_to_csf_lv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * (1 - sigma_V_BCSFB_AB{ab_type}Mu) * Q_CSF_brain * AB{ab_type}Mu_Brain_Plasma")
klaw_ab{ab_type}mu_brain_to_csf_lv.setMath(math_ast)
            ''',
            # Reaction block 18
            f'''
# 2. Flow of AB{ab_type}Mu from brain plasma to CSF through BCSFB - TFV compartment
reaction_ab{ab_type}mu_brain_to_csf_tfv = model.createReaction()
reaction_ab{ab_type}mu_brain_to_csf_tfv.setId("ab{ab_type}mu_brain_plasma_to_csf_tfv")
reaction_ab{ab_type}mu_brain_to_csf_tfv.setReversible(False)

# Reactant: AB{ab_type}Mu_Brain_Plasma
reactant_ab{ab_type}mu_brain_tfv = reaction_ab{ab_type}mu_brain_to_csf_tfv.createReactant()
reactant_ab{ab_type}mu_brain_tfv.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}mu_brain_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}mu_brain_tfv.setConstant(True)

# Product: AB{ab_type}Mu_CSF_TFV
product_ab{ab_type}mu_csf_tfv = reaction_ab{ab_type}mu_brain_to_csf_tfv.createProduct()
product_ab{ab_type}mu_csf_tfv.setSpecies("AB{ab_type}Mu_TFV")
product_ab{ab_type}mu_csf_tfv.setStoichiometry(1.0)
product_ab{ab_type}mu_csf_tfv.setConstant(True)

# Kinetic law: (1 - f_LV) * (1 - sigma_V_BCSFB_AB{ab_type}Mu) * Q_CSF_brain * AB{ab_type}Mu_Brain_Plasma
klaw_ab{ab_type}mu_brain_to_csf_tfv = reaction_ab{ab_type}mu_brain_to_csf_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * (1 - sigma_V_BCSFB_AB{ab_type}Mu) * Q_CSF_brain * AB{ab_type}Mu_Brain_Plasma")
klaw_ab{ab_type}mu_brain_to_csf_tfv.setMath(math_ast)
            ''',
            # Reaction block 19
            f'''
# 3. Flow of AB{ab_type}Mu from brain plasma to CSF through BCSFB - ISF compartment
reaction_ab{ab_type}mu_brain_to_csf_isf = model.createReaction()
reaction_ab{ab_type}mu_brain_to_csf_isf.setId("ab{ab_type}mu_brain_plasma_to_csf_isf")
reaction_ab{ab_type}mu_brain_to_csf_isf.setReversible(False)

# Reactant: AB{ab_type}Mu_Brain_Plasma
reactant_ab{ab_type}mu_brain_isf = reaction_ab{ab_type}mu_brain_to_csf_isf.createReactant()
reactant_ab{ab_type}mu_brain_isf.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}mu_brain_isf.setStoichiometry(1.0)
reactant_ab{ab_type}mu_brain_isf.setConstant(True)

# Product: AB{ab_type}_monomer

product_ab{ab_type}_monomer_isf = reaction_ab{ab_type}mu_brain_to_csf_isf.createProduct()
product_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
product_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
product_ab{ab_type}_monomer_isf.setConstant(True)

# Kinetic law: (1-sigma_V_ISF_Ab) * Q_ISF_brain * AB{ab_type}Mu_Brain_Plasma
klaw_ab{ab_type}mu_brain_to_csf_isf = reaction_ab{ab_type}mu_brain_to_csf_isf.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1-sigma_V_ISF_AB{ab_type}Mu) * Q_ISF_brain * AB{ab_type}Mu_Brain_Plasma")
klaw_ab{ab_type}mu_brain_to_csf_isf.setMath(math_ast)
            ''',
            # Reaction block 20
            f'''
# 4. Uptake of AB{ab_type}Mu from brain plasma to BBB
reaction_ab{ab_type}mu_brain_to_bbb = model.createReaction()
reaction_ab{ab_type}mu_brain_to_bbb.setId("ab{ab_type}mu_brain_plasma_to_bbb")
reaction_ab{ab_type}mu_brain_to_bbb.setReversible(False)

# Reactant: AB{ab_type}Mu_Brain_Plasma
reactant_ab{ab_type}mu_brain_bbb = reaction_ab{ab_type}mu_brain_to_bbb.createReactant()
reactant_ab{ab_type}mu_brain_bbb.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}mu_brain_bbb.setStoichiometry(1.0)
reactant_ab{ab_type}mu_brain_bbb.setConstant(True)

# Product: AB{ab_type}Mu_BBB
product_ab{ab_type}mu_bbb = reaction_ab{ab_type}mu_brain_to_bbb.createProduct()
product_ab{ab_type}mu_bbb.setSpecies("AB{ab_type}Mu_BBB")
product_ab{ab_type}mu_bbb.setStoichiometry(1.0)
product_ab{ab_type}mu_bbb.setConstant(True)

# Kinetic law: CLup_brain * f_BBB * VES_brain * AB{ab_type}Mu_Brain_Plasma 
klaw_ab{ab_type}mu_brain_to_bbb = reaction_ab{ab_type}mu_brain_to_bbb.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain_AB{ab_type}Mu * f_BBB * VES_brain * AB{ab_type}Mu_Brain_Plasma")
klaw_ab{ab_type}mu_brain_to_bbb.setMath(math_ast)
            ''',
            # Reaction block 21
            f'''
# 5. Uptake of AB{ab_type}Mu from brain plasma to BCSFB
reaction_ab{ab_type}mu_brain_to_bcsfb = model.createReaction()
reaction_ab{ab_type}mu_brain_to_bcsfb.setId("ab{ab_type}mu_brain_plasma_to_bcsfb")
reaction_ab{ab_type}mu_brain_to_bcsfb.setReversible(False)

# Reactant: AB{ab_type}Mu_Brain_Plasma
reactant_ab{ab_type}mu_brain_bcsfb = reaction_ab{ab_type}mu_brain_to_bcsfb.createReactant()
reactant_ab{ab_type}mu_brain_bcsfb.setSpecies("AB{ab_type}Mu_Brain_Plasma")
reactant_ab{ab_type}mu_brain_bcsfb.setStoichiometry(1.0)
reactant_ab{ab_type}mu_brain_bcsfb.setConstant(True)

# Product: AB{ab_type}Mu_BCSFB
product_ab{ab_type}mu_bcsfb = reaction_ab{ab_type}mu_brain_to_bcsfb.createProduct()
product_ab{ab_type}mu_bcsfb.setSpecies("AB{ab_type}Mu_BCSFB")
product_ab{ab_type}mu_bcsfb.setStoichiometry(1.0)
product_ab{ab_type}mu_bcsfb.setConstant(True)

# Kinetic law: CLup_brain * f_BCSFB * VES_brain * AB{ab_type}Mu_Brain_Plasma 
klaw_ab{ab_type}mu_brain_to_bcsfb = reaction_ab{ab_type}mu_brain_to_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain_AB{ab_type}Mu * f_BCSFB * VES_brain * AB{ab_type}Mu_Brain_Plasma")
klaw_ab{ab_type}mu_brain_to_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 22
            f'''
# 1. Flow of AB{ab_type}Mu from CSF TFV to BCSFB
reaction_ab{ab_type}mu_csf_tfv_to_bcsfb = model.createReaction()
reaction_ab{ab_type}mu_csf_tfv_to_bcsfb.setId("ab{ab_type}mu_csf_tfv_to_bcsfb")
reaction_ab{ab_type}mu_csf_tfv_to_bcsfb.setReversible(False)

# Reactant: AB{ab_type}Mu_CSF_TFV
reactant_ab{ab_type}mu_csf_tfv = reaction_ab{ab_type}mu_csf_tfv_to_bcsfb.createReactant()
reactant_ab{ab_type}mu_csf_tfv.setSpecies("AB{ab_type}Mu_TFV")
reactant_ab{ab_type}mu_csf_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}mu_csf_tfv.setConstant(True)

# Product: AB{ab_type}Mu_BCSFB
product_ab{ab_type}mu_bcsfb_tfv = reaction_ab{ab_type}mu_csf_tfv_to_bcsfb.createProduct()
product_ab{ab_type}mu_bcsfb_tfv.setSpecies("AB{ab_type}Mu_BCSFB")
product_ab{ab_type}mu_bcsfb_tfv.setStoichiometry(1.0)
product_ab{ab_type}mu_bcsfb_tfv.setConstant(True)

# Kinetic law: (1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * AB{ab_type}Mu_CSF_TFV 
klaw_ab{ab_type}mu_csf_tfv_to_bcsfb = reaction_ab{ab_type}mu_csf_tfv_to_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * CLup_brain_AB{ab_type}Mu * (1 - f_BBB) * VES_brain * AB{ab_type}Mu_TFV")
klaw_ab{ab_type}mu_csf_tfv_to_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 23
            f'''
# 2. Flow of AB{ab_type}Mu from CSF LV to BCSFB
reaction_ab{ab_type}mu_csf_lv_to_bcsfb = model.createReaction()
reaction_ab{ab_type}mu_csf_lv_to_bcsfb.setId("ab{ab_type}mu_csf_lv_to_bcsfb")
reaction_ab{ab_type}mu_csf_lv_to_bcsfb.setReversible(False)

# Reactant: AB{ab_type}Mu_CSF_LV
reactant_ab{ab_type}mu_csf_lv = reaction_ab{ab_type}mu_csf_lv_to_bcsfb.createReactant()
reactant_ab{ab_type}mu_csf_lv.setSpecies("AB{ab_type}Mu_LV")
reactant_ab{ab_type}mu_csf_lv.setStoichiometry(1.0)
reactant_ab{ab_type}mu_csf_lv.setConstant(True)

# Product: AB{ab_type}Mu_BCSFB
product_ab{ab_type}mu_bcsfb_lv = reaction_ab{ab_type}mu_csf_lv_to_bcsfb.createProduct()
product_ab{ab_type}mu_bcsfb_lv.setSpecies("AB{ab_type}Mu_BCSFB")
product_ab{ab_type}mu_bcsfb_lv.setStoichiometry(1.0)
product_ab{ab_type}mu_bcsfb_lv.setConstant(True)

# Kinetic law: f_LV * CLup_brain * (1 - f_BBB) * VES_brain * AB{ab_type}Mu_CSF_LV
# only LV one uses CLup_brain_AB{ab_type}Mu in csv and TFV uses CLup_brain
klaw_ab{ab_type}mu_csf_lv_to_bcsfb = reaction_ab{ab_type}mu_csf_lv_to_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * CLup_brain_AB{ab_type}Mu * (1 - f_BBB) * VES_brain * AB{ab_type}Mu_LV")
klaw_ab{ab_type}mu_csf_lv_to_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 24
            f'''
# 3. Degradation of AB{ab_type}Mu in BCSFB
reaction_ab{ab_type}mu_bcsfb_degradation = model.createReaction()
reaction_ab{ab_type}mu_bcsfb_degradation.setId("ab{ab_type}mu_bcsfb_degradation")
reaction_ab{ab_type}mu_bcsfb_degradation.setReversible(False)

# Reactant: AB{ab_type}Mu_BCSFB
reactant_ab{ab_type}mu_bcsfb_deg = reaction_ab{ab_type}mu_bcsfb_degradation.createReactant()
reactant_ab{ab_type}mu_bcsfb_deg.setSpecies("AB{ab_type}Mu_BCSFB")
reactant_ab{ab_type}mu_bcsfb_deg.setStoichiometry(1.0)
reactant_ab{ab_type}mu_bcsfb_deg.setConstant(True)

# Add Sink as product for degradation
product_sink = reaction_ab{ab_type}mu_bcsfb_degradation.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}Mu_BCSFB"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)

# Kinetic law: kdeg_AB{ab_type}Mu * AB{ab_type}Mu_BCSFB
klaw_ab{ab_type}mu_bcsfb_degradation = reaction_ab{ab_type}mu_bcsfb_degradation.createKineticLaw()
math_ast = libsbml.parseL3Formula("kdeg_AB{ab_type}Mu * AB{ab_type}Mu_BCSFB * V_BCSFB_brain")
klaw_ab{ab_type}mu_bcsfb_degradation.setMath(math_ast)
            ''',
            # Reaction block 25
            f'''
# 1. Flow of AB{ab_type}Mu from ISF to BBB
reaction_ab{ab_type}mu_isf_to_bbb = model.createReaction()
reaction_ab{ab_type}mu_isf_to_bbb.setId("ab{ab_type}mu_isf_to_bbb")
reaction_ab{ab_type}mu_isf_to_bbb.setReversible(False)

# Reactant: (AB{ab_type}_monomer in ISF) 
reactant_ab{ab_type}_monomer_isf = reaction_ab{ab_type}mu_isf_to_bbb.createReactant()
reactant_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
reactant_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
reactant_ab{ab_type}_monomer_isf.setConstant(True)

# Product: AB{ab_type}Mu_BBB
product_ab{ab_type}mu_bbb_isf = reaction_ab{ab_type}mu_isf_to_bbb.createProduct()
product_ab{ab_type}mu_bbb_isf.setSpecies("AB{ab_type}Mu_BBB")
product_ab{ab_type}mu_bbb_isf.setStoichiometry(1.0)
product_ab{ab_type}mu_bbb_isf.setConstant(True)

# Kinetic law: CLup_brain * f_BBB * VES_brain * AB{ab_type}_monomer 
klaw_ab{ab_type}mu_isf_to_bbb = reaction_ab{ab_type}mu_isf_to_bbb.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain_AB{ab_type}Mu * f_BBB * VES_brain * AB{ab_type}_Monomer")
klaw_ab{ab_type}mu_isf_to_bbb.setMath(math_ast)
            ''',
            # Reaction block 26
            f'''
# 2. Degradation of AB{ab_type}Mu in BBB
reaction_ab{ab_type}mu_bbb_degradation = model.createReaction()
reaction_ab{ab_type}mu_bbb_degradation.setId("ab{ab_type}mu_bbb_degradation")
reaction_ab{ab_type}mu_bbb_degradation.setReversible(False)

# Reactant: AB{ab_type}Mu_BBB
reactant_ab{ab_type}mu_bbb_deg = reaction_ab{ab_type}mu_bbb_degradation.createReactant()
reactant_ab{ab_type}mu_bbb_deg.setSpecies("AB{ab_type}Mu_BBB")
reactant_ab{ab_type}mu_bbb_deg.setStoichiometry(1.0)
reactant_ab{ab_type}mu_bbb_deg.setConstant(True)

# Product: Sink
product_sink = reaction_ab{ab_type}mu_bbb_degradation.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}Mu_BBB"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)

# Kinetic law: kdeg_AB{ab_type}Mu * AB{ab_type}Mu_BBB
klaw_ab{ab_type}mu_bbb_degradation = reaction_ab{ab_type}mu_bbb_degradation.createKineticLaw()
math_ast = libsbml.parseL3Formula("kdeg_AB{ab_type}Mu * AB{ab_type}Mu_BBB * VBBB_brain")
klaw_ab{ab_type}mu_bbb_degradation.setMath(math_ast)
            ''',
            # Reaction block 27
            f'''
# 1. Flow of AB{ab_type}Mu from CSF LV to CSF TFV
reaction_ab{ab_type}mu_csf_lv_to_tfv = model.createReaction()
reaction_ab{ab_type}mu_csf_lv_to_tfv.setId("ab{ab_type}mu_csf_lv_to_tfv")
reaction_ab{ab_type}mu_csf_lv_to_tfv.setReversible(False)

# Reactant: AB{ab_type}Mu_CSF_LV
reactant_ab{ab_type}mu_csf_lv_tfv = reaction_ab{ab_type}mu_csf_lv_to_tfv.createReactant()
reactant_ab{ab_type}mu_csf_lv_tfv.setSpecies("AB{ab_type}Mu_LV")
reactant_ab{ab_type}mu_csf_lv_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}mu_csf_lv_tfv.setConstant(True)

# Product: AB{ab_type}Mu_CSF_TFV
product_ab{ab_type}mu_csf_tfv_lv = reaction_ab{ab_type}mu_csf_lv_to_tfv.createProduct()
product_ab{ab_type}mu_csf_tfv_lv.setSpecies("AB{ab_type}Mu_TFV")
product_ab{ab_type}mu_csf_tfv_lv.setStoichiometry(1.0)
product_ab{ab_type}mu_csf_tfv_lv.setConstant(True)

# Kinetic law: f_LV * (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mu_CSF_LV 
klaw_ab{ab_type}mu_csf_lv_to_tfv = reaction_ab{ab_type}mu_csf_lv_to_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mu_LV")
klaw_ab{ab_type}mu_csf_lv_to_tfv.setMath(math_ast)
            ''',
            # Reaction block 28
            f'''
# 2. Flow of AB{ab_type}Mu from ISF to CSF LV
reaction_ab{ab_type}mu_isf_to_csf_lv = model.createReaction()
reaction_ab{ab_type}mu_isf_to_csf_lv.setId("ab{ab_type}mu_isf_to_csf_lv")
reaction_ab{ab_type}mu_isf_to_csf_lv.setReversible(False)

# Reactant: (AB{ab_type}_monomer in ISF) 
reactant_ab{ab_type}_monomer_isf = reaction_ab{ab_type}mu_isf_to_csf_lv.createReactant()
reactant_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
reactant_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
reactant_ab{ab_type}_monomer_isf.setConstant(True)

# Product: AB{ab_type}Mu_CSF_LV
product_ab{ab_type}mu_csf_lv_isf = reaction_ab{ab_type}mu_isf_to_csf_lv.createProduct()
product_ab{ab_type}mu_csf_lv_isf.setSpecies("AB{ab_type}Mu_LV")
product_ab{ab_type}mu_csf_lv_isf.setStoichiometry(1.0)
product_ab{ab_type}mu_csf_lv_isf.setConstant(True)

# Kinetic law: f_LV * Q_ISF_brain * AB{ab_type}_monomer 
klaw_ab{ab_type}mu_isf_to_csf_lv = reaction_ab{ab_type}mu_isf_to_csf_lv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * Q_ISF_brain * AB{ab_type}_Monomer")
klaw_ab{ab_type}mu_isf_to_csf_lv.setMath(math_ast)
            ''',
            # Reaction block 29
            f'''
# 1. Flow of AB{ab_type}Mu from TFV to CM
reaction_ab{ab_type}mu_tfv_to_cm = model.createReaction()
reaction_ab{ab_type}mu_tfv_to_cm.setId("ab{ab_type}mu_tfv_to_cm")
reaction_ab{ab_type}mu_tfv_to_cm.setReversible(False)

# Reactant: AB{ab_type}Mu_TFV
reactant_ab{ab_type}mu_tfv_cm = reaction_ab{ab_type}mu_tfv_to_cm.createReactant()
reactant_ab{ab_type}mu_tfv_cm.setSpecies("AB{ab_type}Mu_TFV")
reactant_ab{ab_type}mu_tfv_cm.setStoichiometry(1.0)
reactant_ab{ab_type}mu_tfv_cm.setConstant(True)

# Product: AB{ab_type}Mu_CM
product_ab{ab_type}mu_cm_tfv = reaction_ab{ab_type}mu_tfv_to_cm.createProduct()
product_ab{ab_type}mu_cm_tfv.setSpecies("AB{ab_type}Mu_CM")
product_ab{ab_type}mu_cm_tfv.setStoichiometry(1.0)
product_ab{ab_type}mu_cm_tfv.setConstant(True)

# Kinetic law: (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mu_TFV
klaw_ab{ab_type}mu_tfv_to_cm = reaction_ab{ab_type}mu_tfv_to_cm.createKineticLaw()
math_ast = libsbml.parseL3Formula("(Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mu_TFV")
klaw_ab{ab_type}mu_tfv_to_cm.setMath(math_ast)
            ''',
            # Reaction block 30
            f'''
# 2. Flow of AB{ab_type}Mu from ISF to TFV
reaction_ab{ab_type}mu_isf_to_tfv = model.createReaction()
reaction_ab{ab_type}mu_isf_to_tfv.setId("ab{ab_type}mu_isf_to_tfv")
reaction_ab{ab_type}mu_isf_to_tfv.setReversible(False)

# Reactant: (AB{ab_type}_monomer in ISF) # Unsure if this is a species or a parameter
reactant_ab{ab_type}_monomer_isf = reaction_ab{ab_type}mu_isf_to_tfv.createReactant()
reactant_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
reactant_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
reactant_ab{ab_type}_monomer_isf.setConstant(True)

# Product: AB{ab_type}Mu_CSF_TFV
product_ab{ab_type}mu_csf_tfv_isf = reaction_ab{ab_type}mu_isf_to_tfv.createProduct()
product_ab{ab_type}mu_csf_tfv_isf.setSpecies("AB{ab_type}Mu_TFV")
product_ab{ab_type}mu_csf_tfv_isf.setStoichiometry(1.0)
product_ab{ab_type}mu_csf_tfv_isf.setConstant(True)

# Kinetic law: (1 - f_LV) * Q_ISF_brain * AB{ab_type}_monomer 
klaw_ab{ab_type}mu_isf_to_tfv = reaction_ab{ab_type}mu_isf_to_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * Q_ISF_brain * AB{ab_type}_Monomer") #error in paper eq 2
klaw_ab{ab_type}mu_isf_to_tfv.setMath(math_ast)
            ''',
            # Reaction block 31
            f'''
# 1. Flow of AB{ab_type}Mu from CM to SAS
reaction_ab{ab_type}mu_cm_to_sas = model.createReaction()
reaction_ab{ab_type}mu_cm_to_sas.setId("ab{ab_type}mu_cm_to_sas")
reaction_ab{ab_type}mu_cm_to_sas.setReversible(False)

# Reactant: AB{ab_type}Mu_CM
reactant_ab{ab_type}mu_cm_sas = reaction_ab{ab_type}mu_cm_to_sas.createReactant()
reactant_ab{ab_type}mu_cm_sas.setSpecies("AB{ab_type}Mu_CM")
reactant_ab{ab_type}mu_cm_sas.setStoichiometry(1.0)
reactant_ab{ab_type}mu_cm_sas.setConstant(True)

# Product: AB{ab_type}Mu_SAS
product_ab{ab_type}mu_sas_cm = reaction_ab{ab_type}mu_cm_to_sas.createProduct()
product_ab{ab_type}mu_sas_cm.setSpecies("AB{ab_type}Mu_SAS")
product_ab{ab_type}mu_sas_cm.setStoichiometry(1.0)
product_ab{ab_type}mu_sas_cm.setConstant(True)

# Kinetic law: (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mu_CM 
klaw_ab{ab_type}mu_cm_to_sas = reaction_ab{ab_type}mu_cm_to_sas.createKineticLaw()
math_ast = libsbml.parseL3Formula("(Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mu_CM")
klaw_ab{ab_type}mu_cm_to_sas.setMath(math_ast)
            ''',
            # Reaction block 32
            f'''
# 2. Flow of AB{ab_type}Mu from SAS to ISF
reaction_ab{ab_type}mu_sas_to_isf = model.createReaction()
reaction_ab{ab_type}mu_sas_to_isf.setId("ab{ab_type}mu_sas_to_isf")
reaction_ab{ab_type}mu_sas_to_isf.setReversible(False)

# Reactant: AB{ab_type}Mu_SAS
reactant_ab{ab_type}mu_sas_isf = reaction_ab{ab_type}mu_sas_to_isf.createReactant()
reactant_ab{ab_type}mu_sas_isf.setSpecies("AB{ab_type}Mu_SAS")
reactant_ab{ab_type}mu_sas_isf.setStoichiometry(1.0)
reactant_ab{ab_type}mu_sas_isf.setConstant(True)

# Product: (AB{ab_type}_monomer in ISF) # Unsure of product species
product_ab{ab_type}_monomer_isf = reaction_ab{ab_type}mu_sas_to_isf.createProduct()
product_ab{ab_type}_monomer_isf.setSpecies("AB{ab_type}_Monomer")
product_ab{ab_type}_monomer_isf.setStoichiometry(1.0)
product_ab{ab_type}_monomer_isf.setConstant(True)

# Kinetic law: Q_ISF_brain * AB{ab_type}Mu_SAS 
klaw_ab{ab_type}mu_sas_to_isf = reaction_ab{ab_type}mu_sas_to_isf.createKineticLaw()
math_ast = libsbml.parseL3Formula("Q_ISF_brain * AB{ab_type}Mu_SAS")
klaw_ab{ab_type}mu_sas_to_isf.setMath(math_ast)
            ''',
            # Reaction block 33
            f'''
# 1. Clearance of AB{ab_type}Mb from central compartment
reaction_ab{ab_type}mb_cent_clearance = model.createReaction()
reaction_ab{ab_type}mb_cent_clearance.setId("ab{ab_type}mb_cent_clearance")
reaction_ab{ab_type}mb_cent_clearance.setReversible(False)

# Reactant: AB{ab_type}Mb_Central
reactant_ab{ab_type}mb_cent_clearance = reaction_ab{ab_type}mb_cent_clearance.createReactant()
reactant_ab{ab_type}mb_cent_clearance.setSpecies("AB{ab_type}Mb_Central")
reactant_ab{ab_type}mb_cent_clearance.setStoichiometry(1.0)
reactant_ab{ab_type}mb_cent_clearance.setConstant(True)

# Product: Sink
product_sink = reaction_ab{ab_type}mb_cent_clearance.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}Mb_Central"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)

# Kinetic law: (PK_CL) * AB{ab_type}Mb_Central
klaw_ab{ab_type}mb_cent_clearance = reaction_ab{ab_type}mb_cent_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula("(PK_CL) * AB{ab_type}Mb_Central")
klaw_ab{ab_type}mb_cent_clearance.setMath(math_ast)
            ''',
            # Reaction block 34
            f'''
# 2. Distribution of AB{ab_type}Mb from central to peripheral compartment
reaction_ab{ab_type}mb_cent_to_per = model.createReaction()
reaction_ab{ab_type}mb_cent_to_per.setId("ab{ab_type}mb_cent_to_per")
reaction_ab{ab_type}mb_cent_to_per.setReversible(False)

# Reactant: AB{ab_type}Mb_Central
reactant_ab{ab_type}mb_cent_per = reaction_ab{ab_type}mb_cent_to_per.createReactant()
reactant_ab{ab_type}mb_cent_per.setSpecies("AB{ab_type}Mb_Central")
reactant_ab{ab_type}mb_cent_per.setStoichiometry(1.0)
reactant_ab{ab_type}mb_cent_per.setConstant(True)

# Product: AB{ab_type}Mb_Peripheral
product_ab{ab_type}mb_per_cent = reaction_ab{ab_type}mb_cent_to_per.createProduct()
product_ab{ab_type}mb_per_cent.setSpecies("AB{ab_type}Mb_Peripheral")
product_ab{ab_type}mb_per_cent.setStoichiometry(1.0)
product_ab{ab_type}mb_per_cent.setConstant(True)

# Kinetic law: (PK_CLd2) * AB{ab_type}Mb_Central
klaw_ab{ab_type}mb_cent_to_per = reaction_ab{ab_type}mb_cent_to_per.createKineticLaw()
math_ast = libsbml.parseL3Formula("(PK_CLd2) * AB{ab_type}Mb_Central")
klaw_ab{ab_type}mb_cent_to_per.setMath(math_ast)
            ''',
            # Reaction block 35
            f'''
# 3. Distribution of AB{ab_type}Mb from peripheral to central compartment
reaction_ab{ab_type}mb_per_to_cent = model.createReaction()
reaction_ab{ab_type}mb_per_to_cent.setId("ab{ab_type}mb_per_to_cent")
reaction_ab{ab_type}mb_per_to_cent.setReversible(False)

# Reactant: AB{ab_type}Mb_Peripheral
reactant_ab{ab_type}mb_per_cent = reaction_ab{ab_type}mb_per_to_cent.createReactant()
reactant_ab{ab_type}mb_per_cent.setSpecies("AB{ab_type}Mb_Peripheral")
reactant_ab{ab_type}mb_per_cent.setStoichiometry(1.0)
reactant_ab{ab_type}mb_per_cent.setConstant(True)

# Product: AB{ab_type}Mb_Central
product_ab{ab_type}mb_cent_per = reaction_ab{ab_type}mb_per_to_cent.createProduct()
product_ab{ab_type}mb_cent_per.setSpecies("AB{ab_type}Mb_Central")
product_ab{ab_type}mb_cent_per.setStoichiometry(1.0)
product_ab{ab_type}mb_cent_per.setConstant(True)

# Kinetic law: (PK_CLd2) * AB{ab_type}Mb_Peripheral
klaw_ab{ab_type}mb_per_to_cent = reaction_ab{ab_type}mb_per_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(PK_CLd2) * AB{ab_type}Mb_Peripheral")
klaw_ab{ab_type}mb_per_to_cent.setMath(math_ast)
            ''',
            # Reaction block 36
            f'''
# 4. Flow of AB{ab_type}Mb from central compartment to brain plasma
reaction_ab{ab_type}mb_cent_to_brain_plasma = model.createReaction()
reaction_ab{ab_type}mb_cent_to_brain_plasma.setId("ab{ab_type}mb_cent_to_brain_plasma")
reaction_ab{ab_type}mb_cent_to_brain_plasma.setReversible(False)

# Reactant: AB{ab_type}Mb_Central
reactant_ab{ab_type}mb_cent_brain = reaction_ab{ab_type}mb_cent_to_brain_plasma.createReactant()
reactant_ab{ab_type}mb_cent_brain.setSpecies("AB{ab_type}Mb_Central")
reactant_ab{ab_type}mb_cent_brain.setStoichiometry(1.0)
reactant_ab{ab_type}mb_cent_brain.setConstant(True)

# Product: AB{ab_type}Mb_Brain_Plasma
product_ab{ab_type}mb_brain_cent = reaction_ab{ab_type}mb_cent_to_brain_plasma.createProduct()
product_ab{ab_type}mb_brain_cent.setSpecies("AB{ab_type}Mb_Brain_Plasma")
product_ab{ab_type}mb_brain_cent.setStoichiometry(1.0)
product_ab{ab_type}mb_brain_cent.setConstant(True)

# Kinetic law: Q_p_brain * (AB{ab_type}Mb_Central)
klaw_ab{ab_type}mb_cent_to_brain_plasma = reaction_ab{ab_type}mb_cent_to_brain_plasma.createKineticLaw()
math_ast = libsbml.parseL3Formula("Q_p_brain * (AB{ab_type}Mb_Central)")
klaw_ab{ab_type}mb_cent_to_brain_plasma.setMath(math_ast)
            ''',
            # Reaction block 37
            f'''
# 5. Flow of AB{ab_type}Mb from brain plasma to central compartment
reaction_ab{ab_type}mb_brain_plasma_to_cent = model.createReaction()
reaction_ab{ab_type}mb_brain_plasma_to_cent.setId("ab{ab_type}mb_brain_plasma_to_cent")
reaction_ab{ab_type}mb_brain_plasma_to_cent.setReversible(False)

# Reactant: AB{ab_type}Mb_Brain_Plasma
reactant_ab{ab_type}mb_brain_cent = reaction_ab{ab_type}mb_brain_plasma_to_cent.createReactant()
reactant_ab{ab_type}mb_brain_cent.setSpecies("AB{ab_type}Mb_Brain_Plasma")
reactant_ab{ab_type}mb_brain_cent.setStoichiometry(1.0)
reactant_ab{ab_type}mb_brain_cent.setConstant(True)

# Product: AB{ab_type}Mb_Central
product_ab{ab_type}mb_cent_brain = reaction_ab{ab_type}mb_brain_plasma_to_cent.createProduct()
product_ab{ab_type}mb_cent_brain.setSpecies("AB{ab_type}Mb_Central")
product_ab{ab_type}mb_cent_brain.setStoichiometry(1.0)
product_ab{ab_type}mb_cent_brain.setConstant(True)

# Kinetic law: (Q_p_brain - L_brain) * AB{ab_type}Mb_Brain_Plasma
klaw_ab{ab_type}mb_brain_plasma_to_cent = reaction_ab{ab_type}mb_brain_plasma_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(Q_p_brain - L_brain) * AB{ab_type}Mb_Brain_Plasma")
klaw_ab{ab_type}mb_brain_plasma_to_cent.setMath(math_ast)
            ''',
            # Reaction block 38
            f'''
# 6. Flow of AB{ab_type}Mb from CSF SAS to central compartment
reaction_ab{ab_type}mb_csf_sas_to_cent = model.createReaction()
reaction_ab{ab_type}mb_csf_sas_to_cent.setId("ab{ab_type}mb_csf_sas_to_cent")
reaction_ab{ab_type}mb_csf_sas_to_cent.setReversible(False)

# Reactant: AB{ab_type}Mb_SAS
reactant_ab{ab_type}mb_sas_cent = reaction_ab{ab_type}mb_csf_sas_to_cent.createReactant()
reactant_ab{ab_type}mb_sas_cent.setSpecies("AB{ab_type}Mb_SAS")
reactant_ab{ab_type}mb_sas_cent.setStoichiometry(1.0)
reactant_ab{ab_type}mb_sas_cent.setConstant(True)

# Product: AB{ab_type}Mb_Central
product_ab{ab_type}mb_cent_sas = reaction_ab{ab_type}mb_csf_sas_to_cent.createProduct()
product_ab{ab_type}mb_cent_sas.setSpecies("AB{ab_type}Mb_Central")
product_ab{ab_type}mb_cent_sas.setStoichiometry(1.0)
product_ab{ab_type}mb_cent_sas.setConstant(True)

# Kinetic law: (1 - sigma_L_SAS) * Q_CSF_brain * AB{ab_type}Mb_SAS
klaw_ab{ab_type}mb_csf_sas_to_cent = reaction_ab{ab_type}mb_csf_sas_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - sigma_L_SAS) * Q_CSF_brain * AB{ab_type}Mb_SAS")
klaw_ab{ab_type}mb_csf_sas_to_cent.setMath(math_ast)
            ''',
            # Reaction block 39
            f'''
# 7. Flow of AB{ab_type}Mb from ISF (antibody bound) to central compartment
reaction_ab{ab_type}mb_isf_to_cent = model.createReaction()
reaction_ab{ab_type}mb_isf_to_cent.setId("ab{ab_type}mb_isf_to_cent")
reaction_ab{ab_type}mb_isf_to_cent.setReversible(False)

# Reactant: AB{ab_type}_monomer_antibody_bound
reactant_ab{ab_type}mb_isf_cent = reaction_ab{ab_type}mb_isf_to_cent.createReactant()
reactant_ab{ab_type}mb_isf_cent.setSpecies("AB{ab_type}_monomer_antibody_bound")
reactant_ab{ab_type}mb_isf_cent.setStoichiometry(1.0)
reactant_ab{ab_type}mb_isf_cent.setConstant(True)

# Product: AB{ab_type}Mb_Central
product_ab{ab_type}mb_cent_isf = reaction_ab{ab_type}mb_isf_to_cent.createProduct()
product_ab{ab_type}mb_cent_isf.setSpecies("AB{ab_type}Mb_Central")
product_ab{ab_type}mb_cent_isf.setStoichiometry(1.0)
product_ab{ab_type}mb_cent_isf.setConstant(True)

# Kinetic law: (1 - sigma_L_brain_ISF) * Q_ISF_brain * AB{ab_type}_monomer_antibody_bound
klaw_ab{ab_type}mb_isf_to_cent = reaction_ab{ab_type}mb_isf_to_cent.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - sigma_L_brain_ISF) * Q_ISF_brain * AB{ab_type}_monomer_antibody_bound")
klaw_ab{ab_type}mb_isf_to_cent.setMath(math_ast)
            ''',
            # Reaction block 40
            f'''
# 3a. Flow of AB{ab_type}Mb from brain plasma to CSF LV
reaction_ab{ab_type}mb_brain_plasma_to_csf_lv = model.createReaction()
reaction_ab{ab_type}mb_brain_plasma_to_csf_lv.setId("ab{ab_type}mb_brain_plasma_to_csf_lv")
reaction_ab{ab_type}mb_brain_plasma_to_csf_lv.setReversible(False)

# Reactant: AB{ab_type}Mb_Brain_Plasma
reactant_ab{ab_type}mb_brain_csf_lv = reaction_ab{ab_type}mb_brain_plasma_to_csf_lv.createReactant()
reactant_ab{ab_type}mb_brain_csf_lv.setSpecies("AB{ab_type}Mb_Brain_Plasma")
reactant_ab{ab_type}mb_brain_csf_lv.setStoichiometry(1.0)
reactant_ab{ab_type}mb_brain_csf_lv.setConstant(True)

# Product: AB{ab_type}Mb_LV
product_ab{ab_type}mb_csf_lv_brain = reaction_ab{ab_type}mb_brain_plasma_to_csf_lv.createProduct()
product_ab{ab_type}mb_csf_lv_brain.setSpecies("AB{ab_type}Mb_LV")
product_ab{ab_type}mb_csf_lv_brain.setStoichiometry(1.0)
product_ab{ab_type}mb_csf_lv_brain.setConstant(True)

# Kinetic law: f_LV * (1 - sigma_V_BCSFB) * Q_CSF_brain * (AB{ab_type}Mb_Brain_Plasma)
klaw_ab{ab_type}mb_brain_plasma_to_csf_lv = reaction_ab{ab_type}mb_brain_plasma_to_csf_lv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * (1 - sigma_V_BCSFB) * Q_CSF_brain * (AB{ab_type}Mb_Brain_Plasma)")
klaw_ab{ab_type}mb_brain_plasma_to_csf_lv.setMath(math_ast)
            ''',
            # Reaction block 41
            f'''
# 3b. Flow of AB{ab_type}Mb from brain plasma to CSF TFV
reaction_ab{ab_type}mb_brain_plasma_to_csf_tfv = model.createReaction()
reaction_ab{ab_type}mb_brain_plasma_to_csf_tfv.setId("ab{ab_type}mb_brain_plasma_to_csf_tfv")
reaction_ab{ab_type}mb_brain_plasma_to_csf_tfv.setReversible(False)

# Reactant: AB{ab_type}Mb_Brain_Plasma
reactant_ab{ab_type}mb_brain_csf_tfv = reaction_ab{ab_type}mb_brain_plasma_to_csf_tfv.createReactant()
reactant_ab{ab_type}mb_brain_csf_tfv.setSpecies("AB{ab_type}Mb_Brain_Plasma")
reactant_ab{ab_type}mb_brain_csf_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}mb_brain_csf_tfv.setConstant(True)

# Product: AB{ab_type}Mb_TFV
product_ab{ab_type}mb_csf_tfv_brain = reaction_ab{ab_type}mb_brain_plasma_to_csf_tfv.createProduct()
product_ab{ab_type}mb_csf_tfv_brain.setSpecies("AB{ab_type}Mb_TFV")
product_ab{ab_type}mb_csf_tfv_brain.setStoichiometry(1.0)
product_ab{ab_type}mb_csf_tfv_brain.setConstant(True)

# Kinetic law: (1 - f_LV) * (1 - sigma_V_BCSFB) * Q_CSF_brain * (AB{ab_type}Mb_Brain_Plasma)
klaw_ab{ab_type}mb_brain_plasma_to_csf_tfv = reaction_ab{ab_type}mb_brain_plasma_to_csf_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * (1 - sigma_V_BCSFB) * Q_CSF_brain * (AB{ab_type}Mb_Brain_Plasma)")
klaw_ab{ab_type}mb_brain_plasma_to_csf_tfv.setMath(math_ast)
            ''',
            # Reaction block 42
            f'''
# 4. Flow of AB{ab_type}Mb from brain plasma to ISF
reaction_ab{ab_type}mb_brain_plasma_to_isf = model.createReaction()
reaction_ab{ab_type}mb_brain_plasma_to_isf.setId("ab{ab_type}mb_brain_plasma_to_isf")
reaction_ab{ab_type}mb_brain_plasma_to_isf.setReversible(False)

# Reactant: AB{ab_type}Mb_Brain_Plasma
reactant_ab{ab_type}mb_brain_isf = reaction_ab{ab_type}mb_brain_plasma_to_isf.createReactant()
reactant_ab{ab_type}mb_brain_isf.setSpecies("AB{ab_type}Mb_Brain_Plasma")
reactant_ab{ab_type}mb_brain_isf.setStoichiometry(1.0)
reactant_ab{ab_type}mb_brain_isf.setConstant(True)

# Product: AB{ab_type}_monomer_antibody_bound (ISF) # Unsure if this is correct
product_ab{ab_type}mb_isf_brain = reaction_ab{ab_type}mb_brain_plasma_to_isf.createProduct()
product_ab{ab_type}mb_isf_brain.setSpecies("AB{ab_type}_monomer_antibody_bound")
product_ab{ab_type}mb_isf_brain.setStoichiometry(1.0)
product_ab{ab_type}mb_isf_brain.setConstant(True)

# Kinetic law: (1 - sigma_V_brain_ISF) * Q_ISF_brain * (AB{ab_type}Mb_Brain_Plasma)
klaw_ab{ab_type}mb_brain_plasma_to_isf = reaction_ab{ab_type}mb_brain_plasma_to_isf.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - sigma_V_brain_ISF) * Q_ISF_brain * (AB{ab_type}Mb_Brain_Plasma)")
klaw_ab{ab_type}mb_brain_plasma_to_isf.setMath(math_ast)
            ''',
            # Reaction block 43
            f'''
# 5. Uptake of AB{ab_type}Mb from brain plasma to BCSFB
reaction_ab{ab_type}mb_brain_plasma_to_bcsfb = model.createReaction()
reaction_ab{ab_type}mb_brain_plasma_to_bcsfb.setId("ab{ab_type}mb_brain_plasma_to_bcsfb")
reaction_ab{ab_type}mb_brain_plasma_to_bcsfb.setReversible(False)

# Reactant: AB{ab_type}Mb_Brain_Plasma
reactant_ab{ab_type}mb_brain_bcsfb = reaction_ab{ab_type}mb_brain_plasma_to_bcsfb.createReactant()
reactant_ab{ab_type}mb_brain_bcsfb.setSpecies("AB{ab_type}Mb_Brain_Plasma")
reactant_ab{ab_type}mb_brain_bcsfb.setStoichiometry(1.0)
reactant_ab{ab_type}mb_brain_bcsfb.setConstant(True)

# Product: AB{ab_type}Mb_BCSFB_Unbound
product_ab{ab_type}mb_bcsfb_brain = reaction_ab{ab_type}mb_brain_plasma_to_bcsfb.createProduct()
product_ab{ab_type}mb_bcsfb_brain.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
product_ab{ab_type}mb_bcsfb_brain.setStoichiometry(1.0)
product_ab{ab_type}mb_bcsfb_brain.setConstant(True)

# Kinetic law: CLup_brain * f_BCSFB * VES_brain * AB{ab_type}Mb_Brain_Plasma
klaw_ab{ab_type}mb_brain_plasma_to_bcsfb = reaction_ab{ab_type}mb_brain_plasma_to_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain * f_BCSFB * VES_brain * AB{ab_type}Mb_Brain_Plasma")
klaw_ab{ab_type}mb_brain_plasma_to_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 44
            f'''
# 6. Uptake of AB{ab_type}Mb from brain plasma to BBB
reaction_ab{ab_type}mb_brain_plasma_to_bbb = model.createReaction()
reaction_ab{ab_type}mb_brain_plasma_to_bbb.setId("ab{ab_type}mb_brain_plasma_to_bbb")
reaction_ab{ab_type}mb_brain_plasma_to_bbb.setReversible(False)

# Reactant: AB{ab_type}Mb_Brain_Plasma
reactant_ab{ab_type}mb_brain_bbb = reaction_ab{ab_type}mb_brain_plasma_to_bbb.createReactant()
reactant_ab{ab_type}mb_brain_bbb.setSpecies("AB{ab_type}Mb_Brain_Plasma")
reactant_ab{ab_type}mb_brain_bbb.setStoichiometry(1.0)
reactant_ab{ab_type}mb_brain_bbb.setConstant(True)

# Product: AB{ab_type}Mb_BBB_Unbound
product_ab{ab_type}mb_bbb_brain = reaction_ab{ab_type}mb_brain_plasma_to_bbb.createProduct()
product_ab{ab_type}mb_bbb_brain.setSpecies("AB{ab_type}Mb_BBB_Unbound")
product_ab{ab_type}mb_bbb_brain.setStoichiometry(1.0)
product_ab{ab_type}mb_bbb_brain.setConstant(True)

# Kinetic law: CLup_brain * f_BBB * VES_brain * AB{ab_type}Mb_Brain_Plasma
klaw_ab{ab_type}mb_brain_plasma_to_bbb = reaction_ab{ab_type}mb_brain_plasma_to_bbb.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * AB{ab_type}Mb_Brain_Plasma")
klaw_ab{ab_type}mb_brain_plasma_to_bbb.setMath(math_ast)
            ''',
            # Reaction block 45
            f'''
# 7. Recycling of AB{ab_type}Mb from BBB to brain plasma
reaction_ab{ab_type}mb_bbb_to_brain_plasma = model.createReaction()
reaction_ab{ab_type}mb_bbb_to_brain_plasma.setId("ab{ab_type}mb_bbb_to_brain_plasma")
reaction_ab{ab_type}mb_bbb_to_brain_plasma.setReversible(False)

# Reactant: AB{ab_type}Mb_BBB_Bound
reactant_ab{ab_type}mb_bbb_brain_recycling = reaction_ab{ab_type}mb_bbb_to_brain_plasma.createReactant()
reactant_ab{ab_type}mb_bbb_brain_recycling.setSpecies("AB{ab_type}Mb_BBB_Bound")
reactant_ab{ab_type}mb_bbb_brain_recycling.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bbb_brain_recycling.setConstant(True)

# Product: AB{ab_type}Mb_Brain_Plasma
product_ab{ab_type}mb_brain_bbb_recycling = reaction_ab{ab_type}mb_bbb_to_brain_plasma.createProduct()
product_ab{ab_type}mb_brain_bbb_recycling.setSpecies("AB{ab_type}Mb_Brain_Plasma")
product_ab{ab_type}mb_brain_bbb_recycling.setStoichiometry(1.0)
product_ab{ab_type}mb_brain_bbb_recycling.setConstant(True)

product_fcrn_free_bind = reaction_ab{ab_type}mb_bbb_to_brain_plasma.createProduct()
product_fcrn_free_bind.setSpecies("FcRn_free_BBB")
product_fcrn_free_bind.setStoichiometry(1.0)
product_fcrn_free_bind.setConstant(True)

# Kinetic law: CLup_brain * f_BBB * VES_brain * FR * AB{ab_type}Mb_BBB_Bound
klaw_ab{ab_type}mb_bbb_to_brain_plasma = reaction_ab{ab_type}mb_bbb_to_brain_plasma.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * FR * AB{ab_type}Mb_BBB_Bound")
klaw_ab{ab_type}mb_bbb_to_brain_plasma.setMath(math_ast)
            ''',
            # Reaction block 46
            f'''
# 8. Recycling of AB{ab_type}Mb from BCSFB to brain plasma
reaction_ab{ab_type}mb_bcsfb_to_brain_plasma = model.createReaction()
reaction_ab{ab_type}mb_bcsfb_to_brain_plasma.setId("ab{ab_type}mb_bcsfb_to_brain_plasma")
reaction_ab{ab_type}mb_bcsfb_to_brain_plasma.setReversible(False)

# Reactant: AB{ab_type}Mb_BCSFB_Bound
reactant_ab{ab_type}mb_bcsfb_brain_recycling = reaction_ab{ab_type}mb_bcsfb_to_brain_plasma.createReactant()
reactant_ab{ab_type}mb_bcsfb_brain_recycling.setSpecies("AB{ab_type}Mb_BCSFB_Bound")
reactant_ab{ab_type}mb_bcsfb_brain_recycling.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bcsfb_brain_recycling.setConstant(True)

# Product: AB{ab_type}Mb_Brain_Plasma
product_ab{ab_type}mb_brain_bcsfb_recycling = reaction_ab{ab_type}mb_bcsfb_to_brain_plasma.createProduct()
product_ab{ab_type}mb_brain_bcsfb_recycling.setSpecies("AB{ab_type}Mb_Brain_Plasma")
product_ab{ab_type}mb_brain_bcsfb_recycling.setStoichiometry(1.0)
product_ab{ab_type}mb_brain_bcsfb_recycling.setConstant(True)

product_fcrn_free_bind = reaction_ab{ab_type}mb_bcsfb_to_brain_plasma.createProduct()
product_fcrn_free_bind.setSpecies("FcRn_free_BCSFB")
product_fcrn_free_bind.setStoichiometry(1.0)
product_fcrn_free_bind.setConstant(True)

# Kinetic law: CLup_brain * (1 - f_BBB) * VES_brain * FR * AB{ab_type}Mb_BCSFB_Bound
klaw_ab{ab_type}mb_bcsfb_to_brain_plasma = reaction_ab{ab_type}mb_bcsfb_to_brain_plasma.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain * (1 - f_BBB) * VES_brain * FR * AB{ab_type}Mb_BCSFB_Bound")
klaw_ab{ab_type}mb_bcsfb_to_brain_plasma.setMath(math_ast)
            ''',
            # Reaction block 47
            f'''
# 1. Flow of AB{ab_type}Mb from CSF TFV to BCSFB unbound
reaction_ab{ab_type}mb_csf_tfv_to_bcsfb = model.createReaction()
reaction_ab{ab_type}mb_csf_tfv_to_bcsfb.setId("ab{ab_type}mb_csf_tfv_to_bcsfb")
reaction_ab{ab_type}mb_csf_tfv_to_bcsfb.setReversible(False)

# Reactant: AB{ab_type}Mb_TFV
reactant_ab{ab_type}mb_tfv_bcsfb = reaction_ab{ab_type}mb_csf_tfv_to_bcsfb.createReactant()
reactant_ab{ab_type}mb_tfv_bcsfb.setSpecies("AB{ab_type}Mb_TFV")
reactant_ab{ab_type}mb_tfv_bcsfb.setStoichiometry(1.0)
reactant_ab{ab_type}mb_tfv_bcsfb.setConstant(True)

# Product: AB{ab_type}Mb_BCSFB_Unbound
product_ab{ab_type}mb_bcsfb_tfv = reaction_ab{ab_type}mb_csf_tfv_to_bcsfb.createProduct()
product_ab{ab_type}mb_bcsfb_tfv.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
product_ab{ab_type}mb_bcsfb_tfv.setStoichiometry(1.0)
product_ab{ab_type}mb_bcsfb_tfv.setConstant(True)

# Kinetic law: (1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * AB{ab_type}Mb_TFV
klaw_ab{ab_type}mb_csf_tfv_to_bcsfb = reaction_ab{ab_type}mb_csf_tfv_to_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * AB{ab_type}Mb_TFV")
klaw_ab{ab_type}mb_csf_tfv_to_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 48
            f'''
# 2. Flow of AB{ab_type}Mb from CSF LV to BCSFB unbound
reaction_ab{ab_type}mb_csf_lv_to_bcsfb = model.createReaction()
reaction_ab{ab_type}mb_csf_lv_to_bcsfb.setId("ab{ab_type}mb_csf_lv_to_bcsfb")
reaction_ab{ab_type}mb_csf_lv_to_bcsfb.setReversible(False)

# Reactant: AB{ab_type}Mb_LV
reactant_ab{ab_type}mb_lv_bcsfb = reaction_ab{ab_type}mb_csf_lv_to_bcsfb.createReactant()
reactant_ab{ab_type}mb_lv_bcsfb.setSpecies("AB{ab_type}Mb_LV")
reactant_ab{ab_type}mb_lv_bcsfb.setStoichiometry(1.0)
reactant_ab{ab_type}mb_lv_bcsfb.setConstant(True)

# Product: AB{ab_type}Mb_BCSFB_Unbound
product_ab{ab_type}mb_bcsfb_lv = reaction_ab{ab_type}mb_csf_lv_to_bcsfb.createProduct()
product_ab{ab_type}mb_bcsfb_lv.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
product_ab{ab_type}mb_bcsfb_lv.setStoichiometry(1.0)
product_ab{ab_type}mb_bcsfb_lv.setConstant(True)

# Kinetic law: f_LV * CLup_brain * (1 - f_BBB) * VES_brain * AB{ab_type}Mb_LV
klaw_ab{ab_type}mb_csf_lv_to_bcsfb = reaction_ab{ab_type}mb_csf_lv_to_bcsfb.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * CLup_brain * (1 - f_BBB) * VES_brain * AB{ab_type}Mb_LV")
klaw_ab{ab_type}mb_csf_lv_to_bcsfb.setMath(math_ast)
            ''',
            # Reaction block 49
            f'''
# 4. Degradation of AB{ab_type}Mb in BCSFB unbound
reaction_ab{ab_type}mb_bcsfb_unbound_degradation = model.createReaction()
reaction_ab{ab_type}mb_bcsfb_unbound_degradation.setId("ab{ab_type}mb_bcsfb_unbound_degradation")
reaction_ab{ab_type}mb_bcsfb_unbound_degradation.setReversible(False)

# Reactant: AB{ab_type}Mb_BCSFB_Unbound
reactant_ab{ab_type}mb_bcsfb_unbound_deg = reaction_ab{ab_type}mb_bcsfb_unbound_degradation.createReactant()
reactant_ab{ab_type}mb_bcsfb_unbound_deg.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
reactant_ab{ab_type}mb_bcsfb_unbound_deg.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bcsfb_unbound_deg.setConstant(True)

# Product: Sink
product_sink = reaction_ab{ab_type}mb_bcsfb_unbound_degradation.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}Mb_BCSFB_Unbound"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)


# Kinetic law: kdeg * AB{ab_type}Mb_BCSFB_Unbound
klaw_ab{ab_type}mb_bcsfb_unbound_degradation = reaction_ab{ab_type}mb_bcsfb_unbound_degradation.createKineticLaw()
math_ast = libsbml.parseL3Formula("kdeg * AB{ab_type}Mb_BCSFB_Unbound * V_BCSFB_brain")
klaw_ab{ab_type}mb_bcsfb_unbound_degradation.setMath(math_ast)
            ''',
            # Reaction block 50
            f'''
# 5. Binding of AB{ab_type}Mb BCSFB unbound to FcRn (to bound)
reaction_ab{ab_type}mb_bcsfb_unbound_to_bound = model.createReaction()
reaction_ab{ab_type}mb_bcsfb_unbound_to_bound.setId("ab{ab_type}mb_bcsfb_unbound_to_bound")
reaction_ab{ab_type}mb_bcsfb_unbound_to_bound.setReversible(False)

# Reactants: AB{ab_type}Mb_BCSFB_Unbound and FcRn_free_BCSFB
reactant_ab{ab_type}mb_bcsfb_unbound = reaction_ab{ab_type}mb_bcsfb_unbound_to_bound.createReactant()
reactant_ab{ab_type}mb_bcsfb_unbound.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
reactant_ab{ab_type}mb_bcsfb_unbound.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bcsfb_unbound.setConstant(True)

reactant_fcrn_free_bind = reaction_ab{ab_type}mb_bcsfb_unbound_to_bound.createReactant()
reactant_fcrn_free_bind.setSpecies("FcRn_free_BCSFB")
reactant_fcrn_free_bind.setStoichiometry(1.0)
reactant_fcrn_free_bind.setConstant(True)

# Product: AB{ab_type}Mb_BCSFB_Bound
product_ab{ab_type}mb_bcsfb_bound = reaction_ab{ab_type}mb_bcsfb_unbound_to_bound.createProduct()
product_ab{ab_type}mb_bcsfb_bound.setSpecies("AB{ab_type}Mb_BCSFB_Bound")
product_ab{ab_type}mb_bcsfb_bound.setStoichiometry(1.0)
product_ab{ab_type}mb_bcsfb_bound.setConstant(True)

# Kinetic law: kon_FcRn * AB{ab_type}Mb_BCSFB_Unbound * FcRn_free_BCSFB
klaw_ab{ab_type}mb_bcsfb_unbound_to_bound = reaction_ab{ab_type}mb_bcsfb_unbound_to_bound.createKineticLaw()
math_ast = libsbml.parseL3Formula("kon_FcRn * AB{ab_type}Mb_BCSFB_Unbound * FcRn_free_BCSFB * V_BCSFB_brain")
klaw_ab{ab_type}mb_bcsfb_unbound_to_bound.setMath(math_ast)
            ''',
            # Reaction block 51
            f'''
# 6. Unbinding of AB{ab_type}Mb BCSFB bound from FcRn (to unbound)
reaction_ab{ab_type}mb_bcsfb_bound_to_unbound = model.createReaction()
reaction_ab{ab_type}mb_bcsfb_bound_to_unbound.setId("ab{ab_type}mb_bcsfb_bound_to_unbound")
reaction_ab{ab_type}mb_bcsfb_bound_to_unbound.setReversible(False)

# Reactant: AB{ab_type}Mb_BCSFB_Bound
reactant_ab{ab_type}mb_bcsfb_bound = reaction_ab{ab_type}mb_bcsfb_bound_to_unbound.createReactant()
reactant_ab{ab_type}mb_bcsfb_bound.setSpecies("AB{ab_type}Mb_BCSFB_Bound")
reactant_ab{ab_type}mb_bcsfb_bound.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bcsfb_bound.setConstant(True)

# Products: AB{ab_type}Mb_BCSFB_Unbound and FcRn_free_BCSFB
product_ab{ab_type}mb_bcsfb_unbound = reaction_ab{ab_type}mb_bcsfb_bound_to_unbound.createProduct()
product_ab{ab_type}mb_bcsfb_unbound.setSpecies("AB{ab_type}Mb_BCSFB_Unbound")
product_ab{ab_type}mb_bcsfb_unbound.setStoichiometry(1.0)
product_ab{ab_type}mb_bcsfb_unbound.setConstant(True)

product_fcrn_free_from_bcsfb = reaction_ab{ab_type}mb_bcsfb_bound_to_unbound.createProduct()
product_fcrn_free_from_bcsfb.setSpecies("FcRn_free_BCSFB")
product_fcrn_free_from_bcsfb.setStoichiometry(1.0)
product_fcrn_free_from_bcsfb.setConstant(True)


# Kinetic law: koff_FcRn * AB{ab_type}Mb_BCSFB_Bound
klaw_ab{ab_type}mb_bcsfb_bound_to_unbound = reaction_ab{ab_type}mb_bcsfb_bound_to_unbound.createKineticLaw()
math_ast = libsbml.parseL3Formula("koff_FcRn * AB{ab_type}Mb_BCSFB_Bound * V_BCSFB_brain")
klaw_ab{ab_type}mb_bcsfb_bound_to_unbound.setMath(math_ast)
            ''',
            # Reaction block 52
            f'''
# 2. Flow of AB{ab_type}Mb from BCSFB bound to CSF LV
reaction_ab{ab_type}mb_bcsfb_bound_to_lv = model.createReaction()
reaction_ab{ab_type}mb_bcsfb_bound_to_lv.setId("ab{ab_type}mb_bcsfb_bound_to_lv")
reaction_ab{ab_type}mb_bcsfb_bound_to_lv.setReversible(False)

# Reactant: AB{ab_type}Mb_BCSFB_Bound
reactant_ab{ab_type}mb_bcsfb_bound_lv = reaction_ab{ab_type}mb_bcsfb_bound_to_lv.createReactant()
reactant_ab{ab_type}mb_bcsfb_bound_lv.setSpecies("AB{ab_type}Mb_BCSFB_Bound")
reactant_ab{ab_type}mb_bcsfb_bound_lv.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bcsfb_bound_lv.setConstant(True)

# Product: AB{ab_type}Mb_LV
product_ab{ab_type}mb_lv_bcsfb_bound = reaction_ab{ab_type}mb_bcsfb_bound_to_lv.createProduct()
product_ab{ab_type}mb_lv_bcsfb_bound.setSpecies("AB{ab_type}Mb_LV")
product_ab{ab_type}mb_lv_bcsfb_bound.setStoichiometry(1.0)
product_ab{ab_type}mb_lv_bcsfb_bound.setConstant(True)

product_fcrn_free_from_bcsfb_lv = reaction_ab{ab_type}mb_bcsfb_bound_to_lv.createProduct()
product_fcrn_free_from_bcsfb_lv.setSpecies("FcRn_free_BCSFB")
product_fcrn_free_from_bcsfb_lv.setStoichiometry(1.0)
product_fcrn_free_from_bcsfb_lv.setConstant(True)

# Kinetic law: f_LV * CLup_brain * (1 - f_BBB) * VES_brain * (1 - FR) * AB{ab_type}Mb_BCSFB_Bound
klaw_ab{ab_type}mb_bcsfb_bound_to_lv = reaction_ab{ab_type}mb_bcsfb_bound_to_lv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * CLup_brain * (1 - f_BBB) * VES_brain * (1 - FR) * AB{ab_type}Mb_BCSFB_Bound")
klaw_ab{ab_type}mb_bcsfb_bound_to_lv.setMath(math_ast)
            ''',
            # Reaction block 53
            f'''
# 3. Flow of AB{ab_type}Mb from BCSFB bound to CSF TFV
reaction_ab{ab_type}mb_bcsfb_bound_to_tfv = model.createReaction()
reaction_ab{ab_type}mb_bcsfb_bound_to_tfv.setId("ab{ab_type}mb_bcsfb_bound_to_tfv")
reaction_ab{ab_type}mb_bcsfb_bound_to_tfv.setReversible(False)

# Reactant: AB{ab_type}Mb_BCSFB_Bound
reactant_ab{ab_type}mb_bcsfb_bound_tfv = reaction_ab{ab_type}mb_bcsfb_bound_to_tfv.createReactant()
reactant_ab{ab_type}mb_bcsfb_bound_tfv.setSpecies("AB{ab_type}Mb_BCSFB_Bound")
reactant_ab{ab_type}mb_bcsfb_bound_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bcsfb_bound_tfv.setConstant(True)

# Product: AB{ab_type}Mb_TFV
product_ab{ab_type}mb_tfv_bcsfb_bound = reaction_ab{ab_type}mb_bcsfb_bound_to_tfv.createProduct()
product_ab{ab_type}mb_tfv_bcsfb_bound.setSpecies("AB{ab_type}Mb_TFV")
product_ab{ab_type}mb_tfv_bcsfb_bound.setStoichiometry(1.0)
product_ab{ab_type}mb_tfv_bcsfb_bound.setConstant(True)

product_fcrn_free_from_bcsfb_tfv = reaction_ab{ab_type}mb_bcsfb_bound_to_tfv.createProduct()
product_fcrn_free_from_bcsfb_tfv.setSpecies("FcRn_free_BCSFB")
product_fcrn_free_from_bcsfb_tfv.setStoichiometry(1.0)
product_fcrn_free_from_bcsfb_tfv.setConstant(True)

# Kinetic law: (1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * (1 - FR) * AB{ab_type}Mb_BCSFB_Bound
klaw_ab{ab_type}mb_bcsfb_bound_to_tfv = reaction_ab{ab_type}mb_bcsfb_bound_to_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * CLup_brain * (1 - f_BBB) * VES_brain * (1 - FR) * AB{ab_type}Mb_BCSFB_Bound")
klaw_ab{ab_type}mb_bcsfb_bound_to_tfv.setMath(math_ast)
            ''',
            # Reaction block 54
            f'''
# 1. Uptake of AB{ab_type}Mb from ISF (antibody bound) to BBB unbound
reaction_ab{ab_type}mb_isf_to_bbb = model.createReaction()
reaction_ab{ab_type}mb_isf_to_bbb.setId("ab{ab_type}mb_isf_to_bbb")
reaction_ab{ab_type}mb_isf_to_bbb.setReversible(False)

# Reactant: AB{ab_type}_monomer_antibody_bound (ISF)
reactant_ab{ab_type}mb_isf_bbb = reaction_ab{ab_type}mb_isf_to_bbb.createReactant()
reactant_ab{ab_type}mb_isf_bbb.setSpecies("AB{ab_type}_monomer_antibody_bound")
reactant_ab{ab_type}mb_isf_bbb.setStoichiometry(1.0)
reactant_ab{ab_type}mb_isf_bbb.setConstant(True)

# Product: AB{ab_type}Mb_BBB_Unbound
product_ab{ab_type}mb_bbb_isf = reaction_ab{ab_type}mb_isf_to_bbb.createProduct()
product_ab{ab_type}mb_bbb_isf.setSpecies("AB{ab_type}Mb_BBB_Unbound")
product_ab{ab_type}mb_bbb_isf.setStoichiometry(1.0)
product_ab{ab_type}mb_bbb_isf.setConstant(True)

# Kinetic law: CLup_brain * f_BBB * VES_brain * AB{ab_type}_monomer_antibody_bound
klaw_ab{ab_type}mb_isf_to_bbb = reaction_ab{ab_type}mb_isf_to_bbb.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * AB{ab_type}_monomer_antibody_bound")
klaw_ab{ab_type}mb_isf_to_bbb.setMath(math_ast)
            ''',
            # Reaction block 55
            f'''
# 3. Degradation of AB{ab_type}Mb in BBB unbound
reaction_ab{ab_type}mb_bbb_unbound_degradation = model.createReaction()
reaction_ab{ab_type}mb_bbb_unbound_degradation.setId("ab{ab_type}mb_bbb_unbound_degradation")
reaction_ab{ab_type}mb_bbb_unbound_degradation.setReversible(False)

# Reactant: AB{ab_type}Mb_BBB_Unbound
reactant_ab{ab_type}mb_bbb_unbound_deg = reaction_ab{ab_type}mb_bbb_unbound_degradation.createReactant()
reactant_ab{ab_type}mb_bbb_unbound_deg.setSpecies("AB{ab_type}Mb_BBB_Unbound")
reactant_ab{ab_type}mb_bbb_unbound_deg.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bbb_unbound_deg.setConstant(True)

# Product: Sink
product_sink = reaction_ab{ab_type}mb_bbb_unbound_degradation.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}Mb_BBB_Unbound"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)

# Kinetic law: kdeg * AB{ab_type}Mb_BBB_Unbound
klaw_ab{ab_type}mb_bbb_unbound_degradation = reaction_ab{ab_type}mb_bbb_unbound_degradation.createKineticLaw()
math_ast = libsbml.parseL3Formula("kdeg * AB{ab_type}Mb_BBB_Unbound * VBBB_brain")
klaw_ab{ab_type}mb_bbb_unbound_degradation.setMath(math_ast)
            ''',
            # Reaction block 56
            f'''
# 4. Binding of AB{ab_type}Mb BBB unbound to FcRn (to bound)
reaction_ab{ab_type}mb_bbb_unbound_to_bound = model.createReaction()
reaction_ab{ab_type}mb_bbb_unbound_to_bound.setId("ab{ab_type}mb_bbb_unbound_to_bound")
reaction_ab{ab_type}mb_bbb_unbound_to_bound.setReversible(False)

# Reactants: AB{ab_type}Mb_BBB_Unbound and FcRn_free_BBB
reactant_ab{ab_type}mb_bbb_unbound = reaction_ab{ab_type}mb_bbb_unbound_to_bound.createReactant()
reactant_ab{ab_type}mb_bbb_unbound.setSpecies("AB{ab_type}Mb_BBB_Unbound")
reactant_ab{ab_type}mb_bbb_unbound.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bbb_unbound.setConstant(True)

reactant_fcrn_free_bind = reaction_ab{ab_type}mb_bbb_unbound_to_bound.createReactant()
reactant_fcrn_free_bind.setSpecies("FcRn_free_BBB")
reactant_fcrn_free_bind.setStoichiometry(1.0)
reactant_fcrn_free_bind.setConstant(True)

# Product: AB{ab_type}Mb_BBB_Bound
product_ab{ab_type}mb_bbb_bound = reaction_ab{ab_type}mb_bbb_unbound_to_bound.createProduct()
product_ab{ab_type}mb_bbb_bound.setSpecies("AB{ab_type}Mb_BBB_Bound")
product_ab{ab_type}mb_bbb_bound.setStoichiometry(1.0)
product_ab{ab_type}mb_bbb_bound.setConstant(True)

# Kinetic law: kon_FcRn * AB{ab_type}Mb_BBB_Unbound * FcRn_free_BBB
klaw_ab{ab_type}mb_bbb_unbound_to_bound = reaction_ab{ab_type}mb_bbb_unbound_to_bound.createKineticLaw()
math_ast = libsbml.parseL3Formula("kon_FcRn * AB{ab_type}Mb_BBB_Unbound * FcRn_free_BBB * VBBB_brain")
klaw_ab{ab_type}mb_bbb_unbound_to_bound.setMath(math_ast)
            ''',
            # Reaction block 57
            f'''
# 5. Unbinding of AB{ab_type}Mb BBB bound from FcRn (to unbound)
reaction_ab{ab_type}mb_bbb_bound_to_unbound = model.createReaction()
reaction_ab{ab_type}mb_bbb_bound_to_unbound.setId("ab{ab_type}mb_bbb_bound_to_unbound")
reaction_ab{ab_type}mb_bbb_bound_to_unbound.setReversible(False)

# Reactant: AB{ab_type}Mb_BBB_Bound
reactant_ab{ab_type}mb_bbb_bound = reaction_ab{ab_type}mb_bbb_bound_to_unbound.createReactant()
reactant_ab{ab_type}mb_bbb_bound.setSpecies("AB{ab_type}Mb_BBB_Bound")
reactant_ab{ab_type}mb_bbb_bound.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bbb_bound.setConstant(True)

# Products: AB{ab_type}Mb_BBB_Unbound and FcRn_free_BBB
product_ab{ab_type}mb_bbb_unbound = reaction_ab{ab_type}mb_bbb_bound_to_unbound.createProduct()
product_ab{ab_type}mb_bbb_unbound.setSpecies("AB{ab_type}Mb_BBB_Unbound")
product_ab{ab_type}mb_bbb_unbound.setStoichiometry(1.0)
product_ab{ab_type}mb_bbb_unbound.setConstant(True)

product_fcrn_free_from_bbb = reaction_ab{ab_type}mb_bbb_bound_to_unbound.createProduct()
product_fcrn_free_from_bbb.setSpecies("FcRn_free_BBB")
product_fcrn_free_from_bbb.setStoichiometry(1.0)
product_fcrn_free_from_bbb.setConstant(True)

# Kinetic law: koff_FcRn * AB{ab_type}Mb_BBB_Bound
klaw_ab{ab_type}mb_bbb_bound_to_unbound = reaction_ab{ab_type}mb_bbb_bound_to_unbound.createKineticLaw()
math_ast = libsbml.parseL3Formula("koff_FcRn * AB{ab_type}Mb_BBB_Bound * VBBB_brain")
klaw_ab{ab_type}mb_bbb_bound_to_unbound.setMath(math_ast)
            ''',
            # Reaction block 58
            f'''
# 2. Flow of AB{ab_type}Mb from BBB bound to ISF
reaction_ab{ab_type}mb_bbb_bound_to_isf = model.createReaction()
reaction_ab{ab_type}mb_bbb_bound_to_isf.setId("ab{ab_type}mb_bbb_bound_to_isf")
reaction_ab{ab_type}mb_bbb_bound_to_isf.setReversible(False)

# Reactant: AB{ab_type}Mb_BBB_Bound
reactant_ab{ab_type}mb_bbb_bound_isf = reaction_ab{ab_type}mb_bbb_bound_to_isf.createReactant()
reactant_ab{ab_type}mb_bbb_bound_isf.setSpecies("AB{ab_type}Mb_BBB_Bound")
reactant_ab{ab_type}mb_bbb_bound_isf.setStoichiometry(1.0)
reactant_ab{ab_type}mb_bbb_bound_isf.setConstant(True)

# Product: AB{ab_type}_monomer_antibody_bound
product_ab{ab_type}mb_isf_bbb_bound = reaction_ab{ab_type}mb_bbb_bound_to_isf.createProduct()
product_ab{ab_type}mb_isf_bbb_bound.setSpecies("AB{ab_type}_monomer_antibody_bound")
product_ab{ab_type}mb_isf_bbb_bound.setStoichiometry(1.0)
product_ab{ab_type}mb_isf_bbb_bound.setConstant(True)

product_fcrn_free_from_bbb_isf = reaction_ab{ab_type}mb_bbb_bound_to_isf.createProduct()
product_fcrn_free_from_bbb_isf.setSpecies("FcRn_free_BBB")
product_fcrn_free_from_bbb_isf.setStoichiometry(1.0)
product_fcrn_free_from_bbb_isf.setConstant(True)

# Kinetic law: CLup_brain * f_BBB * VES_brain * (1 - FR) * AB{ab_type}Mb_BBB_Bound
klaw_ab{ab_type}mb_bbb_bound_to_isf = reaction_ab{ab_type}mb_bbb_bound_to_isf.createKineticLaw()
math_ast = libsbml.parseL3Formula("CLup_brain * f_BBB * VES_brain * (1 - FR) * AB{ab_type}Mb_BBB_Bound")
klaw_ab{ab_type}mb_bbb_bound_to_isf.setMath(math_ast)
            ''',
            # Reaction block 59
            f'''
# 1. Degradation/clearance of AB{ab_type}_monomer_antibody_bound
reaction_ab{ab_type}_isf_degradation = model.createReaction()
reaction_ab{ab_type}_isf_degradation.setId("ab{ab_type}_isf_degradation")
reaction_ab{ab_type}_isf_degradation.setReversible(False)

# Reactant: AB{ab_type}_monomer_antibody_bound
reactant_ab{ab_type}_isf_deg = reaction_ab{ab_type}_isf_degradation.createReactant()
reactant_ab{ab_type}_isf_deg.setSpecies("AB{ab_type}_monomer_antibody_bound")
reactant_ab{ab_type}_isf_deg.setStoichiometry(1.0)
reactant_ab{ab_type}_isf_deg.setConstant(True)

# Product: Sink
product_sink = reaction_ab{ab_type}_isf_degradation.createProduct()
product_sink.setSpecies(sinks[f"AB{ab_type}_monomer_antibody_bound"])
product_sink.setStoichiometry(1.0)
product_sink.setConstant(True)

# modifier: Microglia_Hi_Fract is the fraction of time that the antibody is in the high volume compartment Microglia_CL_high_AB{ab_type}
modifier_cell_count = reaction_ab{ab_type}_isf_degradation.createModifier()
modifier_cell_count.setSpecies("Microglia_cell_count")

modifier_hi_fract = reaction_ab{ab_type}_isf_degradation.createModifier()
modifier_hi_fract.setSpecies("Microglia_Hi_Fract")

# Microglia_Hi_Fract is the fraction of time that the antibody is in the high volume compartment Microglia_CL_high_AB{ab_type}
# 1 - Microglia_Hi_Fract is the fraction of time that the antibody is in the low volume compartment Microglia_CL_low_AB{ab_type}

klaw_ab{ab_type}_isf_degradation = reaction_ab{ab_type}_isf_degradation.createKineticLaw()
math_ast = libsbml.parseL3Formula("AB{ab_type}_monomer_antibody_bound * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type})")
klaw_ab{ab_type}_isf_degradation.setMath(math_ast)
            ''',
            # Reaction block 60
            f'''
# 2. Flow of AB{ab_type}Mb from CSF SAS to ISF
reaction_ab{ab_type}mb_sas_to_isf = model.createReaction()
reaction_ab{ab_type}mb_sas_to_isf.setId("ab{ab_type}mb_sas_to_isf")
reaction_ab{ab_type}mb_sas_to_isf.setReversible(False)

# Reactant: AB{ab_type}Mb_SAS
reactant_ab{ab_type}mb_sas_isf = reaction_ab{ab_type}mb_sas_to_isf.createReactant()
reactant_ab{ab_type}mb_sas_isf.setSpecies("AB{ab_type}Mb_SAS")
reactant_ab{ab_type}mb_sas_isf.setStoichiometry(1.0)
reactant_ab{ab_type}mb_sas_isf.setConstant(True)

# Product: AB{ab_type}_monomer_antibody_bound
product_ab{ab_type}_isf_sas = reaction_ab{ab_type}mb_sas_to_isf.createProduct()
product_ab{ab_type}_isf_sas.setSpecies("AB{ab_type}_monomer_antibody_bound")
product_ab{ab_type}_isf_sas.setStoichiometry(1.0)
product_ab{ab_type}_isf_sas.setConstant(True)

# Kinetic law: Q_ISF_brain * AB{ab_type}Mb_SAS
klaw_ab{ab_type}mb_sas_to_isf = reaction_ab{ab_type}mb_sas_to_isf.createKineticLaw()
math_ast = libsbml.parseL3Formula("Q_ISF_brain * AB{ab_type}Mb_SAS")
klaw_ab{ab_type}mb_sas_to_isf.setMath(math_ast)
            ''',
            # Reaction block 61
            f'''
# 3. Flow of AB{ab_type}_monomer_antibody_bound from ISF to CSF LV
reaction_ab{ab_type}_isf_to_lv = model.createReaction()
reaction_ab{ab_type}_isf_to_lv.setId("ab{ab_type}_isf_to_lv")
reaction_ab{ab_type}_isf_to_lv.setReversible(False)

# Reactant: AB{ab_type}_monomer_antibody_bound
reactant_ab{ab_type}_isf_lv = reaction_ab{ab_type}_isf_to_lv.createReactant()
reactant_ab{ab_type}_isf_lv.setSpecies("AB{ab_type}_monomer_antibody_bound")
reactant_ab{ab_type}_isf_lv.setStoichiometry(1.0)
reactant_ab{ab_type}_isf_lv.setConstant(True)

# Product: AB{ab_type}Mb_LV
product_ab{ab_type}mb_lv_isf = reaction_ab{ab_type}_isf_to_lv.createProduct()
product_ab{ab_type}mb_lv_isf.setSpecies("AB{ab_type}Mb_LV")
product_ab{ab_type}mb_lv_isf.setStoichiometry(1.0)
product_ab{ab_type}mb_lv_isf.setConstant(True)

# Kinetic law: f_LV * Q_ISF_brain * AB{ab_type}_monomer_antibody_bound
klaw_ab{ab_type}_isf_to_lv = reaction_ab{ab_type}_isf_to_lv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * Q_ISF_brain * AB{ab_type}_monomer_antibody_bound")
klaw_ab{ab_type}_isf_to_lv.setMath(math_ast)
            ''',
            # Reaction block 62
            f'''
# 4. Flow of AB{ab_type}_monomer_antibody_bound from ISF to CSF TFV
reaction_ab{ab_type}_isf_to_tfv = model.createReaction()
reaction_ab{ab_type}_isf_to_tfv.setId("ab{ab_type}_isf_to_tfv")
reaction_ab{ab_type}_isf_to_tfv.setReversible(False)

# Reactant: AB{ab_type}_monomer_antibody_bound
reactant_ab{ab_type}_isf_tfv = reaction_ab{ab_type}_isf_to_tfv.createReactant()
reactant_ab{ab_type}_isf_tfv.setSpecies("AB{ab_type}_monomer_antibody_bound")
reactant_ab{ab_type}_isf_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}_isf_tfv.setConstant(True)

# Product: AB{ab_type}Mb_TFV
product_ab{ab_type}mb_tfv_isf = reaction_ab{ab_type}_isf_to_tfv.createProduct()
product_ab{ab_type}mb_tfv_isf.setSpecies("AB{ab_type}Mb_TFV")
product_ab{ab_type}mb_tfv_isf.setStoichiometry(1.0)
product_ab{ab_type}mb_tfv_isf.setConstant(True)

# Kinetic law: (1 - f_LV) * Q_ISF_brain * AB{ab_type}_monomer_antibody_bound
klaw_ab{ab_type}_isf_to_tfv = reaction_ab{ab_type}_isf_to_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("(1 - f_LV) * Q_ISF_brain * AB{ab_type}_monomer_antibody_bound")
klaw_ab{ab_type}_isf_to_tfv.setMath(math_ast)
            ''',
            # Reaction block 63
            f'''
# 1. Flow of AB{ab_type}Mb from LV to TFV
reaction_ab{ab_type}mb_lv_to_tfv = model.createReaction()
reaction_ab{ab_type}mb_lv_to_tfv.setId("ab{ab_type}mb_lv_to_tfv")
reaction_ab{ab_type}mb_lv_to_tfv.setReversible(False)

# Reactant: AB{ab_type}Mb_LV
reactant_ab{ab_type}mb_lv_tfv = reaction_ab{ab_type}mb_lv_to_tfv.createReactant()
reactant_ab{ab_type}mb_lv_tfv.setSpecies("AB{ab_type}Mb_LV")
reactant_ab{ab_type}mb_lv_tfv.setStoichiometry(1.0)
reactant_ab{ab_type}mb_lv_tfv.setConstant(True)

# Product: AB{ab_type}Mb_TFV
product_ab{ab_type}mb_tfv_lv = reaction_ab{ab_type}mb_lv_to_tfv.createProduct()
product_ab{ab_type}mb_tfv_lv.setSpecies("AB{ab_type}Mb_TFV")
product_ab{ab_type}mb_tfv_lv.setStoichiometry(1.0)
product_ab{ab_type}mb_tfv_lv.setConstant(True)

# Kinetic law: f_LV * (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mb_LV
klaw_ab{ab_type}mb_lv_to_tfv = reaction_ab{ab_type}mb_lv_to_tfv.createKineticLaw()
math_ast = libsbml.parseL3Formula("f_LV * (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mb_LV")
klaw_ab{ab_type}mb_lv_to_tfv.setMath(math_ast)
            ''',
            # Reaction block 64
            f'''
# 2. Flow of AB{ab_type}Mb from TFV to CM
reaction_ab{ab_type}mb_tfv_to_cm = model.createReaction()
reaction_ab{ab_type}mb_tfv_to_cm.setId("ab{ab_type}mb_tfv_to_cm")
reaction_ab{ab_type}mb_tfv_to_cm.setReversible(False)

# Reactant: AB{ab_type}Mb_TFV
reactant_ab{ab_type}mb_tfv_cm = reaction_ab{ab_type}mb_tfv_to_cm.createReactant()
reactant_ab{ab_type}mb_tfv_cm.setSpecies("AB{ab_type}Mb_TFV")
reactant_ab{ab_type}mb_tfv_cm.setStoichiometry(1.0)
reactant_ab{ab_type}mb_tfv_cm.setConstant(True)

# Product: AB{ab_type}Mb_CM
product_ab{ab_type}mb_cm_tfv = reaction_ab{ab_type}mb_tfv_to_cm.createProduct()
product_ab{ab_type}mb_cm_tfv.setSpecies("AB{ab_type}Mb_CM")
product_ab{ab_type}mb_cm_tfv.setStoichiometry(1.0)
product_ab{ab_type}mb_cm_tfv.setConstant(True)

# Kinetic law: (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mb_TFV
klaw_ab{ab_type}mb_tfv_to_cm = reaction_ab{ab_type}mb_tfv_to_cm.createKineticLaw()
math_ast = libsbml.parseL3Formula("(Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mb_TFV")
klaw_ab{ab_type}mb_tfv_to_cm.setMath(math_ast)
            ''',
            # Reaction block 65
            f'''
# Flow of AB{ab_type}Mb from CM to SAS
reaction_ab{ab_type}mb_cm_to_sas = model.createReaction()
reaction_ab{ab_type}mb_cm_to_sas.setId("ab{ab_type}mb_cm_to_sas")
reaction_ab{ab_type}mb_cm_to_sas.setReversible(False)

# Reactant: AB{ab_type}Mb_CM
reactant_ab{ab_type}mb_cm_sas = reaction_ab{ab_type}mb_cm_to_sas.createReactant()
reactant_ab{ab_type}mb_cm_sas.setSpecies("AB{ab_type}Mb_CM")
reactant_ab{ab_type}mb_cm_sas.setStoichiometry(1.0)
reactant_ab{ab_type}mb_cm_sas.setConstant(True)

# Product: AB{ab_type}Mb_SAS
product_ab{ab_type}mb_sas_cm = reaction_ab{ab_type}mb_cm_to_sas.createProduct()
product_ab{ab_type}mb_sas_cm.setSpecies("AB{ab_type}Mb_SAS")
product_ab{ab_type}mb_sas_cm.setStoichiometry(1.0)
product_ab{ab_type}mb_sas_cm.setConstant(True)

# Kinetic law: (Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mb_CM 
klaw_ab{ab_type}mb_cm_to_sas = reaction_ab{ab_type}mb_cm_to_sas.createKineticLaw()
math_ast = libsbml.parseL3Formula("(Q_CSF_brain + Q_ISF_brain) * AB{ab_type}Mb_CM")
klaw_ab{ab_type}mb_cm_to_sas.setMath(math_ast)
            ''',
        ]
        
        # Execute each reaction block with access to the model variable
        for formatted_block in reaction_blocks:
            # Pass the model, libsbml, and dictionaries in the globals dictionary to make them available in the executed code
            exec(formatted_block, {"model": model, "libsbml": libsbml, "sinks": sinks, "sources": sources})

    # Create species and reactions for AB40
    create_reactions("40")

    # Create species and reactions for AB42
    create_reactions("42")

    return document

def create_parameterized_model_runner(drug_type="gantenerumab"):
    """Create a function to load parameters and create the parameterized model
    
    Args:
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    def model_runner(csv_path):
        # Load parameters using our simplified approach
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        
        # Create and return the model
        return create_parameterized_model(params, params_with_units, drug_type=drug_type)
    
    return model_runner

if __name__ == "__main__":
    # Test the model creation
    doc = create_parameterized_model_runner("gantenerumab")
    print("Model validation complete")