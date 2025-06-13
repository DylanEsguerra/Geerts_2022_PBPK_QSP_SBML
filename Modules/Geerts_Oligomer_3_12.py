"""
Module for modeling small Aβ oligomer formation and dynamics (3-12mers).
This is a QSP component that models the formation and behavior of small Aβ oligomers.

The module includes:
- Oligomer formation from smaller species and monomers
- Oligomer dissociation to smaller species
- Plaque-catalyzed oligomer formation
- Antibody binding to oligomers
- Microglia-mediated clearance of oligomers
- Transport and reflection coefficients for oligomers

Rate Constants:
This module uses rate constants extrapolated by K_rates_extrapolate.py, which calculates
forward and backward rates for oligomerization based on known experimental values for
small oligomers. The rates are extrapolated using Hill functions to capture the size-dependent
behavior of oligomer formation and dissociation.
"""

# updated to lump oligomer 12 into the model
# deleted formation of next from current and monomer and formation of next from current and monomer plaque catalyzed
# deleted dissociation of next to current and monomer
# deleted 24 to 12 reaction and moved to fibril 19-24 model
import libsbml
from pathlib import Path
import pandas as pd
from K_rates_extrapolate import calculate_k_rates

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} in Oligomer 3-12 module ===")
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
    print("\nAntibody binding parameters for Oligomer 3-12:")
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

def create_oligomer_3_12_model(params, params_with_units):
    """Create a parameterized SBML model for the Geerts model
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
        
    Returns:
        SBML document
    """
    print("\nCreating Geerts oligomer 3-12 model...")
    print("Number of parameters loaded:", len(params))
    
    # Print some parameter values to verify loading
    for param_name in ["AB40_Monomer_0", "AB42_Monomer_0", "VIS_brain"]:
        if param_name in params:
            print(f"  Parameter {param_name} = {params[param_name]}")
        else:
            print(f"  Parameter {param_name} not found in CSV")
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_Oligomer_3_12_Model")
    
    # Add a global dictionary to track used reaction IDs across all AB types
    created_reaction_ids = {}
    
    # Create dictionaries to hold sink and source species
    sinks = {}
    sources = {}
    
    # Helper function to ensure unique reaction IDs and track all reactions
    def get_unique_reaction_id(base_id):
        # Simple tracker for total count
        created_reaction_ids[base_id] = created_reaction_ids.get(base_id, 0) + 1
        
        # Return the base_id the first time, otherwise add a suffix
        count = created_reaction_ids[base_id]
        if count == 1:
            return base_id
        else:
            return f"{base_id}_{count-1}"
    
    # Add get_unique_reaction_id to global variables used in exec
    global_vars = globals().copy()
    global_vars.update({
        'model': model, 
        'libsbml': libsbml, 
        'get_unique_reaction_id': get_unique_reaction_id,
        'created_reaction_ids': created_reaction_ids,
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

    # Create common compartments
    compartments = {
        "comp_ISF_brain": params["VIS_brain"],
        "comp_microglia": params["V_microglia"],  # Add microglia compartment
    }

    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setSpatialDimensions(3)
        comp.setUnits('litre')

    # Pre-create all sinks and sources needed for both AB types
    for ab_type in ["40", "42"]:
        # Create sinks for monomer and oligomers
        sinks[f"AB{ab_type}_Monomer"] = create_sink_for_species(f"AB{ab_type}_Monomer", model)
        
        # Create sinks for oligomers 2-13
        for j in range(2, 14):
            oligomer_id = f"AB{ab_type}_Oligomer{j:02d}"
            sinks[oligomer_id] = create_sink_for_species(oligomer_id, model)
            
            # Also create for antibody-bound versions
            oligomer_bound_id = f"{oligomer_id}_Antibody_bound"
            sinks[oligomer_bound_id] = create_sink_for_species(oligomer_bound_id, model)
    
    # No need for the generic Sink species anymore
    
    # Create oligomer-specific parameters
    oligomer_params = []
    
    # Generate forward and backward rate parameters for oligomers 3-12
    for i in range(3, 12):
        # Forward rates
        oligomer_params.extend([
            (f"k_O{i}_O{i+1}_AB40", params[f"k_O{i}_O{i+1}_forty"]),  # AB40 forward
            (f"k_O{i}_O{i+1}_AB42", params[f"k_O{i}_O{i+1}_fortytwo"]),  # AB42 forward
        ])
        # Backward rates
        oligomer_params.extend([
            (f"k_O{i+1}_O{i}_AB40", params[f"k_O{i+1}_O{i}_forty"]),  # AB40 backward
            (f"k_O{i+1}_O{i}_AB42", params[f"k_O{i+1}_O{i}_fortytwo"]),  # AB42 backward
        ])
    
    # Add other non-oligomerization parameters
    oligomer_params.extend([
        # Plaque-induced oligomerization parameters
        ("AB40_Plaque_Vmax", params["AB40_Plaque_Driven_Monomer_Addition_Vmax"]),
        ("AB42_Plaque_Vmax", params["AB42_Plaque_Driven_Monomer_Addition_Vmax"]),
        ("AB40_Plaque_EC50", params["AB40_Plaque_Driven_Monomer_Addition_EC50"]),
        ("AB42_Plaque_EC50", params["AB42_Plaque_Driven_Monomer_Addition_EC50"]),
        # Antibody binding parameters
        ("fta1", params["fta1"]),
        # Transport and clearance parameters
        ("VIS_brain", params["VIS_brain"]),
        # Microglia-related parameters
        ("Microglia_CL_high_AB40", params["Microglia_CL_high_AB40"]),
        ("Microglia_CL_low_AB40", params["Microglia_CL_low_AB40"]),
        ("Microglia_CL_high_AB42", params["Microglia_CL_high_AB42"]),
        ("Microglia_CL_low_AB42", params["Microglia_CL_low_AB42"]),
        ("Microglia_Vmax_forty", params["Microglia_Vmax_forty"]),
        ("Microglia_Vmax_fortytwo", params["Microglia_Vmax_fortytwo"]),
    ])

    # Add reflection coefficients for oligomers 3-12
    for ab_type in ["40", "42"]:
        for j in range(3, 13):
            param_name = f"sigma_ISF_ABeta{ab_type}_oligomer{j:02d}"
            oligomer_params.append((param_name, params[param_name]))

    # Create all parameters in the model
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

    # Create comprehensive set of parameters for all oligomer sizes and AB types
    def create_oligomer_kinetic_parameters():
        """Create all required kinetic parameters for oligomer dynamics across sizes 3-12"""
        print("\nCreating comprehensive set of oligomer kinetic parameters...")
        
        # Define parameter patterns with default values
        for ab_type in ["40", "42"]:  # AB40 and AB42
            for j in range(3, 13):  # Oligomer sizes 3 to 12
                sigma_param = f"sigma_ISF_ABeta{ab_type}_oligomer{j:02d}"
                if model.getParameter(sigma_param) is None:
                    param = model.createParameter()
                    param.setId(sigma_param)
                    param.setValue(params[sigma_param])
                    param.setUnits("dimensionless")
                    param.setConstant(True)
                    print(f"Created parameter: {sigma_param}")
    
    # Call the function to create all parameters
    create_oligomer_kinetic_parameters()

    # Make sure common parameters for all oligomer reactions are defined
    # Define plaque-related parameters for each AB type
    for ab_type in ["40", "42"]:
        # Plaque-driven parameters
        plaque_vmax_param = f"AB{ab_type}_Plaque_Driven_Monomer_Addition_Vmax"
        if model.getParameter(plaque_vmax_param) is None:
            param = model.createParameter()
            param.setId(plaque_vmax_param)
            param.setValue(params[plaque_vmax_param])
            param.setUnits("dimensionless")
            param.setConstant(True)
        
        plaque_ec50_param = f"AB{ab_type}_Plaque_Driven_Monomer_Addition_EC50"
        if model.getParameter(plaque_ec50_param) is None:
            param = model.createParameter()
            param.setId(plaque_ec50_param)
            param.setValue(params[plaque_ec50_param])
            param.setUnits("nanomole_per_litre")
            param.setConstant(True)

    # Create additional oligomer species for sizes 3-11
    for ab_type in ["40", "42"]:
        # Loop through oligomer sizes 3-12
        for j in range(3, 13):
            oligomer_id = f"AB{ab_type}_Oligomer{j:02d}"
            oligomer_bound_id = f"AB{ab_type}_Oligomer{j:02d}_Antibody_bound"
            
            # Create species if it doesn't exist
            if model.getSpecies(oligomer_id) is None:
                species = model.createSpecies()
                species.setId(oligomer_id)
                species.setCompartment("comp_ISF_brain")
                species.setInitialConcentration(0.0)
                species.setSubstanceUnits("nanomole_per_litre")
                species.setHasOnlySubstanceUnits(False)
                species.setBoundaryCondition(False)
                species.setConstant(False)
                print(f"Created species: {oligomer_id}")

             # Create species if it doesn't exist
            if model.getSpecies(oligomer_bound_id) is None:
                species = model.createSpecies()
                species.setId(oligomer_bound_id)
                species.setCompartment("comp_ISF_brain")
                species.setInitialConcentration(0.0)
                species.setSubstanceUnits("nanomole_per_litre")
                species.setHasOnlySubstanceUnits(False)
                species.setBoundaryCondition(False)
                species.setConstant(False)
                species.setBoundaryCondition(False)
                species.setConstant(False)
                print(f"Created species: {oligomer_bound_id}")
            
            # Create Oligomer13 for reactions with Oligomer12
            if j == 12:
                oligomer13_id = f"AB{ab_type}_Oligomer13"
                oligomer13_bound_id = f"AB{ab_type}_Oligomer13_Antibody_bound"
                if model.getSpecies(oligomer13_id) is None:
                    species = model.createSpecies()
                    species.setId(oligomer13_id)
                    species.setCompartment("comp_ISF_brain")
                    species.setInitialConcentration(0.0)
                    species.setSubstanceUnits("nanomole_per_litre")
                    species.setHasOnlySubstanceUnits(False)
                    species.setBoundaryCondition(False)
                    species.setConstant(False)
                    print(f"Created species: {oligomer13_id}")

                if model.getSpecies(oligomer13_bound_id) is None:
                    species = model.createSpecies()
                    species.setId(oligomer13_bound_id)
                    species.setCompartment("comp_ISF_brain")
                    species.setInitialConcentration(0.0)
                    species.setSubstanceUnits("nanomole_per_litre")
                    species.setHasOnlySubstanceUnits(False)
                    species.setBoundaryCondition(False)
                    species.setConstant(False)
                    print(f"Created species: {oligomer13_bound_id}")
        
        # Create plaque species if needed
        plaque_id = f"AB{ab_type}_Plaque_unbound"
        if model.getSpecies(plaque_id) is None:
            species = model.createSpecies()
            species.setId(plaque_id)
            species.setCompartment("comp_ISF_brain")
            species.setInitialConcentration(0.0)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {plaque_id}")
        
    # Add code to create fibril species
    # Create fibril species for each AB type
    for ab_type in ["40", "42"]:
        fibril_id = f"AB{ab_type}_Fibril24"
        
        # Create fibril species if it doesn't exist
        if model.getSpecies(fibril_id) is None:
            species = model.createSpecies()
            species.setId(fibril_id)
            species.setCompartment("comp_ISF_brain")
            species.setInitialConcentration(0.0)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {fibril_id}")

    # For each AB type (40 and 42), create all needed species first
    print("\nCreating all required species...")
    for ab_type in ["40", "42"]:
        # Create monomer species
        monomer_id = f"AB{ab_type}_Monomer"
        
        # Create monomer species if it doesn't exist
        if model.getSpecies(monomer_id) is None:
            species = model.createSpecies()
            species.setId(monomer_id)
            species.setCompartment("comp_ISF_brain")
            species.setInitialConcentration(0.0)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {monomer_id}")
        
        # Create oligomer 2 (needed for reactions with oligomer 3)
        oligomer2_id = f"AB{ab_type}_Oligomer02"
        
        # Create oligomer 2 species
        if model.getSpecies(oligomer2_id) is None:
            species = model.createSpecies()
            species.setId(oligomer2_id)
            species.setCompartment("comp_ISF_brain")
            species.setInitialConcentration(0.0)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {oligomer2_id}")

    # At the end of the species creation section, add code to verify all species exist
    print("All species created successfully.")

    # Add antibody-related species that might be missing
    antibody_species = [
        ("Ab_t", "comp_ISF_brain", 0.0),
    ]
    
    # Create antibody species if they don't exist
    for species_id, compartment_id, initial_value in antibody_species:
        if model.getSpecies(species_id) is None:
            species = model.createSpecies()
            species.setId(species_id)
            species.setCompartment(compartment_id)
            species.setInitialConcentration(initial_value)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {species_id}")
    
    # Create microglia species
    microglia_species = [
        ("Microglia_Hi_Fract", "comp_microglia", params["Microglia_Hi_Fract_0"]),
        ("Microglia_cell_count", "comp_microglia", params["Microglia_cell_count_0"]),
    ]

    # Create microglia species if they don't exist
    for species_id, compartment_id, initial_value in microglia_species:
        if model.getSpecies(species_id) is None:
            species = model.createSpecies()
            species.setId(species_id)
            species.setCompartment(compartment_id)
            species.setInitialConcentration(initial_value)
            species.setSubstanceUnits("nanomole_per_litre")
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setConstant(False)
            print(f"Created species: {species_id}")

    # Add a function to verify species exist, add this before create_oligomer_3_12_reactions()
    def check_species_exists(species_id):
        """Verify the species exists, create a helpful error message if not"""
        if model.getSpecies(species_id) is None:
            print(f"ERROR: Required species {species_id} does not exist. Add it to the model before creating reactions.")
            return False
        return True

    # At the beginning of create_oligomer_3_12_reactions, add code to verify all needed species exist
    def create_oligomer_3_12_reactions(ab_type):
        """Create all species and reactions for oligomerization of a specific AB type (40 or 42)."""
        
        # Verify required species exist
        monomer_id = f"AB{ab_type}_Monomer"
        if not check_species_exists(monomer_id):
            print(f"Cannot create reactions for AB{ab_type} without monomer species.")
            return
        
        # Verify oligomer 2 exists (needed for oligomer 3 reactions)
        oligomer2_id = f"AB{ab_type}_Oligomer02" 
        if not check_species_exists(oligomer2_id):
            print(f"Cannot create reactions for AB{ab_type} without oligomer 2 species.")
            return
        
        # Verify all oligomer species from 3-12 exist
        for j in range(3, 13):
            oligomer_id = f"AB{ab_type}_Oligomer{j:02d}"
            if not check_species_exists(oligomer_id):
                print(f"Cannot create reactions for {oligomer_id}.")
                return
        
        # Verify plaque species exists
        plaque_id = f"AB{ab_type}_Plaque_unbound"
        if not check_species_exists(plaque_id):
            print(f"Cannot create reactions for AB{ab_type} without plaque species.")
            return
        
        # Different parameter name suffix based on AB type
        suffix = "forty" if ab_type == "40" else "fortytwo"
        
        # Function to ensure sigma parameter exists for a given oligomer size
        def ensure_sigma_parameter(j):
            """Make sure the reflection coefficient parameter exists for this oligomer size"""
            sigma_param_id = f"sigma_ISF_ABeta{ab_type}_oligomer{j:02d}"
            if model.getParameter(sigma_param_id) is None:
                print(f"Creating missing reflection coefficient: {sigma_param_id}")
                param = model.createParameter()
                param.setId(sigma_param_id)
                param.setValue(0.9)  # Default reflection coefficient
                param.setUnits("dimensionless")
                param.setConstant(True)
            return sigma_param_id
        
       # Function to ensure rate constant parameter exists
        def ensure_rate_parameter(param_id, default_value=0.1):
            """Make sure the rate constant parameter exists and has correct value"""
            if model.getParameter(param_id) is None:
                print(f"Creating rate constant: {param_id}")
                param = model.createParameter()
                param.setId(param_id)
                
                # First try direct lookup with current name (now that CSV has these names)
                if param_id in params:
                    value = params[param_id]
                    print(f"  Using direct value from CSV or extrapolated rates: {value}")
                else:
                    value = default_value
                    print(f"  WARNING: No value found in params, using default: {value}")
                
                param.setValue(value)
                param.setUnits("per_hour")
                param.setConstant(True)
            return param_id
        
        print(f'Creating oligomer 3-12 reactions for AB{ab_type}...')
        
        # First create the common reactions for all oligomers 3-12
        for j in range(3, 13):
            current_oligomer = f"AB{ab_type}_Oligomer{j:02d}"
            prev_oligomer = f"AB{ab_type}_Oligomer{j-1:02d}"
            next_oligomer = f"AB{ab_type}_Oligomer{j+1:02d}" if j < 12 else f"AB{ab_type}_Oligomer13"
            
            # Define oligomer-specific parameter names based on j
            k_prev_current = ensure_rate_parameter(f"k_O{j-1}_O{j}_{suffix}", 0.0)   # e.g. k_O2_O3_forty for j=3
            k_current_prev = ensure_rate_parameter(f"k_O{j}_O{j-1}_{suffix}", 0.0)   # e.g. k_O3_O2_forty for j=3
            k_current_next = ensure_rate_parameter(f"k_O{j}_O{j+1}_{suffix}", 0.0)   # e.g. k_O3_O4_forty for j=3
            k_next_current = ensure_rate_parameter(f"k_O{j+1}_O{j}_{suffix}", 0.0)   # e.g. k_O4_O3_forty for j=3
            sigma_param = ensure_sigma_parameter(j)
            
            oligomer_blocks = [
                # Reaction 1: Formation of current oligomer from previous oligomer and monomer
                f'''
# Formation of {current_oligomer} from {prev_oligomer} and monomer
reaction_formation = model.createReaction()
base_id = f"{current_oligomer}_formation"
reaction_formation.setId(get_unique_reaction_id(base_id))
reaction_formation.setReversible(False)

# Reactants: {prev_oligomer} and monomer
reactant_oligomer = reaction_formation.createReactant()
reactant_oligomer.setSpecies("{prev_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

reactant_monomer = reaction_formation.createReactant()
reactant_monomer.setSpecies("AB{ab_type}_Monomer")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Product: {current_oligomer}
product_oligomer = reaction_formation.createProduct()
product_oligomer.setSpecies("{current_oligomer}")
product_oligomer.setStoichiometry(1.0)
product_oligomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_oligomer_formation = reaction_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"{k_prev_current} * {prev_oligomer} * AB{ab_type}_Monomer * VIS_brain")
klaw_oligomer_formation.setMath(math_ast)
                ''',
                
                # Reaction 2: Dissociation of current oligomer to previous oligomer and monomer
                f'''
# Dissociation of {current_oligomer} to {prev_oligomer} and monomer
reaction_dissociation = model.createReaction()
base_id = f"{current_oligomer}_dissociation"
reaction_dissociation.setId(get_unique_reaction_id(base_id))
reaction_dissociation.setReversible(False)

# Reactant: {current_oligomer}
reactant_oligomer = reaction_dissociation.createReactant()
reactant_oligomer.setSpecies("{current_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

# Products: {prev_oligomer} and monomer
product_smaller_oligomer = reaction_dissociation.createProduct()
product_smaller_oligomer.setSpecies("{prev_oligomer}")
product_smaller_oligomer.setStoichiometry(1.0)
product_smaller_oligomer.setConstant(True)

product_monomer = reaction_dissociation.createProduct()
product_monomer.setSpecies("AB{ab_type}_Monomer")
product_monomer.setStoichiometry(1.0)
product_monomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_oligomer_dissociation = reaction_dissociation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"{k_current_prev} * {current_oligomer} * VIS_brain")
klaw_oligomer_dissociation.setMath(math_ast)
                ''',
                
                # Reaction 4: Plaque-catalyzed formation of current oligomer
                f'''
# Plaque-catalyzed formation of {current_oligomer} from {prev_oligomer} and monomer
reaction_plaque_formation = model.createReaction()
base_id = f"{current_oligomer}_plaque_catalyzed_formation"
reaction_plaque_formation.setId(get_unique_reaction_id(base_id))
reaction_plaque_formation.setReversible(False)

# Reactants: {prev_oligomer} and monomer
reactant_oligomer = reaction_plaque_formation.createReactant()
reactant_oligomer.setSpecies("{prev_oligomer}")
reactant_oligomer.setStoichiometry(1.0)
reactant_oligomer.setConstant(True)

reactant_monomer = reaction_plaque_formation.createReactant()
reactant_monomer.setSpecies("AB{ab_type}_Monomer")
reactant_monomer.setStoichiometry(1.0)
reactant_monomer.setConstant(True)

# Modifier: AB{ab_type}_Plaque_unbound (catalyst, not consumed)
modifier_plaque = reaction_plaque_formation.createModifier()
modifier_plaque.setSpecies("AB{ab_type}_Plaque_unbound")

# Product: {current_oligomer}
product_oligomer = reaction_plaque_formation.createProduct()
product_oligomer.setSpecies("{current_oligomer}")
product_oligomer.setStoichiometry(1.0)
product_oligomer.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_plaque_oligomer_formation = reaction_plaque_formation.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"{k_prev_current} * AB{ab_type}_Plaque_Driven_Monomer_Addition_Vmax * {prev_oligomer} * AB{ab_type}_Monomer * (AB{ab_type}_Plaque_unbound / (AB{ab_type}_Plaque_unbound + AB{ab_type}_Plaque_Driven_Monomer_Addition_EC50)) * VIS_brain")
klaw_plaque_oligomer_formation.setMath(math_ast)
                ''',
                
                # Reaction 6: Antibody binding to current oligomer
                f'''
# Antibody binding to {current_oligomer}
reaction_antibody_binding = model.createReaction()
base_id = f"{current_oligomer}_antibody_binding"
reaction_antibody_binding.setId(get_unique_reaction_id(base_id))
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
product_bound.setSpecies(f"AB{ab_type}_Oligomer{j:02d}_Antibody_bound")
product_bound.setStoichiometry(1.0)
product_bound.setConstant(True)

# Kinetic law: Update to include VIS_brain in formula
klaw_antibody_oligomer_binding = reaction_antibody_binding.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"fta1 * {current_oligomer} * Ab_t * VIS_brain")
klaw_antibody_oligomer_binding.setMath(math_ast)
                ''',
                
                # NEW Reaction 8: Microglia clearance of current oligomer
                f'''
# Microglia clearance of {current_oligomer}
reaction_microglia_clearance = model.createReaction()
base_id = f"{current_oligomer}_microglia_clearance"
reaction_microglia_clearance.setId(get_unique_reaction_id(base_id))
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
product.setSpecies(sinks[f"{current_oligomer}"])
product.setConstant(True)
product.setStoichiometry(1.0)

# Kinetic law: Update to include VIS_brain in formula
klaw_microglia_clearance = reaction_microglia_clearance.createKineticLaw()
math_ast = libsbml.parseL3Formula(f"Microglia_Vmax_{suffix} * {current_oligomer} * Microglia_cell_count * (Microglia_Hi_Fract * Microglia_CL_high_AB{ab_type} + (1 - Microglia_Hi_Fract) * Microglia_CL_low_AB{ab_type}) * VIS_brain")
klaw_microglia_clearance.setMath(math_ast)
                '''
            ]
            
            # Execute each reaction block for this oligomer size
            for block in oligomer_blocks:
                exec(block, global_vars)

      
    
    # Create species and reactions for AB40
    create_oligomer_3_12_reactions("40")

    # Create species and reactions for AB42
    create_oligomer_3_12_reactions("42")

    # Log the number of reactions created
    print(f"Created {len(created_reaction_ids)} unique reactions in the oligomer 3-12 model")

    
    return document

def create_oligomer_3_12_model_runner():
    """Create a function that loads parameters and creates a parameterized model
        
    Returns:
        Function that takes a parameter CSV path and returns an SBML document
    """
    def model_runner(csv_path, drug_type="gantenerumab"):
        # Load parameters with drug-specific parameter mapping
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        
        # Create model - now using properly mapped parameters
        document = create_oligomer_3_12_model(params, params_with_units)
        
        return document
    
    return model_runner

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
    sink.setInitialConcentration(0.0)  # Using 0 instead of 1.0 for sinks in oligomer model
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
    source.setInitialConcentration(0.0)  # Using 0 instead of 1.0 for sources in oligomer model
    source.setConstant(True)
    source.setHasOnlySubstanceUnits(False)
    source.setBoundaryCondition(True)  # Boundary species
    source.setSubstanceUnits("nanomole_per_litre")
    return source_id

if __name__ == "__main__":
    # Test the model creation
    runner = create_oligomer_3_12_model_runner()
    doc = runner("parameters/PK_Geerts.csv")
    print("Model validation complete") 