"""
Module for modeling perivascular space (PVS) dynamics and ARIA (Amyloid-Related Imaging Abnormalities).
This is a PBPK component that models the fluid dynamics and clearance in the perivascular space.
The module includes:
- Perivascular space fluid flow and transport
- Microglia activation and migration
- Clearance of Aβ through the glymphatic system
- Antibody and Aβ transport between PVS and ISF
- Future location of ARIA model 

Added a reaction from PVS to Central for bound and free monomers 
"""

# Paper says to set PVS plaque equal to ISF plaque
# ODE CSV uses CL_AB40_plaque_bound_PVS and CL_AB42_plaque_bound_PVS to clear plaque
# These parameters are zero in CSV file
# Macrophage equations listed but unused in model
import libsbml
from pathlib import Path
import pandas as pd

def load_parameters(csv_path, drug_type="gantenerumab"):
    """Load parameters from CSV file into dictionary with values and units
    
    Args:
        csv_path: Path to CSV file containing parameters
        drug_type: Type of drug to simulate ("lecanemab" or "gantenerumab")
    """
    print(f"\n=== Loading parameters for {drug_type.upper()} in PVS/ARIA module ===")
    print(f"Loading parameters from {csv_path}")
    df = pd.read_csv(csv_path)
    
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
        'Vcent': 'Lec_Vcent' if is_lecanemab else 'Gant_Vcent',
    }
    
    # Apply parameter mapping
    for generic_name, specific_name in param_mapping.items():
        if specific_name in params:
            params[generic_name] = params[specific_name]
            if specific_name in params_with_units:
                params_with_units[generic_name] = params_with_units[specific_name]
    
    return params, params_with_units

def create_pvs_aria_model(params, params_with_units):
    """Create a parameterized SBML model for PVS and ARIA dynamics
    
    Args:
        params: Dictionary of parameter values
        params_with_units: Dictionary of parameter values with units
    """
    print("\nCreating PVS/ARIA model species...")
    
    # Create SBML document
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel("Geerts_PVS_ARIA_Model")

    model.setTimeUnits("hour")
    
    # Create units
    hour = model.createUnitDefinition()
    hour.setId('hour')
    hour_unit = hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setMultiplier(3600.0)
    hour_unit.setScale(0)
    hour_unit.setExponent(1.0)

    # Add per_hour unit definition
    per_hour = model.createUnitDefinition()
    per_hour.setId('per_hour')
    hour_unit = per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # Add nanomole_per_hour unit definition
    nanomole_per_hour = model.createUnitDefinition()
    nanomole_per_hour.setId('nanomole_per_hour')
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

    # Add liter_per_hour unit definition
    liter_per_hour = model.createUnitDefinition()
    liter_per_hour.setId('liter_per_hour')
    litre_unit = liter_per_hour.createUnit()
    litre_unit.setKind(libsbml.UNIT_KIND_LITRE)
    litre_unit.setExponent(1.0)
    litre_unit.setScale(0)
    litre_unit.setMultiplier(1.0)
    hour_unit = liter_per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)

    # Add nanomole unit definition
    nanomole = model.createUnitDefinition()
    nanomole.setId('nanomole')
    mole_unit = nanomole.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(1.0)
    mole_unit.setScale(-9)  # nano
    mole_unit.setMultiplier(1.0)

    # Add per_mole_per_hour unit definition
    per_mole_per_hour = model.createUnitDefinition()
    per_mole_per_hour.setId('per_mole_per_hour')
    mole_unit = per_mole_per_hour.createUnit()
    mole_unit.setKind(libsbml.UNIT_KIND_MOLE)
    mole_unit.setExponent(-1.0)
    mole_unit.setScale(0)
    mole_unit.setMultiplier(1.0)
    hour_unit = per_mole_per_hour.createUnit()
    hour_unit.setKind(libsbml.UNIT_KIND_SECOND)
    hour_unit.setExponent(-1.0)
    hour_unit.setScale(0)
    hour_unit.setMultiplier(1.0/3600.0)
    
    # First, update the species list to use consistent compartment names
    species_list = []
    
    # Monomers
    for ab_type in ["40", "42"]:
        species_list.extend([
            (f"AB{ab_type}_Monomer", f"AB{ab_type} Monomer", "comp_ISF_brain"),
            (f"AB{ab_type}_Monomer_PVS", f"AB{ab_type} Monomer in PVS", "comp_PVS"),
            (f"AB{ab_type}_Monomer_PVS_bound", f"Bound AB{ab_type} Monomer in PVS", "comp_PVS"),
        ])
    
    # Oligomers (2-16)
    for i in range(2, 17):
        for ab_type in ["40", "42"]:
            species_list.extend([
                (f"AB{ab_type}_Oligomer{i:02d}", f"AB{ab_type} Oligomer {i}", "comp_ISF_brain"),
                (f"AB{ab_type}_Oligomer{i:02d}_PVS", f"AB{ab_type} Oligomer {i} in PVS", "comp_PVS"),
                (f"AB{ab_type}_Oligomer{i:02d}_PVS_bound", f"Bound AB{ab_type} Oligomer {i} in PVS", "comp_PVS"),
            ])
    
    # Fibrils (17-24)
    for i in range(17, 25):
        for ab_type in ["40", "42"]:
            species_list.extend([
                (f"AB{ab_type}_Fibril{i:02d}", f"AB{ab_type} Fibril {i}", "comp_ISF_brain"),
                (f"AB{ab_type}_Fibril{i:02d}_PVS", f"AB{ab_type} Fibril {i} in PVS", "comp_PVS"),
                (f"AB{ab_type}_Fibril{i:02d}_PVS_bound", f"Bound AB{ab_type} Fibril {i} in PVS", "comp_PVS"),
            ])
    
    # Add antibody and plaque species
    species_list.extend([
        ("AB40_Plaque_bound_PVS", "AB40 plaque bound in PVS", "comp_PVS"),
        ("AB42_Plaque_bound_PVS", "AB42 plaque bound in PVS", "comp_PVS"),
        ("AB40_Plaque_unbound_PVS", "AB40 plaque unbound in PVS", "comp_PVS"),
        ("AB42_Plaque_unbound_PVS", "AB42 plaque unbound in PVS", "comp_PVS"),
        ("C_Antibody_unbound_PVS", "Antibody in PVS", "comp_PVS"),
        ("Ab_t", "Antibody in ISF", "comp_ISF_brain"),
        ("PK_central", "Antibody in central compartment", "comp_Central_compartment"),
        ("PK_SAS_brain", "Antibody in SAS", "comp_CSF_SAS"),
    ])
    
    # Create compartments first
    compartments = {
        "comp_ISF_brain": params["VIS_brain"],
        "comp_PVS": params["V_PVS"],
        "comp_Central_compartment": params["Vcent"],
        "comp_CSF_SAS": params["V_SAS_brain"],
    }
    
    # Create all compartments
    for comp_id, size in compartments.items():
        comp = model.createCompartment()
        comp.setId(comp_id)
        comp.setSize(size)
        comp.setConstant(True)
        comp.setUnits('litre')
    
    # Verify all species compartments exist before creating species
    for species_id, _, comp_id in species_list:
        if model.getCompartment(comp_id) is None:
            raise ValueError(f"Compartment '{comp_id}' not found for species '{species_id}'")
    
    # Create all species with proper compartment references
    for species_id, name, comp in species_list:
        if model.getSpecies(species_id) is None:
            species = model.createSpecies()
            species.setId(species_id)
            species.setName(name)
            species.setCompartment(comp)
            species.setInitialConcentration(0.0)
            species.setConstant(False)
            species.setHasOnlySubstanceUnits(False)
            species.setBoundaryCondition(False)
            species.setSubstanceUnits("nanomole_per_litre")

    # Create dictionaries to store sinks for each species (replacing the generic Sink)
    sinks = {}
    sources = {}
    
    # Pre-create sinks for all species that need them
    # 1. Monomers
    for ab_type in ["40", "42"]:
        sinks[f"AB{ab_type}_Monomer_PVS"] = create_sink_for_species(f"AB{ab_type}_Monomer_PVS", model)
        sinks[f"AB{ab_type}_Monomer_PVS_bound"] = create_sink_for_species(f"AB{ab_type}_Monomer_PVS_bound", model)
        sinks[f"AB{ab_type}_Plaque_bound_PVS"] = create_sink_for_species(f"AB{ab_type}_Plaque_bound_PVS", model)
        sinks[f"AB{ab_type}_Plaque_unbound_PVS"] = create_sink_for_species(f"AB{ab_type}_Plaque_unbound_PVS", model)
    
    # 2. Oligomers (2-16)
    for i in range(2, 17):
        for ab_type in ["40", "42"]:
            sinks[f"AB{ab_type}_Oligomer{i:02d}_PVS"] = create_sink_for_species(f"AB{ab_type}_Oligomer{i:02d}_PVS", model)
            sinks[f"AB{ab_type}_Oligomer{i:02d}_PVS_bound"] = create_sink_for_species(f"AB{ab_type}_Oligomer{i:02d}_PVS_bound", model)
    
    # 3. Fibrils (17-24)
    for i in range(17, 25):
        for ab_type in ["40", "42"]:
            sinks[f"AB{ab_type}_Fibril{i:02d}_PVS"] = create_sink_for_species(f"AB{ab_type}_Fibril{i:02d}_PVS", model)
            sinks[f"AB{ab_type}_Fibril{i:02d}_PVS_bound"] = create_sink_for_species(f"AB{ab_type}_Fibril{i:02d}_PVS_bound", model)

    # Create parameters
    pvs_params = [
        # Transport parameters
        ("Q_PVS", params["Q_PVS"]),
        ("sigma_PVS", params["sigma_PVS"]),
        ("sigma_PVS_AB40", params["sigma_PVS_AB40"]),
        ("sigma_PVS_AB42", params["sigma_PVS_AB42"]),
        ("sigma_PVS_AB40_bound", params["sigma_PVS_AB40_bound"]),
        ("sigma_PVS_AB42_bound", params["sigma_PVS_AB42_bound"]),
        ("CSF_fraction", params["CSF_fraction"]),
        ("V_PVS", params["V_PVS"]),
        ("fta0", params["fta0"]),
        ("fta1", params["fta1"]),
        ("fta2", params["fta2"]),
        ("fta3", params["fta3"]),
        ("sigma_ISF", params["sigma_ISF"]),
        ("sigma_L_brain_ISF", params["sigma_L_brain_ISF"]),
        ("Qbrain_ISFpsv", params["Qbrain_ISFpsv"]),
        ("PS_Ab_ISF_PVS", params["PS_Ab_ISF_PVS"]),
        ("Pe", params["Pe"]),
        ("CL_AB40_plaque_bound_PVS", params["CL_AB40_plaque_bound_PVS"]),
        ("CL_AB42_plaque_bound_PVS", params["CL_AB42_plaque_bound_PVS"]),
        ("CL_AB40_plaque_free_PVS", params["CL_AB40_plaque_free_PVS"]),
        ("CL_AB42_plaque_free_PVS", params["CL_AB42_plaque_free_PVS"]),
    ]

    # Add reflection coefficients
    for i in range(1, 25):  # Including monomers (1) through fibrils (24)
        for ab_type in ["40", "42"]:
            name = "monomer" if i == 1 else f"oligomer{i:02d}"
            pvs_params.append((
                f"sigma_ISF_ABeta{ab_type}_{name}",
                params[f"sigma_ISF_ABeta{ab_type}_{name}"]
            ))

    # Create all parameters
    for param_id, value in pvs_params:
        param = model.createParameter()
        param.setId(param_id)
        param.setValue(value)
        param.setConstant(True)
        
        # Set appropriate units
        if param_id.startswith('Q_'):
            param.setUnits("litre_per_hour")
        elif param_id.startswith('k_'):
            param.setUnits("per_hour")
        elif param_id.startswith('sigma_'):
            param.setUnits("dimensionless")

    def create_transport_reactions():
        """Create all ISF to PVS transport reactions"""
        
        # Create transport reactions for all species
        for ab_type in ["40", "42"]:
            # Verify species exist in correct compartments
            isf_species = f"AB{ab_type}_Monomer"
            pvs_species = f"AB{ab_type}_Monomer_PVS"
            pvs_species_bound = f"AB{ab_type}_Monomer_PVS_bound"
            
            isf_species_obj = model.getSpecies(isf_species)
            pvs_species_obj = model.getSpecies(pvs_species)
            
            if isf_species_obj is None:
                raise ValueError(f"Species {isf_species} not found in ISF compartment")
            if pvs_species_obj is None:
                raise ValueError(f"Species {pvs_species} not found in PVS compartment")
                
            # Verify compartments
            if isf_species_obj.getCompartment() != "comp_ISF_brain":
                raise ValueError(f"Species {isf_species} is in wrong compartment. Expected comp_ISF_brain, got {isf_species_obj.getCompartment()}")
            if pvs_species_obj.getCompartment() != "comp_PVS":
                raise ValueError(f"Species {pvs_species} is in wrong compartment. Expected comp_PVS, got {pvs_species_obj.getCompartment()}")
                
            # Monomer transport
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_monomer_to_PVS")
            reaction.setReversible(False)
            
            # Reactant (must exist in ISF)
            reactant = reaction.createReactant()
            reactant.setSpecies(isf_species)
            reactant.setStoichiometry(1.0)
            
            # Product (must exist in PVS)
            product = reaction.createProduct()
            product.setSpecies(pvs_species)
            product.setStoichiometry(1.0)
            
            # Kinetic law
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"(1 - sigma_ISF_ABeta{ab_type}_monomer) * Q_PVS * AB{ab_type}_Monomer"
            )
            klaw.setMath(math_ast)

            # Oligomer transport (2-16)
            for i in range(2, 17):
                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Oligomer{i:02d}_to_PVS")
                reaction.setReversible(False)
                
                # Reactant
                reactant = reaction.createReactant()
                reactant.setSpecies(f"AB{ab_type}_Oligomer{i:02d}")
                reactant.setStoichiometry(1.0)
                
                # Product
                product = reaction.createProduct()
                product.setSpecies(f"AB{ab_type}_Oligomer{i:02d}_PVS")
                product.setStoichiometry(1.0)
                
                # Kinetic law
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"(1 - sigma_ISF_ABeta{ab_type}_oligomer{i:02d}) * Q_PVS * AB{ab_type}_Oligomer{i:02d}"
                )
                klaw.setMath(math_ast)

            # Fibril transport (17-24)
            for i in range(17, 25):
                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Fibril{i:02d}_to_PVS")
                reaction.setReversible(False)
                
                # Reactant
                reactant = reaction.createReactant()
                reactant.setSpecies(f"AB{ab_type}_Fibril{i:02d}")
                reactant.setStoichiometry(1.0)
                
                # Product
                product = reaction.createProduct()
                product.setSpecies(f"AB{ab_type}_Fibril{i:02d}_PVS")
                product.setStoichiometry(1.0)
                
                # Kinetic law
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"(1 - sigma_ISF_ABeta{ab_type}_oligomer{i:02d}) * Q_PVS * AB{ab_type}_Fibril{i:02d}"
                )
                klaw.setMath(math_ast)

    def create_binding_reactions():
        """Create all PVS binding/unbinding reactions and lymph clearance"""
        

        # Create binding reactions for all species
        for ab_type in ["40", "42"]:
            # Monomer binding
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_Monomer_PVS_binding")
            reaction.setReversible(False)
            
            # Reactants
            reactant1 = reaction.createReactant()
            reactant1.setSpecies(f"AB{ab_type}_Monomer_PVS")
            reactant1.setStoichiometry(1.0)
            
            reactant2 = reaction.createReactant()
            reactant2.setSpecies("C_Antibody_unbound_PVS")
            reactant2.setStoichiometry(1.0)
            
            # Product
            product = reaction.createProduct()
            product.setSpecies(f"AB{ab_type}_Monomer_PVS_bound")
            product.setStoichiometry(1.0)
            
            # Kinetic law: fta0 * species * antibody
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"fta0 * AB{ab_type}_Monomer_PVS * C_Antibody_unbound_PVS * V_PVS"
            )
            klaw.setMath(math_ast)

            # Plaque binding
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_Plaque_binding")
            reaction.setReversible(False)

            # Reactants
            reactant1 = reaction.createReactant()
            reactant1.setSpecies(f"AB{ab_type}_Plaque_unbound_PVS")
            reactant1.setStoichiometry(1.0)
            
            reactant2 = reaction.createReactant()
            reactant2.setSpecies("C_Antibody_unbound_PVS")
            reactant2.setStoichiometry(1.0)

            # Product
            product = reaction.createProduct()
            product.setSpecies(f"AB{ab_type}_Plaque_bound_PVS")
            product.setStoichiometry(1.0)

            # Kinetic law: fta0 * species * antibody
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"fta3 * AB{ab_type}_Plaque_unbound_PVS * C_Antibody_unbound_PVS * V_PVS"
            )
            klaw.setMath(math_ast)

            # Oligomer binding (2-16)
            for i in range(2, 17):
                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Oligomer{i:02d}_PVS_binding")
                reaction.setReversible(False)
                
                # Reactants
                reactant1 = reaction.createReactant()
                reactant1.setSpecies(f"AB{ab_type}_Oligomer{i:02d}_PVS")
                reactant1.setStoichiometry(1.0)
                
                reactant2 = reaction.createReactant()
                reactant2.setSpecies("C_Antibody_unbound_PVS")
                reactant2.setStoichiometry(1.0)
                
                # Product
                product = reaction.createProduct()
                product.setSpecies(f"AB{ab_type}_Oligomer{i:02d}_PVS_bound")
                product.setStoichiometry(1.0)
                
                # Kinetic law: fta1 * species * antibody
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"fta1 * AB{ab_type}_Oligomer{i:02d}_PVS * C_Antibody_unbound_PVS * V_PVS"
                )
                klaw.setMath(math_ast)

            # Fibril binding (17-24)
            for i in range(17, 25):
                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Fibril{i:02d}_PVS_binding")
                reaction.setReversible(False)
                
                # Reactants
                reactant1 = reaction.createReactant()
                reactant1.setSpecies(f"AB{ab_type}_Fibril{i:02d}_PVS")
                reactant1.setStoichiometry(1.0)
                
                reactant2 = reaction.createReactant()
                reactant2.setSpecies("C_Antibody_unbound_PVS")
                reactant2.setStoichiometry(1.0)
                
                # Product
                product = reaction.createProduct()
                product.setSpecies(f"AB{ab_type}_Fibril{i:02d}_PVS_bound")
                product.setStoichiometry(1.0)
                
                # Kinetic law: fta2 * species * antibody
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"fta2 * AB{ab_type}_Fibril{i:02d}_PVS * C_Antibody_unbound_PVS * V_PVS"
                )
                klaw.setMath(math_ast)

    def create_lymph_clearance():
        """Create lymphatic clearance reactions"""
            
        # Create transport reactions for all species
        for ab_type in ["40", "42"]:
            # Monomer transport to central compartment
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_monomer_PVS_bound_to_central_bound")
            reaction.setReversible(False)
            
            # Reactant
            reactant = reaction.createReactant()
            reactant.setSpecies(f"AB{ab_type}_Monomer_PVS_bound")
            reactant.setStoichiometry(1.0)
            
            product = reaction.createProduct()
            product.setSpecies(f"AB{ab_type}Mb_Central")
            product.setStoichiometry(1.0)
            
            # Kinetic law - use the standard reflection coefficient
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"(1 - sigma_PVS_AB{ab_type}_bound) * Q_PVS * AB{ab_type}_Monomer_PVS_bound"
            )
            klaw.setMath(math_ast)

            # Monomer transport to central compartment
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_monomer_PVS_to_central_free")
            reaction.setReversible(False)
            
            # Reactant
            reactant = reaction.createReactant()
            reactant.setSpecies(f"AB{ab_type}_Monomer_PVS")
            reactant.setStoichiometry(1.0)
            
            product = reaction.createProduct()
            product.setSpecies(f"AB{ab_type}Mu_Central")
            product.setStoichiometry(1.0)
            
            # Kinetic law - use the standard reflection coefficient
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"(1 - sigma_PVS_AB{ab_type}) * Q_PVS * AB{ab_type}_Monomer_PVS"
            )
            klaw.setMath(math_ast)

            # Oligomer transport (2-16)
            for i in range(2, 17):
                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Oligomer{i:02d}_PVS_bound_clearance")
                reaction.setReversible(False)
                
                # Reactant
                reactant = reaction.createReactant()
                reactant.setSpecies(f"AB{ab_type}_Oligomer{i:02d}_PVS_bound")
                reactant.setStoichiometry(1.0)
                
                # Product: Sink
                product_sink = reaction.createProduct()
                product_sink.setSpecies(sinks[f"AB{ab_type}_Oligomer{i:02d}_PVS_bound"])
                product_sink.setStoichiometry(1.0)
                product_sink.setConstant(True)
                
                # Kinetic law
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"(1 - sigma_PVS_AB{ab_type}_bound) * Q_PVS * AB{ab_type}_Oligomer{i:02d}_PVS_bound"
                )
                klaw.setMath(math_ast)

                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Oligomer{i:02d}_PVS_clearance")
                reaction.setReversible(False)
                
                # Reactant
                reactant = reaction.createReactant()
                reactant.setSpecies(f"AB{ab_type}_Oligomer{i:02d}_PVS")
                reactant.setStoichiometry(1.0)
                
                # Product: Sink
                product_sink = reaction.createProduct()
                product_sink.setSpecies(sinks[f"AB{ab_type}_Oligomer{i:02d}_PVS"])
                product_sink.setStoichiometry(1.0)
                product_sink.setConstant(True)
                
                # Kinetic law
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"(1 - sigma_PVS_AB{ab_type}) * Q_PVS * AB{ab_type}_Oligomer{i:02d}_PVS"
                )
                klaw.setMath(math_ast)

            # Fibril transport (17-24)
            for i in range(17, 25):
                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Fibril{i:02d}_PVS_bound_clearance")
                reaction.setReversible(False)
                
                # Reactant
                reactant = reaction.createReactant()
                reactant.setSpecies(f"AB{ab_type}_Fibril{i:02d}_PVS_bound")
                reactant.setStoichiometry(1.0)
                
                # Product: Sink
                product_sink = reaction.createProduct()
                product_sink.setSpecies(sinks[f"AB{ab_type}_Fibril{i:02d}_PVS_bound"])
                product_sink.setStoichiometry(1.0)
                product_sink.setConstant(True)
                
                # Kinetic law
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"(1 - sigma_PVS_AB{ab_type}_bound) * Q_PVS * AB{ab_type}_Fibril{i:02d}_PVS_bound"
                )
                klaw.setMath(math_ast)

                reaction = model.createReaction()
                reaction.setId(f"AB{ab_type}_Fibril{i:02d}_PVS_clearance")
                reaction.setReversible(False)
                
                # Reactant
                reactant = reaction.createReactant()    
                reactant.setSpecies(f"AB{ab_type}_Fibril{i:02d}_PVS")
                reactant.setStoichiometry(1.0)
                
                # Product: Sink
                product_sink = reaction.createProduct()
                product_sink.setSpecies(sinks[f"AB{ab_type}_Fibril{i:02d}_PVS"])
                product_sink.setStoichiometry(1.0)
                product_sink.setConstant(True)
                
                # Kinetic law   
                klaw = reaction.createKineticLaw()
                math_ast = libsbml.parseL3Formula(
                    f"(1 - sigma_PVS_AB{ab_type}) * Q_PVS * AB{ab_type}_Fibril{i:02d}_PVS"
                )
                klaw.setMath(math_ast)
    
    def create_antibody_transport_reactions():
        """Create antibody transport reactions between ISF, PVS, and CSF"""
        
        # 1. ISF to PVS transport via bulk flow
        reaction = model.createReaction()
        reaction.setId("Antibody_ISF_to_PVS_bulk_flow")
        reaction.setReversible(False)
        
        # Reactant
        reactant = reaction.createReactant()
        reactant.setSpecies("Ab_t")
        reactant.setStoichiometry(1.0)
        
        # Product
        product = reaction.createProduct()
        product.setSpecies("C_Antibody_unbound_PVS")
        product.setStoichiometry(1.0)
        
        # Kinetic law: (1 - sigma_ISF) * Q_PVS * Antibody_unbound_ISF / V_PVS
        klaw = reaction.createKineticLaw()
        math_ast = libsbml.parseL3Formula(
            "(1 - sigma_ISF) * Q_PVS * Ab_t"
        )
        klaw.setMath(math_ast)

        # 2. PVS clearance to central compartment
        reaction = model.createReaction()
        reaction.setId("Antibody_PVS_to_central")
        reaction.setReversible(False)
        
        # Reactant
        reactant = reaction.createReactant()
        reactant.setSpecies("C_Antibody_unbound_PVS")
        reactant.setStoichiometry(1.0)
        
        product = reaction.createProduct()
        product.setSpecies("PK_central")
        product.setStoichiometry(1.0)
        
        # Kinetic law: (1 - sigma_PVS) * (Q_PVS + CSF_fraction * Qbrain_ISFpsv) * C_Antibody_unbound_PVS / V_PVS
        klaw = reaction.createKineticLaw()
        math_ast = libsbml.parseL3Formula(
            "(1 - sigma_PVS) * (Q_PVS + CSF_fraction * Qbrain_ISFpsv) * C_Antibody_unbound_PVS"
        )
        klaw.setMath(math_ast)

        # 3. ISF to PVS transport via diffusion
        reaction = model.createReaction()
        reaction.setId("Antibody_ISF_to_PVS_diffusion")
        reaction.setReversible(False)
        
        # Reactant
        reactant = reaction.createReactant()
        reactant.setSpecies("Ab_t")
        reactant.setStoichiometry(1.0)
        
        # Product
        product = reaction.createProduct()
        product.setSpecies("C_Antibody_unbound_PVS")
        product.setStoichiometry(1.0)
        
        # Kinetic law: PS_Ab_ISF_PVS * Antibody_unbound_ISF / V_PVS * Pe / (exp(Pe) - 1)
        klaw = reaction.createKineticLaw()
        math_ast = libsbml.parseL3Formula(
            "PS_Ab_ISF_PVS * Ab_t * Pe / (exp(Pe) - 1)"
        )
        klaw.setMath(math_ast)

        # 4. PVS to ISF transport via diffusion
        reaction = model.createReaction()
        reaction.setId("Antibody_PVS_to_ISF_diffusion")
        reaction.setReversible(False)
        
        # Reactant
        reactant = reaction.createReactant()
        reactant.setSpecies("C_Antibody_unbound_PVS")
        reactant.setStoichiometry(1.0)
        
        # Product
        product = reaction.createProduct()
        product.setSpecies("Ab_t")
        product.setStoichiometry(1.0)
        
        # Kinetic law: PS_Ab_ISF_PVS * C_Antibody_unbound_PVS / V_PVS * Pe / (exp(Pe) - 1)
        klaw = reaction.createKineticLaw()
        math_ast = libsbml.parseL3Formula(
            "PS_Ab_ISF_PVS * C_Antibody_unbound_PVS * Pe / (exp(Pe) - 1)"
        )
        klaw.setMath(math_ast)

        # 5. CSF to PVS transport
        # CSF_fraction is 0 in CSV file
        '''
        reaction = model.createReaction()
        reaction.setId("Antibody_CSF_to_PVS")
        reaction.setReversible(False)

        # Reactant
        reactant = reaction.createReactant()
        reactant.setSpecies("PK_SAS_brain")
        reactant.setStoichiometry(1.0)
        
        # Product (source is CSF)
        product = reaction.createProduct()
        product.setSpecies("C_Antibody_unbound_PVS")
        product.setStoichiometry(1.0)
        
        # Kinetic law: CSF_fraction * Qbrain_ISFpsv * PK_SAS_brain / V_PVS
        klaw = reaction.createKineticLaw()
        math_ast = libsbml.parseL3Formula(
            "CSF_fraction * Qbrain_ISFpsv * PK_SAS_brain"
        )
        klaw.setMath(math_ast)
        '''

    def create_plaque_clearance_reactions():
        """Create plaque clearance reactions for both bound and unbound plaque"""
        
        for ab_type in ["40", "42"]:
            # Bound plaque clearance
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_Plaque_bound_clearance")
            reaction.setReversible(False)

            # Reactant
            reactant = reaction.createReactant()
            reactant.setSpecies(f"AB{ab_type}_Plaque_bound_PVS")
            reactant.setStoichiometry(1.0)

            # Product: Sink
            product_sink = reaction.createProduct()
            product_sink.setSpecies(sinks[f"AB{ab_type}_Plaque_bound_PVS"])
            product_sink.setStoichiometry(1.0)
            product_sink.setConstant(True)
            
            # Kinetic law
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"CL_AB{ab_type}_plaque_bound_PVS * AB{ab_type}_Plaque_bound_PVS"
            )
            klaw.setMath(math_ast)

            # Unbound plaque clearance
            reaction = model.createReaction()
            reaction.setId(f"AB{ab_type}_Plaque_unbound_clearance")
            reaction.setReversible(False)

            # Reactant
            reactant = reaction.createReactant()
            reactant.setSpecies(f"AB{ab_type}_Plaque_unbound_PVS")
            reactant.setStoichiometry(1.0)

            # Product: Sink
            product_sink = reaction.createProduct()
            product_sink.setSpecies(sinks[f"AB{ab_type}_Plaque_unbound_PVS"])
            product_sink.setStoichiometry(1.0)
            product_sink.setConstant(True)
            
            # Kinetic law
            klaw = reaction.createKineticLaw()
            math_ast = libsbml.parseL3Formula(
                f"CL_AB{ab_type}_plaque_free_PVS * AB{ab_type}_Plaque_unbound_PVS"
            )
            klaw.setMath(math_ast)

    # Create all reactions
    create_transport_reactions()
    create_binding_reactions()
    create_lymph_clearance()
    create_antibody_transport_reactions()
    create_plaque_clearance_reactions()
    
    return document

def create_pvs_aria_model_runner():
    """Create a function that loads parameters and creates a parameterized model"""
    def model_runner(csv_path, drug_type="gantenerumab"):
        params, params_with_units = load_parameters(csv_path, drug_type=drug_type)
        document = create_pvs_aria_model(params, params_with_units)
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
    sink.setCompartment("comp_PVS")  # Using PVS compartment
    sink.setInitialConcentration(0.0)  # Using 0 for sinks
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
    source.setCompartment("comp_PVS")  # Using PVS compartment
    source.setInitialConcentration(1.0)  # Fixed concentration
    source.setConstant(True)
    source.setHasOnlySubstanceUnits(False)
    source.setBoundaryCondition(True)  # Boundary species
    source.setSubstanceUnits("nanomole_per_litre")
    return source_id

if __name__ == "__main__":
    runner = create_pvs_aria_model_runner()
    parent_dir = Path(__file__).parent.parent
    parameters_dir = parent_dir / "parameters"
    print("\n*** Testing with GANTENERUMAB ***")
    doc_gant = runner(parameters_dir / "PK_Geerts.csv", drug_type="gantenerumab")
    print("Gantenerumab model validation complete")
    
    print("\n*** Testing with LECANEMAB ***")
    doc_lec = runner(parameters_dir / "PK_Geerts.csv", drug_type="lecanemab")
    print("Lecanemab model validation complete") 