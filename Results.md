# Results

This document presents the key validation results, model comparisons, and sensitivity analyses for the Geerts model implementation.

## Table of Contents
- [Model Validation](#model-validation)
- [Implementation Comparisons](#implementation-comparisons)  
- [Sensitivity Analysis](#sensitivity-analysis)
- [Known Limitations](#known-limitations)

## Model Validation

### Single Dose Antibody Pharmacokinetics (Figure 4 Replication)

The figure below demonstrates that our model accurately reproduces the published plasma pharmacokinetics (PK) for a single 300 mg subcutaneous dose of gantenerumab. The solid black line shows the model output, while the red circles represent experimental data from the literature extracted using WebPlotDigitizer.

![Gantenerumab Plasma PK: 300 mg subcutaneous](generated/figures/drug_simulation/gantenerumab_pk_brain_plasma_master.png)

*Replication of Figure 4: Comparison of model-predicted and experimental plasma concentrations for gantenerumab after a single 300 mg subcutaneous dose. The model closely matches the published data, validating our implementation for this scenario.*

### Natural Life Cycle Simulation (Figure 3 Replication)

However, running our model for the natural life cycle of amyloid aggregation shows significant differences from the published material.

![Natural Life Cycle](generated/figures/steady_state/gantenerumab_ab42_ratios_and_concentrations.png)

*Replication of Figure 3: Simulation of the natural life cycle (from 20–100 years) of amyloid aggregation for an individual virtual Alzheimer's disease patient as in an observational study. Age-related pathology is implemented as an exponential decrease in monomer degradation. The publication used linear decline, but stated that both provide similar results and only the exponential parameters were made available. This decline resulted in an increased amount of monomers being pushed into the aggregation pathway. We fail to reproduce the steep transition around the age of 60 years followed by a slow saturation at higher ages for the ABeta42/40 ratio, ISF or CSF ABeta 42 concentration.*

![Figure 3 Geerts](generated/figures/Figure3.png)
*Published Fig 3*

**Key Discrepancies:**
- ❌ Missing steep transition around age 60
- ❌ Insufficient saturation behavior at higher ages  
- ❌ ISF and CSF ABeta42 concentrations don't match published curves
- ❌ ABeta 42/40 ratio shows the same issue although it does match in terms of scale

## Implementation Comparisons

### SBML vs ODE Model Agreement

This analysis was performed to diagnose why we failed to reproduce the natural life cycle results. 

Cross-validation between our modular SBML implementation and the direct ODE implementation shows excellent agreement, confirming the accuracy of our SBML-to-reactions translation.

These results confirm that we have accurately translated the ODE system to a reaction network in SBML and that the issue lies in the published material itself.

![ODE and SBML brain plasma comparison](generated/figures/comparison/compare_ab42_ab40_monomer_ratio.png)

*Comparison of SBML-based and direct ODE implementations for AB42/AB40 monomer ratios in brain plasma. The excellent agreement validates our translation from ODEs to reaction-based SBML format.*

**Validation Results:**
- ✅ Brain plasma AB42/AB40 ratios
- ✅ ISF concentrations
- ✅ CSF concentrations
- ✅ Oligomer dynamics
- ✅ Overall kinetics: Excellent agreement

### Tellurium Implementation Validation

The ability to use additional solver tools without needing to re write out model shows the benefit of SBML and being able to save models in XML format. 

We have confirmed that the Tellurium/RoadRunner simulation offers the same results for the no dose case as our JAX version. 
This is useful for debugging our natural life cycle issue as Tellurium is much faster. 

- ✅ Identical results to SBML implementation in no-dose simulations
- ✅ Faster simulation times for parameter exploration
- ✅ Robust numerical stability across parameter ranges
- ✅ Consistent with both SBML and ODE versions


However we are currently unable to get Tellurium to work for our multi dose tests. 


## Sensitivity Analysis

### Aggregation Rate Sensitivity

Analysis of sensitivity to amyloid beta aggregation rates reveals key model behaviors:

![Dimer to Trimer Rate Sensitivity](Tellurium/simulation_plots/sensitivity_analysis/agg_rate_sensitivity_kb1.png)

*Sensitivity analysis showing the effects of varying backward rate constants (kb0, kb1) on monomer, oligomer, fibril, and plaque concentrations over a 100-year simulation.*

![Dimer to Trimer Rate Sensitivity and Gain values](Tellurium/simulation_plots/sensitivity_analysis/agg_rate_extrapolation_sensitivity.png)

*Sensitivity analysis showing the effects of varying backward rate constants (kb0, kb1), higher order backward rates, and calculated Gain factors*

**Key Findings:**

- Much lower backward rates can result in more interesting Abeta42 dynamics and Gain factors that more closely match those published in the supplement


**Parameter Ranges Tested:**
- kb0 (dimer dissociation Abeta 42): [0.1, 50.0]
- kb1 (trimer dissociation Abeta 42): [1e-05, 0.01] 
- Forward rates: Maintained at literature values

## Known Limitations

### Current Issues Under Investigation

#### Amyloid Beta Aggregation Dynamics

The model currently has unidentified issues affecting the amyloid beta aggregation pathway:

**Symptoms:**
- Monomers accumulate at higher than expected concentrations
- Oligomer and fibril formation concentrations are too low
- No transition in monomer/oligomer/fibril values around age 60

**Investigation Status:**
- ❌ **Root Cause**: Under active investigation
- We feel the root cause is most likely in the way aggregation rates are extrapolated from the low order rates. 
- Backward rates are significantly higher than forward for Ab42 perhaps leading to the overabundance of monomers. 
- Another potential issue is the IDE clearance which is the only time dependent feature that could cause an age related cascade
- These two features have the least details in the supplementary material
- 🔍 **Parameter Review**: Systematic validation of parameters. Many parameters match one-to-one between equations in TableS1 and parameters in TableS2, but a subset had ambiguous names leading to our need to guess where they fit in.  

#### Parameter Validation Gaps

Some parameters lack direct experimental validation:

| Parameter Category | Validation Status | Notes |
|-------------------|------------------|-------|
| Aggregation rates | ⚠️ Partial | Small oligomer rates validated |

### Computational Performance

**Runtime Considerations:**
- Full multi-dose simulation: NA
- No-dose simulation (20 years): NA
- Parameter sensitivity analysis: NA

**Optimization Approaches:**
- Use Tellurium or ODE version for sensitivity analysis
- Scipy or JAX Optax for optimization

## Validation Roadmap

### Short-term Goals
1. **Parameter Investigation**: Systematic review of all parameters in the supplementary Tables
2. **Literature Cross-validation**: Detailed comparison with additional published datasets
3. **Parameter Sensitivity Completion**: Full sensitivity analysis across all uncertain parameters


### Long-term Goals
1. **Open Sourced Alzheimer's Disease Models**: Make this model along with others available on the BioModels Repository

2. **Modular Design**: Using the modular design we have developed we aim to have a mix and match set of modules that can be recombined to gain additional insight. 
