## Model Validation: Single Dose Antibody PK

The figure below demonstrates that our model accurately reproduces the published plasma pharmacokinetics (PK) for a single 300 mg subcutaneous dose of gantenerumab. The solid black line shows the model output, while the red circles represent experimental data from the literature.

![Gantenerumab Plasma PK: 300 mg subcutaneous](generated/figures/drug_simulation/gantenerumab_pk_brain_plasma_master.png)

*Replication of Figure 4: Comparison of model-predicted and experimental plasma concentrations for gantenerumab after a single 300 mg subcutaneous dose. The model closely matches the published data, validating our implementation for this scenario.*

However we running out model for the natural life cycle of amyloid aggregation shows significant differences from the published material. 

![Natural Life Cycle](generated/figures/steady_state/gantenerumab_ab42_ratios_and_concentrations.png)

*Replication of Figure 3: Simulation of the natural life cycle (from 20–100 years) of amyloid aggregation for an individual virtual Alzheimer's disease patient as in an observational study. Age-related pathology is implemented as a exponential decrease in monomer degradation. The publication used linear, but stated that both provide similar results and only the exponential parameters were made available. This decline resulted in an increased amount of monomers being pushed into the aggregation pathway. We fail o reproduce the steep transition around the age of 60 years followed by a slow saturation at higher ages for the ABeta42/40 ratio, ISF or CSF ABeta 42 concnetration*


## Known Limitations and Validation

### Current Bug in Amyloid Beta Aggregation

The model currently has an unidentified bug or misspecified parameter affecting the amyloid beta aggregation pathway. This issue causes excessive monomer generation without appropriate formation of oligomers and fibrils. The aggregation dynamics do not match the expected behavior described in the Geerts et al. 2023 paper. Specifically:

- Monomers accumulate at higher than expected concentrations
- Oligomer and fibril formation rates are too low
- No tranistion in monomer/oligomer/fibril values at age of 60

This issue is being actively investigated using the validation tools described below.

### Parameter Validation

The model parameters are stored in [`parameters/PK_Geerts.csv`](parameters/PK_Geerts.csv) with enhanced documentation. Here is an example of the parameter file structure:

| name | value | units | Sup_Name | Source | Validated | Notes |
|------|-------|-------|----------|--------|-----------|-------|
| Lec_Vcent | 3.18 | L | Vcent_BAN2401 | Chang HY 2019 | 0 | No central in Chang |
| Gant_Vcent | 19.69 | L | Vcent_Gantenerumab | Chang HY 2019 | 0 | No central in Chang |
| Lec_Vper | 2.24 | L | Vper_BAN2401 | Chang HY 2019 | 0 | No Periph in Chang |
| Gant_Vper | 1 | L | Vper_Gantenerumab | Chang HY 2019 | 0 | No Periph in Chang |

Each parameter includes:
- Its original name from the Geerts supplement in addition to the name used in this implementation 
- Sources are explicitly documented for each parameter
- A "Validated" binary column (validated = 1) indicates whether the parameter has been personally verified by the model developers
- Units are standardized and documented

This structured approach allows for systematic validation of parameter values against the literature.