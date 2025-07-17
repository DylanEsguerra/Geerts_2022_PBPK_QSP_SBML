# IDE Model Implementation Analysis

## Equation from CSV

```
-(IDE_conc * AB42_IDE_Kcat_lin * ((AB42_Monomer * Unit_removal_1)^AB42_IDE_Hill / ((AB42_Monomer * Unit_removal_1)^AB42_IDE_Hill + AB42_IDE_IC50^AB42_IDE_Hill)))
```

### Units

| Parameter | Value | Units |
|-----------|-------|-------|
| AB42_IDE_Hill | 2 | DIMENSIONLESS |
| AB42_IDE_IC50 | 28.635642 | DIMENSIONLESS |
| AB42_IDE_Kcat_baseline | 50 | 1 / h |
| AB42_IDE_Kcat_exp | 50 | 1 / h |
| AB42_IDE_Kcat_lin | 50 | 1 / h |
| exp_decline_rate_IDE_forty_two | -1.15E-06 | 1 / h |
| exp_decline_multiplier_IDE_forty_two | 1 | DIMENSIONLESS |
| IDE_conc | 0.005 | nano * mol / L |

### Full Equation Unit Analysis

```
(nano * mol / L) * (1 / h) * (((nano*mol/L)*Unit_removal_1)^(DIMENSIONLESS) / (((nano*mol/L)*Unit_removal_1)^(DIMENSIONLESS) + DIMENSIONLESS^DIMENSIONLESS))
```

Dropping DIMENSIONLESS:
```
(nano * mol / L) * (1 / h) * (((nano*mol/L)*Unit_removal_1) / ((nano*mol/L)*Unit_removal_1))
```

Canceling ((nano*mol/L)*Unit_removal_1) in both numerator and denominator:
```
(nano * mol / L) * (1 / h) = nM/h
```

**This is exactly what we want.**

## Equation from Paper

### Clearance Term
```
− CL_b_IDE(t) ∗ IDE_conc ∗ (b0(t)^n)/(b0(t)^n + K_b_IDE^n)
```

### IDE Rate Decay Linear
```
CL_b_IDE(t) = CL_b_IDE(t0)*(1-slope*t)
```

### Parameter Mapping

| Paper Parameter | CSV Parameter |
|----------------|---------------|
| IDE_conc | IDE_conc |
| CL_b_IDE(t) | AB42_IDE_Kcat_lin |
| b0(t) | AB42_Monomer |
| n | AB42_IDE_Hill |
| K_b_IDE | AB42_IDE_IC50 |

### Decay Equation

```
CL_b_IDE(t0) = AB42_IDE_Kcat_lin
slope = exp_decline_rate_IDE_forty_two
```

**No linear slope, so we must use exponential:**
```
CL_b_IDE(t) = CL_b_IDE(t0)*(1-slope)^t
```

Or:
```
dCL_b_IDE / dt = -slope*CL_b_IDE
```

### Units for Decay
```
(1/h) * (1/h)
```

## Don's Implementation

### Species Units
- No Libsbml
- Antimony: `substanceOnly species X in Y`
- Likely implies no division of species by volume automatically

### Clearance Term Definition
```
IDE_activity_ISF = 100.2/ISF; ~ about 400 which we can set CL_AB42_IDE in Dylan Model
substanceOnly species IDE_activity_ISF in ISF;
```
IDE_conc_ISF just a regular parameter

### Clearance Term
```
IDE_conc_ISF*IDE_activity_ISF*((AB42_O1_ISF/V_ISF)^AB42_IDE_Hill_ISF/((AB42_O1_ISF/V_ISF)^AB42_IDE_Hill_ISF + AB42_IDE_IC50_ISF^AB42_IDE_Hill_ISF))*V_ISF;
```

**Unit Analysis:**
```
nM * (1/h) * ((nano*mol/L)^DIMENSIONLESS/((nano*mol/L)^DIMENSIONLESS + DIMENSIONLESS^DIMENSIONLESS)) * L
```

Remove DIMENSIONLESS:
```
nM * (1/h) * ((nano*mol/L)/((nano*mol/L))) * L
```

Cancel ((nano*mol/L)):
```
nM * (1/h) * L = nano * mol / h
```

**This is what SBML wants for RHS.**

### Decay Equation
Written as a reaction:
```
k_exp_decline_IDE*IDE_activity_ISF*V_ISF
```

**Units:** `(1/h) * (1/h) * L`

## Dylan's Implementation

### Species Units
- Libsbml HasonlySubstanceUnits = False
- ⟹ sbmltoodejax to divide each y by compartment volume
- ⟹ Antimony "species X in Y"

### Clearance Rate
```
CL_AB42_IDE = 50;
// Other declarations:
var …., CL_AB42_IDE, …..
```

Non constant parameter ⟹ sbmltoodejax does not divide this element by a volume despite it being in "y"

IDE_conc just a regular parameter

### Clearance Term
```
IDE_conc*CL_AB40_IDE*((AB40_Monomer*Unit_removal_1)^AB40_IDE_Hill/((AB40_Monomer*Unit_removal_1)^AB40_IDE_Hill + AB40_IDE_IC50^AB40_IDE_Hill))*VIS_brain;
```

**Unit Analysis:**

Dylans species are nM becauase SBMLtoODEjax definently divides the terms by volume. We belive roadrunner does this as well 

```
nM * (1/h) * ((nM*DIMENSIONLESS)^DIMENSIONLESS/((nM*DIMENSIONLESS)^DIMENSIONLESS + DIMENSIONLESS^DIMENSIONLESS)) * L
```

Remove DIMENSIONLESS:
```
nM * (1/h) * ((nM)/((nM))) * L
```

Cancel nM:
```
nM * (1/h) * L = nano*mol/h
```

**This is what SBML wants for RHS.**

### Decay Equation
Rate Rule:
```
-exp_decline_rate_IDE_fortytwo*CL_AB42_IDE
```

**Units:** `(1/h) * (1/h)`

---

## Differences Summarized

RHS looks the same in terms of units for Clearance terms. Only notable difference is that CL_AB42_IDE is a non constant parameter and IDE_activity_ISF is a substanceOnly species in ISF.

Decay rates are implemented differently:

**Don has a reaction:**
```
IDE_activity_ISF -> ; k_exp_decline_IDE*IDE_activity_ISF*V_ISF;
```

**While I have a Rate Rule:**
```
CL_AB42_IDE' = -exp_decline_rate_IDE_fortytwo*CL_AB42_IDE;
```

### Differences include:
1. Don has volume multiplication and I do not
2. Rate Rule for non constant parameter vs reaction for substanceOnly species in ISF

**Key Observation:**
- As IDE_conc → ∞, models diverge
- As IDE_conc → 0, models converge

---

## How to Unite Models

**(1)** Fixing this on its own does not unify our models but it does make IDE_activity_ISF and CL_AB42_IDE curves match when neither solution is divided by a volume.

**Models do match when standardizing (1) and either:**
- **A)** Adding a second Volume multiplication to Dylan's Clearance Term

```
IDE_conc*CL_AB40_IDE*((AB40_Monomer*Unit_removal_1)^AB40_IDE_Hill/((AB40_Monomer*Unit_removal_1)^AB40_IDE_Hill + AB40_IDE_IC50^AB40_IDE_Hill))*VIS_brain*VIS_brain;
```

- **B)** Multiplying initial IDE_conc in Dylan's model by Volume
```
rr.setValue('IDE_conc', 3 * 0.2505)
```

Inverse of these modifications applied to Don's model also standardizes results.

---

## What Causes This Difference and Why Does the Fix Work?

**Part 1 of solution works because** we need to standardize formula for clearance rate decay.

**Does this imply that the reaction for a substance only species = Rate Rule for a non constant parameter?**

**Part 2 of solution works because** Either Somewhere in Don's model is an additional multiplication by volume or in my model is an additional division 

When setting initial value for IDE_activity_ISF that value may be multiplied by volume before it is used 

```
IDE_activity_ISF = 100.2/ISF;

```

Initial value for IDE_Activity is 100.2 NOT 400 

This is close to the answer and also makes models agree. However the IDE_activity_ISF and CL_AB42_IDE curves are now off by the volume but all other resulst agree. 

This suggests the volume division is still being done Dylans model or multiplication in Dons 

---

## What is Correct?

---

## Additional Investigation Tools

### Comparison Script and Test Files

To further investigate the differences between the two model implementations, we have created:

1. **`compare_models_combined.py`** - A comprehensive comparison script that can switch between:
   - **Antimony mode**: Uses Antimony text files with CL_AB42_IDE = 400
   - **SBML mode**: Uses SBML XML files with CL_AB42_IDE = 100.2

2. **Test Files**:
   - **`combined_master_model.xml`** - Dylan's SBML model (original version)
   - **`combined_master_model_MTK_Microglia2_Antimony.txt`** - Dylan's Antimony model (modified version)
   - **`Antimony_PBPK_model.xml`** - Don's SBML model
   - **`Antimony_PBPK_model.txt`** - Don's Antimony model

### Usage for Further Investigation

```bash
# Compare using SBML files (CL_AB42_IDE = 100.2)
python compare_models_combined.py --mode sbml --years 100

# Compare using Antimony files (CL_AB42_IDE = 400)  
python compare_models_combined.py --mode antimony --years 100
```

The comparison script generates:
- Six-panel comparison plots showing oligomers, protofibrils, SUVR, monomers, IDE activity, and plaque dynamics
- CSV output files for both models
- Clear documentation of which parameters were used for each comparison

This tool can be used to:
- Investigate the volume handling differences between implementations
- Compare the parameter vs species approaches for IDE clearance
- Validate that the models produce equivalent results when properly calibrated
- Explore the scaling differences in IDE clearance plots