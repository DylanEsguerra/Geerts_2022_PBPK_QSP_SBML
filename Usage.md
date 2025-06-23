# Usage Instructions

This document provides comprehensive instructions for installing, running, and visualizing results from the Geerts model.

## Table of Contents
- [Installation](#installation)
- [Running Simulations](#running-simulations)
- [Alternative Implementations](#alternative-implementations)
- [Visualization Options](#visualization-options)
- [Dependencies and Configuration](#dependencies-and-configuration)

## Installation

### Prerequisites
- Python 3.12
- Virtual environment (recommended)

### Setup Steps

1. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure JAX (Important):**
   Before running any scripts, set this environment variable to avoid JAX runtime issues:
   ```python
   import os
   os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
   ```

   This is due to a known diffrax/JAX issue with jax==0.5.3.
   See [XLA_FLAGS issue](https://github.com/jax-ml/jax/discussions/25711) for details.

## Running Simulations

### Main SBML-Based Simulations

#### 1. Multi-Dose Simulation (Full Model)
```bash
python run_combined_master_model_multi_dose.py --drug {gantenerumab,lecanemab}
```
- **Runtime**: TBT
- **Purpose**: Complete treatment simulation with full dosing regimen (3 years)
- **Output**: Comprehensive drug and Aβ dynamics over treatment period

#### 2. No-Dose Simulation (Natural History)
```bash
python run_no_dose_combined_master_model.py --drug {gantenerumab,lecanemab} --years 20
```
- **Runtime**: TBT
- **Purpose**: Natural amyloid aggregation without treatment. This can be used to set the initial condition (70 years) for drug studies
- **Arguments**: 
  - `--drug`: Required for volume parameter selection
  - `--years`: Simulation duration (default: 100)

#### 3. Single-Dose Simulation (Validation)
```bash
python run_combined_master_model.py --drug {gantenerumab,lecanemab}
```
- **Runtime**: TBT
- **Purpose**: Single dose validation against published data
- **Process**: Runs steady-state first, then applies single dose

### Command Line Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--drug` | `gantenerumab`, `lecanemab` | Antibody type (affects parameters and dosing method) |
| `--years` | Integer | Simulation duration for no-dose runs |
| `--outdir` | Path | Output directory (ODE version only) |

### Viewing Results
- Generated plots are saved in `generated/figures/` directory
- Different simulation types create subdirectories:
  - `drug_simulation/`: Single/multi-dose results
  - `steady_state/`: No-dose simulation results
  - `comparison/`: Model validation comparisons

## Alternative Implementations

### ODE-Based Implementation (Faster)

For faster simulations and parameter exploration, use the direct ODE implementation:

```bash
cd ODE_version
python run_no_dose.py --drug {gantenerumab,lecanemab} --years 20
```

**Key Features:**
- Directly implements published equations from Supplementary Tables
- Much faster computation time
- Validated against SBML implementation
- Uses exact parameter names from source material

**Generated Outputs:**
- AB42/AB40 ratios over time
- Oligomer and monomer loads
- Plaque and fibril dynamics
- SUVR progression (not validated)
- Microglia activation
- Species composition for both AB40 and AB42

**Limitations:**
- Microglia cell count limitations when antibodies are present
- Uses separate parameter file (`Geerts_Params2.csv`)
- Not set up for drug dosing

### Tellurium Implementation (Parameter Analysis)

For sensitivity analysis and parameter exploration:

```bash
cd Tellurium
python run_default_simulation.py
python agg_rate_sensitivity.py  # Aggregation rate sensitivity
python CL_sensitivity.py        # IDE clearance sensitivity
```

**Advantages:**
- Fast simulations for parameter exploration
- Validated against other implementations
- Ideal for sensitivity analysis

**Current Limitation:**
- Multi-dose simulations not currently supported
- Lower numerical stability

## Visualization Options

### Real-time Visualization
- Plots are automatically generated during simulation runs
- Saved to appropriate subdirectories in `generated/figures/`

### Post-simulation Visualization

#### 1. Full Model Results
```bash
python plot_saved_solution.py --drug {gantenerumab,lecanemab}
```
- Visualizes multi-dose simulation results
- Creates comprehensive plots of drug concentrations and Aβ dynamics

#### 2. Steady State/No-Dose Results  
```bash
python visualize_steady_state.py --drug {gantenerumab,lecanemab} --years 20
```
- Visualizes natural history simulations
- Useful for understanding baseline Aβ behavior
- Saves plots to `generated/figures/steady_state/`

#### 3. Model Comparison
```bash
python compare_no_dose_models.py --drug {gantenerumab,lecanemab} --years 20
```
- Compares SBML vs ODE implementations
- Requires data from both implementations to be generated first:
  ```bash
  # Generate SBML data
  python run_no_dose_combined_master_model.py --drug gantenerumab --years 20
  
  # Generate ODE data  
  cd ODE_version
  python run_no_dose.py --drug gantenerumab --years 20
  ```

### Key Visualizations Generated

- **AB42 Ratios and Concentrations**: Brain plasma AB42/AB40 ratio, ISF AB42, CSF AB42
- **Drug Pharmacokinetics**: Plasma and brain antibody concentrations
- **Amyloid Species**: Individual oligomer, fibril, and plaque concentrations
- **Microglia Dynamics**: Activation and cell count changes
- **Transport Dynamics**: Blood-brain barrier and CSF transport

## Model Generation Process

The SBML model is automatically generated through these steps:

1. **Module Combination**: `Combined_Master_Model.py` merges all modules
2. **SBML Export**: Model saved as XML in `generated/sbml/`
3. **JAX Conversion**: SBML converted to JAX using `sbml_to_ode_jax`
4. **Simulation**: Uses `diffrax` for ODE solving

**Generated Files:**
- `generated/sbml/combined_master_model_{drug_type}.xml`: SBML model
- `generated/jax/combined_master_model_jax.py`: JAX implementation

## Runtime Considerations

| Simulation Type | Typical Runtime | Use Case |
|----------------|----------------|----------|
| Multi-dose (SBML) | hours | Complete treatment analysis |
| No-dose (SBML) |  hours | Natural history, validation |
| Single-dose (SBML) | hours | Pharmacokinetic validation |
| No-dose (ODE) | minutes | Fast parameter exploration |
| Tellurium sensitivity | seconds | Parameter sensitivity analysis |

**Optimization Tips:**
- Use ODE version for parameter exploration
- Use Tellurium for sensitivity analysis  
- Run shorter simulations (fewer years) for testing

## Dependencies and Configuration

### Required Dependencies
```txt
jax==0.5.3
jaxlib==0.5.3
diffrax==0.7.0
python-libsbml
sbml_to_ode_jax
tellurium
matplotlib
pandas
numpy
```

### Important Configuration Notes

1. **JAX Configuration**: Always set the XLA flag before running JAX-based scripts
2. **Parameter Files**: 
   - SBML version uses `parameters/PK_Geerts.csv`
   - ODE version uses `ODE_version/Geerts_Params_2.csv`
   - Tellurium uses the XML file from the SBML version 
3. **Output Directories**: Automatically created if they don't exist
4. **Directory Navigation**: All scripts are set up to be run from the directory they exist in

### Troubleshooting

**Common Issues:**
- JAX runtime errors: Ensure XLA_FLAGS is set
- Long runtimes: Consider using ODE version or Tellurium for simulations
- Missing dependencies: Run `pip install -r requirements.txt`
- File path errors: Run scripts from the directory they exist in