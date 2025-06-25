# Geerts Model: Open-Source PBPK-QSP Model for Alzheimer's Disease

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/DylanEsguerra/Geerts_2023_PBPK_QSP_SBML_Model)

This repository contains an open-sourced implementation of the Geerts et al. 2023 combined physiologically-based pharmacokinetic and quantitative systems pharmacology model for modeling amyloid aggregation in Alzheimer's disease.

We have chosen to use open-sourced tools for all aspects of the model development process and are emphasizing a modular design process. We are using the Systems Biology Markup Language to save the model in XML format with the hope that other researchers can use it with the software of their choice. Our model is written in python with libsbml and solved using JAX diffrax library as well as Tellurium. 

## Original Model Citation

Geerts H, Walker M, Rose R, et al. A combined physiologically-based pharmacokinetic and quantitative systems pharmacology model for modeling amyloid aggregation in Alzheimer's disease. CPT Pharmacometrics Syst Pharmacol. 2023; 12: 444-461. doi:10.1002/psp4.12912

## Project Status

✅ **Single Dose Antibody PK**: Successfully reproduces published plasma pharmacokinetics  
⚠️ **Amyloid Beta Aggregation**: Currently investigating discrepancies in aggregation dynamics  
✅ **Model Translation**: Validated published equations to SBML conversion with direct ODE implementation  

## Quick Start

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Navigate to Tellurium folder for fast initial simulation
cd Tellurium
# Run a quick simulation
python run_default_simulation.py
```

## Documentation

📖 **[Usage Instructions](Usage.md)** - How to install, run simulations, and create visualizations  
📊 **[Results](Results.md)** - Model validation, comparisons, and sensitivity analysis  
🔬 **[Model Details](Model_Details.md)** - Technical implementation, modules, and methodology  


## Key Features

- **Modular Design**: Separate modules for PBPK and QSP components
- **Multiple Implementations**: SBML, ODE, and Tellurium versions
- **Comprehensive Validation**: Cross-validation between implementations
- **Parameter Documentation**: Traceable parameter sources and validation status
- **Visualization Tools**: Extensive plotting and analysis capabilities

## Repository Structure

```
├── Modules/                    # Individual model components
├── ODE_version/               # Direct ODE implementation
├── Tellurium/                 # Tellurium/RoadRunner implementation
├── parameters/                # Parameter files and documentation
├── generated/                 # Auto-generated models and results
├── run_*.py                   # Main simulation scripts
└── visualize_*.py            # Visualization utilities
```

## Important Notes

⚠️ **JAX Configuration**: Before running any scripts, set the environment variable:
```python
import os
os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
```

## Contributing

We welcome contributions from those familiar with this model to improve model accuracy and expand functionality. Please see our documentation for detailed information about the model structure and validation status.