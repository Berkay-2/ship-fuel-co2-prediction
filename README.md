# ship-fuel-co2-prediction
# Data-Driven Prediction of Ship Fuel Consumption and CO₂ Emissions
Operational Decision Support – Master’s Thesis Code & Data

This repository contains the implementation and datasets used in my thesis:
**“Data-Driven Prediction of Ship Fuel Consumption and CO₂ Emissions for Operational Decision Support.”**

## What this project does
- Trains and evaluates regression models to predict:
  - **Fuel consumption**
  - **CO₂ emissions**
- Saves metrics and best-performing model pipelines for reproducibility.

## Repository contents
- `run_thesis_models.py` — main training & evaluation script
- `ship_fuel_efficiency.csv` — main dataset (must include targets: `fuel_consumption`, `CO2_emissions`)
- `Ship_Performance_Dataset.csv` — optional regression experiment (e.g., `Operational_Cost_USD` if present)
- `marine_engine_fault_dataset.csv` — optional classification experiment (`Fault_Label` if present)
- `requirements.txt` — core dependencies
- `requirements-optional.txt` — optional ML libraries (XGBoost/LightGBM/CatBoost)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
