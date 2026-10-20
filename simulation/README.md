# 📡 Antenna Full-Wave EM Simulation

This folder contains MATLAB scripts to run rigorous full-wave Method of Moments (MoM) electromagnetic simulations of AI-generated 12×12 pixel antennas.

## 📋 Requirements
- MATLAB (R2020a or later)
- **Antenna Toolbox**

## 🚀 How to Run
1. Open MATLAB and navigate to this `simulation/` directory.
2. Run `simulate_antenna.m`:
   ```matlab
   simulate_antenna
   ```
3. To simulate a specific design generated from the dashboard, place the downloaded `antenna_mask_*.csv` file in this folder and change `maskFile = 'your_mask.csv';` at the top of `simulate_antenna.m`.

## ⏱️ Simulation Time
Each 12×12 pixel mesh simulation takes approximately **1 to 2.5 minutes** on a standard multi-core CPU.
