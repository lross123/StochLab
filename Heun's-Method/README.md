# Cell Migration Modelling

## Overview
This repository investigates **cell migration in biological tissues** using a combination of  
**stochastic lattice-based models, PDE continuum approximations, and discrete simulations**.  
It focuses on comparing different **measurement error models** and demonstrates how to fit  
simulation outputs to experimental-style data across multiple scenarios.

## Key Features
- **Case 1 – Single Subpopulation**  
  - PDE solver coupled with additive Gaussian error model.  
  - Multinomial error model alternative.  
  - Discrete agent-based random walk simulation.  

- **Case 2 & Case 3 – Extended Scenarios**  
  - Multi-subpopulation interactions.  
  - Calibration against synthetic and experimental-style datasets.  
  - Identifiability analysis via likelihood-based approaches.  

- **Data**  
  - Synthetic data provided (`data.csv`).  
  - Scripts for parameter estimation, identifiability analysis, and prediction intervals.

## Tech Stack
- **Julia** – numerical modelling, PDE solvers, optimisation  
- **CSV integration** – loading and processing data  
- **Statistical & probabilistic tools** – likelihood functions, error models, inference  

