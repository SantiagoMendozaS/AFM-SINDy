# Code for AFM-SINDy: Symbolic Regression for Tip–Sample Force Reconstruction in Dynamic AFM 

This repository contains the implementation of AFM-SINDy, a tailored machine learning framework for reconstructing tip–sample interaction forces in dynamic atomic force microscopy (AFM).
The method implements the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm to perform symbolic regression on time-domain AFM trajectories. It identifies the underlying non-smooth 
and nonlinear interaction forces that govern tapping-mode dynamics. The approach is validated on synthetic data generated from a Derjaguin–Muller–Toporov Kelvin–Voigt model, successfully recovering 
viscoelastic parameters and the transition point between attractive and repulsive regimes.   

## Installation Guide

### Prerequisites

- Python 3.9.7
- Conda
- Installation of packages found in the file "AFM_SINDy_dependencies.txt"

### Steps

1. **Set Up Conda Environment:**

   Create a conda environment and activate it:

   ```bash
   conda create -n your_env_name python=3.9.7
   conda activate your_env_name
   ```

2. **Install Required Packages:**

   To run the Jupyter notebooks, install the Jupyter Notebook package using pip or conda. In addition, install all the required dependencies listed in the file "AFM_SINDy_dependencies.txt".
  

3. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-repository-link
   cd your-repository-directory

3. **Edit the directory path:**

   To run the Jupyter notebooks, put all the codes in a single folder and update the path at the begining of each Jupyter notebook and Python file in the designated `path` code line. 

### Overview of Analyzed Cases

This code runs in multiple stages. First, it reduces the candidate function library by applying both non-contact and contact constraints, narrowing it down to the most relevant 
functions. Next, a training stage using non-contact constraints identifies the best equation of motion for the attractive regime. After validation, the intermolecular distance 
is estimated to determine the transition point between attractive and repulsive regimes. The second training stage focuses on repulsive interactions, using only the clusters 
located beyond the identified intermolecular distance, implementing contact constraints. This stage aims to determine the best-fitting equation of motion for the repulsive regime. 
The resulting equations are benchmarked against the Derjaguin–Muller–Toporov (DMT) model, with and without surface dissipation modeled using the Kelvin–Voigt formulation.

**Derjaguin–Muller–Toporov Case**:

- **AFM_SINDy_lib_reduc_att_const_DMT.py:** This Python script enables the implementation of a wide range of candidate functions to train multiple preliminary SINDy
  models using the DMT model with a hard sample when assuming non-contact constraints. When configured with an extensive number of candidate functions, this script helps identify which functions are more representative
  of the observed dynamics. Additionally, when employing a reduced candidate function library, the code can be modified to train and save multiple SINDy models, which can subsequently
  be utilized in the AIC Jupyter notebook files to determine the most suitable equation of motion for each AFM dynamical regime.  

- **AFM_SINDy_lib_reduc_rep_const_DMT.py:** This Python script enables the implementation of a wide range of candidate functions to train multiple preliminary SINDy
  models using the DMT model with a hard sample when assuming contact constraints. When configured with an extensive number of candidate functions, this script helps identify which functions are more representative
  of the observed dynamics. Additionally, when employing a reduced candidate function library, the code can be modified to train and save multiple SINDy models, which can subsequently
  be utilized in the AIC Jupyter notebook files to determine the most suitable equation of motion for each AFM dynamical regime.

- **AFM_SINDy_AIC_att_const_DMT.ipynb:** This Jupyter Notebook performs an Akaike Information Criterion (AIC) analysis to identify the model types that most accurately
  replicate the validation data when using a DMT model on a hard surface, including both the phase space and tip–sample forces. It also highlights which models with less
  information loss appear most frequently across clusters in the computational domain of the training data. This specific notebook includes non-contact constraints, making
  it suitable for selecting the best equation of motion for the attractive interaction regime. Additionally, it estimates the intermolecular distance, which can be used later
  as a fixed parameter when training models under contact constraints.

- **AFM_SINDy_AIC_rep_const_DMT.ipynb:** This Jupyter Notebook performs an Akaike Information Criterion (AIC) analysis to identify the model types that best replicate
  the validation data when using a DMT model on a hard surface, including both the phase space and the tip–sample force. It also highlights which models with less information
  loss appear most frequently across clusters in the computational domain of the training data. This notebook applies contact constraints, making it suitable for selecting the
  optimal equation of motion for the repulsive interaction regime once the equations for the attractive regime have also been identified. In addition, this notebook simulates
  both regimes, allowing for a full evaluation of the identification accuracy across both attractive and repulsive regimes.


**Derjaguin–Muller–Toporov with Kelvin–Voigt Case**:

- **AFM_SINDy_lib_reduc_att_const_DMT_KV.py:** This Python script enables the implementation of a wide range of candidate functions to train multiple preliminary SINDy
  models using the DMT model with Kelvin-Voigt when assuming non-contact constraints. When configured with a substantial number of candidate functions, this script helps identify which functions are more representative
  of the observed dynamics. Additionally, when employing a reduced candidate function library, the code can be modified to train and save multiple SINDy models, which can subsequently
  be utilized in the AIC Jupyter notebook files to determine the most suitable equation of motion for each AFM dynamical regime.

- **AFM_SINDy_lib_reduc_rep_const_DMT_KV.py:** This Python script enables the implementation of a wide range of candidate functions to train multiple preliminary SINDy
  models using the DMT model with Kelvin-Voigt when assuming non-contact constraints. When configured with an extensive number of candidate functions, this script helps identify which functions are more representative
  of the observed dynamics. Additionally, when employing a reduced candidate function library, the code can be modified to train and save multiple SINDy models, which can subsequently
  be utilized in the AIC Jupyter notebook files to determine the most suitable equation of motion for each AFM dynamical regime.

- **AFM_SINDy_AIC_att_const_DMT_KV.ipynb:** This Jupyter Notebook performs an Akaike Information Criterion (AIC) analysis to identify the model types that most accurately
  replicate the validation data when using a DMT model with Kelvin-Voigt, including both the phase space and tip–sample forces. It also highlights which models with less
  information loss appear most frequently across clusters in the computational domain of the training data. This specific notebook includes non-contact constraints, making
  it suitable for selecting the best equation of motion for the attractive interaction regime. Additionally, it estimates the intermolecular distance, which can be used later
  as a fixed parameter when training models under contact constraints.

- **AFM_SINDy_AIC_rep_const_DMT_KV.ipynb:** This Python script enables the implementation of a diverse amount of candidate functions to train multiple preliminary SINDy
  models using the DMT model with Kelvin-Voigt when assuming contact constraints. When configured with an extensive number of candidate functions, this script helps identify which functions are more representative
  of the observed dynamics. Additionally, when employing a reduced candidate function library, the code can be modified to train and save multiple SINDy models, which can subsequently
  be utilized in the AIC Jupyter notebook files to determine the most suitable equation of motion for each AFM dynamical regime.

### Additional Files

- **AFM_SINDy_algorithm_training.py & AFM_SINDy_algorithm_training.ipynb:** These Python scripts and Jupyter notebooks contain the functions used throughout this repository.
  They are imported and executed automatically by the corresponding code files when needed.
