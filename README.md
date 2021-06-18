# molGA-MPc
Molecular Genetic Algorithm for Metal Phthalocyanine

Copyright © 2020 Zisheng Zhang

Please CITE [THIS PAPER](https://zishengz.github.io/docs/2021jpcc.pdf) if you use any script or model in this repo:
> Zhang, Z.; Wang, Y.-G. "Molecular Design of Dispersed Nickel Phthalocyanine@Nanocarbon Hybrid Catalyst for Active and Stable Electroreduction of CO2." J. Phys. Chem. C, ASAP. https://10.1021/acs.jpcc.1c02508.

## Dependencies
- Python 3+
- ASE
- xTB
- OpenBabel
- pyTorch

## General comments
This code is a genetic algorithm searcher for global optimization of molecular properties (adsorption energy, MO levels, atomic charges) or any performance descriptor based on them. The code is compatible specifically to metal phthalocyanines but can be adapted accordingly to any other molecular systems.

## How to use
We provide three stand-alone scripts for running GA with SQM/DNN and model training:

### GA_scripts/GA_SQM.py
- Define the substituent group dictionary as ```grpDict``` and the SMILES template for the molecular system of interest in ```gene2smi```.
- Duplicate checker ```ifDuplicate``` and ring-order adjuster ```ringGene``` may need modifications if the molecular template has a different symmetry.
- Define in ```descriptor``` how the quantity to be optimized is derived based on SQM-calculated quantities.
- ```genSymm1``` and ```genSymm2``` can be used to generate molecules with a certain pattern such as mono- or di- substitution.
- ```randPop``` is for uniform random sampling of the chemical space defined by the SMILES template and substituent dictionary.
- Set the parameters for GA according to your need. Check the comments in the script for the explanation on each variable.
- After configuring the script, run the GA search by: ```python -u GA_SQM.py```

### DNN_models/train.py
- Put the ```history.dat``` from the SQM GA run to the same directory as the training set.
- Choose the corresponding index of the ```data``` to set the property ```Y``` to predict.
- Run the training by: ```python train.py```. A validation plot and the final DNN model will be saved to the same directory.

### GA_scripts/GA_NN.py
- Use the trained DNN model to predict molecular properties in the GA search, instead of using SQM.
- Parameter settings are basically the same as in ```GA_scripts/GA_SQM.py```
- Switching between GA_NN and GA_SQM can be achieved by simple scripting in the job submission script.

