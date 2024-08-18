# SLEET (SchNet-based Ligand-Embedding Extending Transformer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)

Source paper: 


##### Requirements:：

- **anaconda or minconda**
- python >= 3.9
- ase
- einops
- fasteners
- hydra == 1.2
- numpy < 1.25.0
- pandas >= 1.4.3
- pytorch >= 2.0
- pytorch-lightning >= 2.0.0 if your training task is very slow when uses gpu, try others version pl.
- rdkit == 2023.9.1
- rich

## Installation

### Install requirements

#### Install anaconda / miniconda

See [here](https://www.anaconda.com/download/success).

#### Update anaconda / miniconda

```shell
conda update -n base -c defaults -c conda-forge conda
```

#### Install all other requirements with anaconda / miniconda

```shell
cd root/path/of/SLEET
conda env create -f env/environment.yml
```


## Getting started

### For Linux

#### With slurm system:

Change configs to your own setting in run_slurm.sh files:

```sh
#SBATCH --mail-user=your_mail@address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --job-name=any_custom_name
#SBATCH --account=your_account
```

Submit job:

```shell
sbatch run_[train/test]_slurm.sh
```

#### If not slurm system:

```shell
sbatch run_[train/test].sh
```

### For windows

Typing in command line:

```shell
run_[train/test].bat
```

## Trained model
Please download trained model from [here](https://huggingface.co/MKTMLTeam/SLEET/tree/main), and then extracts it to SLEET root folder.

## Acknowledgements

Our codes are based on *SchNetPack 2.0*, more details about code can be found [here](https://github.com/atomistic-machine-learning/schnetpack).

*Relative Position Encoding* code can be found [here](https://github.com/evelinehong/Transformer_Relative_Position_PyTorch).

Main and hydra configs for PyTorch Lightning are adapted from this template: [![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko. *Quantum-chemical insights from deep tensor neural networks.* Nature Communications **8**. 13890 (2017) [10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller. *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [Paper](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller. *SchNet - a deep learning architecture for molecules and materials.*
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] K. T. Schütt, O. T. Unke, M. Gastegger. *Equivariant message passing for the prediction of tensorial properties and molecular spectra.*
International Conference on Machine Learning (pp. 9377-9388). PMLR, [Paper](https://proceedings.mlr.press/v139/schutt21a.html).

* [5] K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger. *SchNetPack 2.0: A neural network toolbox for atomistic machine learning.* J. Chem. Phys. 158 (14): 144801. (2023) [10.1063/5.0138367](https://doi.org/10.1063/5.0138367)

* [6] K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller. *SchNetPack: A Deep Learning Toolbox For Atomistic Systems.* J. Chem. Theory Comput. 15 (1): 448-455. (2019) [10.1021/acs.jctc.8b00908](http://dx.doi.org/10.1021/acs.jctc.8b00908)

* [7] Ye, Z.-R.; Hung, S.-H.; Chen, B.; Tsai, M.-K. *Assessment of Predicting Frontier Orbital Energies for Small Organic Molecules Using Knowledge-Based and Structural Information.* ACS Eng. Au, 2, pp. 360–368 (2022) [10.1021/acsengineeringau.2c00011](10.1021/acsengineeringau.2c00011)

* [8] Hung, S.-H.; Ye, Z.-R.; Cheng, C.-F.; Chen, B.; Tsai, M.-K. *Enhanced Predictions for the Experimental Photophysical Data Using the Featurized Schnet-base Approach.* J. Chem. Theory Comput. (2023) [10.1021/acs.jctc.3c00054](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00054) [ResearchGate](https://www.researchgate.net/publication/367022743_Enhanced_Predictions_for_the_Experimental_Photophysical_Data_Using_the_Featurized_Schnet-bondstep_Approach)

* [9] S. Peter, U. Jakob, V. Ashish. *Self-Attention with Relative Position Representations.* NAACL 2018. (2018) [10.48550/arXiv.1803.02155](https://doi.org/10.48550/arXiv.1803.02155)
