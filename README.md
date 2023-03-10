# Efficient Certified Training and Robustness Verification of Neural ODEs

This repository contains the code for our ICLR 2023 [paper](https://openreview.net/forum?id=KyoVpYvWWnK), 
which allows the deterministic certification of Neural ODEs. 
The key ideas are the use of our novel controlled adaptive ODE solvers and the construction of 
a trajectory graph, which can be efficiently handled using GAINS and CURLS.
This repository contains all code and pre-trained models necessary to reproduce our results. 

The project was conducted at the [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch/).

## Setup instructions

Clone this repository:
```bash
$ git clone https://github.com/eth-sri/GAINS.git
```
Setup and activate the virtual environment as follows:
```
conda env create -f environment.yml
conda activate myenv
```

## Classification
Please find below the commands to train and certify classification models. 
For convenience's sake, we have included trained models in `models/pre-trained/` corresponding to Table 1 and Table 3.
### Training Classification
For standard training or adversarial training with perturbations of magnitude `TARGET` on `DATASET` (MNIST or F-MNIST) and random seed `SEED` use the following commands:
```
$ (myenv) python3 main.py  --dataset DATASET --seed SEED
$ (myenv) python3 main.py  --dataset DATASET --seed SEED --adv PGD --target TARGET
```

For provable training use the following commands:
```
$ (myenv) python3 main.py --dataset DATASET --seed SEED --cold_start --adv box
$ (myenv) python3 main.py --dataset DATASET --seed SEED --adv box --target TARGET 
```
The first command does the warm-up as explained in Appendix D and the second command performs the actual training after the warm-up training, i.e., they need to be performed sequentially. 
The models will be stored in the folder `models`.

### Certified & Adversarial Robustness
To compute the certified and adversarial robustness of any model use the following commands (e.g. to reproduce Table 1):
```
$ (myenv) python3 adversarial_exp.py --dataset DATASET --seed SEED --adv ADV --target TARGET --samples 1000 --pre_trained
$ (myenv) python3 gains_exp.py --dataset DATASET --seed SEED --adv ADV --target TARGET --samples 1000 --pre_trained
```
Note that to automatically select the right model the following arguments have to be provided:
For standard trained models the `--adv` and `--target` need to be omitted. 
For adversarially trained models set `ADV=PGD`. 
For certifiably trained models use `ADV=box` and set `TARGET` as above (final perturbation magnitude).
When using the pre-trained models, valid `SEED` are 12345, 12346, 12347 and valid `TARGET` are 0.11 and 0.22 for MNIST and 0.11 and 0.16 for F-MNIST.
In the robustness evaluations, we always consider the perturbations magnitudes [0.1,0.15,0.2] for the MNIST dataset, and [0.1,0.15] for the F-MNIST dataset. 

If you have trained the models yourselves, you can omit the argument `--pre_trained` and the models will be loaded from the `models/` folder. 

### Ablation Study Trajectory Attacks
In order to reproduce the results from Table 3 use the following command for the trajectory attacks:
```
$ (myenv) python3 step_attack.py --dataset DATASET --seed SEED --adv ADV --target TARGET --pre_trained 
```
See above for the right argument selection. 

### Ablation Study Bound Tightness
To reproduce our ablation on the tightness of the obtained bounds use the below commands to compute bounds 
and reference the jupyer notebook `Tightness.ipynb` for their analysis.
For convenience's sake, we have included precomputed bounds in`models/pre-trained/`, allowing the jupyter notebook to be used directly.
```
$ (myenv) python3 bounds.py --dataset DATASET --seed SEED --adv ADV --target TARGET --samples 1000 --pre_trained --eps EPS --mode MODE
```
Note that valid inputs for the `MODE` argument are [PGD,GAINS,GAINS-Box,GAINS-Linear] and `EPS` is a choosable perturbation level.
For the remaining arguments see above for the right selection. 

## Time-Series Forecasting
Please find below the commands to train and certify time-series forecasting models. 
For convenience's sake, we have included trained models in `models/pre-trained/`, corresponding to Table 2.

### Training Time-Series Forecasting
In order to use standard training for the latent ODE architecture on the PhysioNet dataset use the command:
```
$ (myenv) python3 latent_main.py  --extrap --run_backwards  --seed SEED --data_mode DATAMODE 
```
Note, that valid inputs for the `DATAMODE` argument are  1, 2 or 3, corresponding to omitting 6h, 12h, and 24h of data, respectively, as described in Appendix E.

For provable training with perturbations of magnitude `TARGET` use the command:

```
$ (myenv) python3 latent_prov.py --extrap --run_backwards --adv box  --target TARGET  --seed SEED  --data_mode DATAMODE
```
### Certified & Adversarial Robustness
Use the following commands to compute the empirical and certified robustness, for time-series forecasting models, e.g., tor reproduce Table 2:
```
$ (myenv) python3 latent_attack_MAE.py  --run_backwards --extrap   --seed SEED --adv ADV  --target TARGET  --data_mode DATAMODE --samples 400 --Nu 0.1 --Delta 0.01 --pre_trained  
$ (myenv) python3 latent_gains.py  --run_backwards --extrap   --seed SEED --adv ADV  --target TARGET  --data_mode DATAMODE --samples 400 --Nu 0.1 --Delta 0.01 --pre_trained
```
Note that to automatically select the right model the following arguments have to be provided:
For standard trained models the `--adv` and `--target` need to be omitted. 
For certifiably trained models use `ADV=box` and set `TARGET` as above (final perturbation magnitude).
Note that certification of standard models is expected to fail as we encounter overflow errors.
Both certification and adversarial attacks consider the follwing perturbation magnitudes [0.05, 0.1, 0.2].
Note that the pre-trained models were trained on the seed 100, 101 and 102 for perturbation levels of either 0.1 or 0.2 as described in Appendix F.

If you have trained the models yourselves, you can omit the argument `--pre_trained` and the models will be loaded from the `models/trained/` folder. 

## LCAP
See the jupyter notebook `LCAP_abl.ipynb` for our ablation experiments on the LCAP.


Citing this work
---------------------

If you find this work useful for your research, please cite it as:
```
@inproceedings{
    zeqiri2023efficient,
    title={Efficient Certified Training and Robustness Verification of Neural {ODE}s},
    author={Mustafa Zeqiri and Mark Niklas M{\"{u}}ller and Marc Fischer and Martin Vechev},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=KyoVpYvWWnK}
}
```

Contributors
------------

* [Mustafa Zeqiri](https://scholar.google.com/citations?user=TPp00-gAAAAJ)
* [Mark Müller](https://www.sri.inf.ethz.ch/people/mark)
* [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
* [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

License and Copyright
---------------------

* Copyright (c) 2022 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the MIT License (see LICENSE.txt)





