# FixMatch
This code integrates DIPS into the semi-supervised [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
It builds on top of the unofficial PyTorch implementation of this [repository](https://github.com/kekmodel/FixMatch-pytorch).
The repository also uses the CIFAR10N dataset and the [repository released by the authors](https://github.com/UCSC-REAL/cifar-10-100n).

### Setup
The requirements file ``requirements.txt`` describes the libraries required to run the code and can be used to create the virtual environment. 
Weight and biases (WandB) is used to log the results of the experiments. In order to log results with WandB, please provide a Wandb API key and an entity in the file ``wandb.yaml``.
Furthermore, the modified library Datagnosis can be installed from source by going to the folder ``external/Datagnosis`` and executing the following line: 
```shell
 pip install . 
```

### Example use
We provide a script to run DIPS+FixMatch on CIFAR10n, located at ``scripts/run_fixmatch_dips.sh``. 


## References
- [Cifar 10n repository](https://github.com/UCSC-REAL/cifar-10-100n)
- [Unoficial implementation of FixMatch repository](https://github.com/kekmodel/FixMatch-pytorch)
- [Datagnosis](https://github.com/vanderschaarlab/Datagnosis)


