<p align="center">
<img src=https://user-images.githubusercontent.com/78376124/214062562-4f8fc73b-5a0a-4cf7-b219-9d0de101528d.png>
</p>

## DYNAM_O: The Dynamic Oscillation Toolbox for Python - Prerau Laboratory ([sleepEEG.org](https://prerau.bwh.harvard.edu/))

#### This repository contains the updated and optimized PYTHON package code for extracting time-frequency peaks from EEG data and creating slow-oscillation power and phase histograms. 

## Citations
### Paper and Toolbox
Please cite the following paper when using this package: 
> Patrick A Stokes, Preetish Rath, Thomas Possidente, Mingjian He, Shaun Purcell, Dara S Manoach, Robert Stickgold, Michael J Prerau, Transient Oscillation Dynamics During Sleep Provide a Robust Basis for Electroencephalographic Phenotyping and Biomarker Identification, Sleep, 2022;, zsac223, https://doi.org/10.1093/sleep/zsac223

The toolbox can be referred to in the text as:
> Prerau Lab's Dynamic Oscillation Toolbox (DYNAM-O) v1.0 (sleepEEG.org/)

The paper is available open access at https://doi.org/10.1093/sleep/zsac223

--- 
## Install

- ### Fork this repo to personal git
    [How to: GitHub fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)    

- ### Clone forked copy to local computer
    ``` git clone <forked copy ssh url> ```

- ### Install conda
    [Recommended conda distribution: miniconda](https://docs.conda.io/en/latest/miniconda.html)

    _Apple silicon Mac: choose conda native to the ARM64 architecture instead of Intel x86_

- ### Create a new conda environment
    ``` conda install mamba -n base -c conda-forge ```\
    ``` cd <repo root directory with environment.yml> ```\
    ``` mamba env create -f environment.yml ```\
    ``` conda activate dynam_o ```

    _You may also install dynam_o in an existing environment by skipping this step._

- ### Install dynam_o as a package in editable mode
    ``` cd <repo root directory with setup.py> ```\
    ``` pip install -e . ```

- ### Configure IDEs to use the conda environment
    [How to: Configure an existing conda environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html)
--- 
## Tutorial
A full description of the toolbox and tutorial [can be found on the Prerau Lab site](https://prerau.bwh.harvard.edu/DYNAM-O/)
