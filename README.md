# Transient Oscillation Dynamics (TOD) Toolbox for Python v1.0 - Prerau Laboratory ([sleepEEG.org](https://prerau.bwh.harvard.edu/))

#### This repository contains the updated and optimized toolbox code for extracting time-frequency peaks from EEG data and creating slow-oscillation power and phase histograms with a Python package. 

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
    ``` conda activate pyTOD ```

    _You may also install pyTOD in an existing environment by skipping this step._

- ### Install pyTOD as a package in editable mode
    ``` cd <repo root directory with setup.py> ```\
    ``` pip install -e . ```

- ### Configure IDEs to use the conda environment
    [How to: Configure an existing conda environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html)


## Citations
### Paper and Toolbox
Please cite the following paper when using this package: 
> Patrick A Stokes, Preetish Rath, Thomas Possidente, Mingjian He, Shaun Purcell, Dara S Manoach, Robert Stickgold, Michael J Prerau, Transient Oscillation Dynamics During Sleep Provide a Robust Basis for Electroencephalographic Phenotyping and Biomarker Identification, Sleep, 2022;, zsac223, https://doi.org/10.1093/sleep/zsac223

The toolbox can be referred to in the text as:
> Transient Oscillation Dynamics (TOD) Toolbox v1.0 (sleepEEG.org/transient-oscillations-dynamics)

The paper is available open access at https://doi.org/10.1093/sleep/zsac223
