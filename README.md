# Mathematical signal and image processing library

This **m**athematical **s**ignal and **i**mage **p**rocessing **lib**rary (msiplib) implements functions and methods useful for processing with signals and images, like denoising and segmentation. The tools are developed in the group of Prof. Benjamin Berkels at AICES, RWTH Aachen University.

In particular, this package contains an implementation of the method proposed in
the paper:

[1] Jan-Christopher Cohrs, Chandrajit Bajaj and Benjamin Berkels. A distribution-dependent Mumford-Shah model for unsupervised hyperspectral image segmentation. *IEEE Transactions on Geoscience and Remote Sensing*, 60:1--21, December 2022, Art no. 5545121. [[DOI](https://doi.org/10.1109/TGRS.2022.3227061) | [arXiv](https://arxiv.org/abs/2203.15058)]

We appreciate any feedback on your experience with our methods. We would also appreciate if you cite the above mentioned paper when you use the software in your work. In case you encounter any problems when using this software, please do not hesitate to contact us: <berkels@aices.rwth-aachen.de>

## Installation
The code can be made importable by creating a local copy of it and installing it as a package with

```
$ conda develop <local path to repo>
```
or
```
$ pip install <local path to repo>
```
In addition, a yaml-file is provided to directly create a conda environment containing the necessary packages to run the code. In order to create such an environment, please run
```
$ conda env create -f env-msiplib.yml
```
from the root directory of the repository. This will also automatically install `msiplib` as a package in the created environment.

## Usage
Examples of simple segmentation and denoising code for grayscale and RGB images can be found in `examples`.
A script to run the code belonging to the published paper [1] can be found in `tools/hsi_segmentation`. Please note that in order to download the test data, the openssl package of version 1.1.1 is needed.

## Documentation
A documentation can be automatically generated by following the steps described in `docs/README`.
