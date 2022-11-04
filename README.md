# PyRAMID-CT
A Python code to Remove Motion Artifacts In Dynamic Computed Tomography. 

- [Current features](#current-features)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Cite](#cite)

# Current features

- Piecewise Linear Interpolation (PLI) in time to augment time resolution by factor a of M and reduce motion artifact in CT.
- Use parallel CT forward and backward projectors from the [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) (PyRAMID-CT currently only supports parallel beam geometries).
- Calculations on single or double precision, on GPU or CPU.
- 3D+time regularization using Total Variation (TV) from [PyTV-4D](https://github.com/eboigne/PyTV-4D).
- Two algorithms available: sub-gradient descent, and [proximal algorithm from Chambolle and Pock](https://doi.org/10.1007/s10851-010-0251-1) (recommended).
- Calculations accelerated on GPU using PyTorch CUDA implementation.

# Installation

### Conda [Recommended]

The recommended setup is to use a new conda environment with python 3.7. Use the below one-line install that takes care of dependencies (if the dependency solve is slow with conda, checkout [mamba](https://mamba.readthedocs.io/)): 

`conda install -c conda-forge -c pytorch -c astra-toolbox/label/dev -c eboigne pyramid_ct`

Once installed, you can run some basic tests in python:

```python
import pyramid_ct

pyramid_ct.run_tests()
```

If you're having trouble with PyTorch running on GPU: install PyTorch first manually following the guidelines [on the official website](https://pytorch.org/). Make sure that a GPU version of PyTorch is installed. The installation of PyTorch should be similar to:

`conda install -c pytorch cudatoolkit=11.3 pytorch`

### Manual installation
PyRAMID-CT can also be installed manually by cloning the repo locally and installing with (dependencies need to be set properly):

`python setup.py install`

# Getting started

### Quick reconstruction example

```python
import pyramid_ct

# Load an example static phantom (without any motion)
sino_static = pyramid_ct.utils.example_moving_phantom()

# Run the Chambolle-Pock algorithm for 200 iterations
case = pyramid_ct.Case()
case.set_sinogram_data(sino_static)
case.nb_it = 200
case.run()
```

### Changing input parameters
See the [getting started Jupyter notebook](https://github.com/eboigne/PyRAMID-CT/blob/main/examples/a_getting_started.ipynb) in the examples folder. 

### Reading output data
The reconstructed data is written in images slices written in .tif format. [Fiji](https://imagej.net/software/fiji/) is recommended for post-processing the reconstructed CT datasets. These .tif stacks can be easily read in Fiji by drag and dropping the stack folder.

# Cite
Please refer to the following article in your publications if you use PyRAMID-CT for your research:
```
@article{boigne2022towards,
author={Boign{\'e}, Emeric and Parkinson, Dilworth Y. and Ihme, Matthias},
journal={{IEEE Transactions on Computational Imaging}},
title={{Towards data-informed motion artifact reduction in quantitative CT using piecewise linear interpolation}},
year={2022},
volume={8},
pages={917-932},
doi={10.1109/TCI.2022.3215096}
}
```

# License

PyRAMID-CT is open source under the GPLv3 license.

