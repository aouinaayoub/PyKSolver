# PyKSolver

This repository contains a Python implementation for solving the Kohn-Sham equations using the plane wave method.

# Requirements

This code requires the following libraries to be installed:
- scipy
- numpy
- pickle
- joblib 
# Usage 
To try the code for Silicon:
- Clone this repository to your local machine:

```bash
git clone https://github.com/aouinaayoub/PyKSolver.git
``` 
- Download the non-local part of the pseudo-potential of Silicon from here: [vnl.dat](https://zenodo.org/record/7661254/files/vnl.tar.gz?download=1) and put it in the same directory. 
- Install the package using pip: 
```bash 
pip install .
```
- Then, execute the example `run_pyksolver.py` file:
```bash 
python run_pyksolver.py
``` 
# Output

The solver will generate an output file `lda.npz` which contains the following information for each iteration:
- The exchange-correlation potential
- The density
- The bands

