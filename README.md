# PyKSolver

This repository contains a Python implementation for solving the Kohn-Sham equations using the plane wave method.

PyKSolver consists of three main files:
- `input_mat.py`: Defines the material parameters, chooses the exchange-correlation functional, and specifies the pseudo-potential file.
- `utils.py`: Defines some useful functions for constructing the Hamiltonian.
- `pyksolver.py`: Solves the Kohn-Sham equations and yields a file which contains the exchange-correlation potential, density, and bands in each iteration.

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
- Then, execute the `pyksolver.py` file:
```bash 
python pyksolver.py
``` 
# Output

The solver will generate an output file `lda.npz` which contains the following information for each iteration:
- The exchange-correlation potential
- The density
- The bands

# Acknowledgments

The author would like to acknowledge Dr. Matteo Gatti and Dr. Lucia Reining for their guidance and support.

# License

This code is licensed under the MIT License.

# Citation 
 If you find this code helpful, please cite: 
> "[Aouina, A. (2022). A novel shortcut for computational materials design (Doctoral dissertation, Institut Polytechnique de Paris).](https://hal-cnrs.archives-ouvertes.fr/X-LSI/tel-03662872v1)" 
