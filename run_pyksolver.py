import numpy as np
import pickle
from pyksolver.logger import Logger
from pyksolver.system import System
from pyksolver.hamiltonian import HamiltonianTemplate
from pyksolver.ks_solver import KsSolver
from pyksolver.xc_functionals.lda import LdaFunctional

def main():
    """
    Main routine to run the PyKSolver workflow for a sample system.
    """
    #-----------------------------------------------------------------------------------------------------
    # Nonlocal potential and system setup
    origin_dir = "./"
    a_l = np.array([10.263087] * 3)
    input_dict = {
        "a_l": a_l,
        "MtR": np.array([
            [0, a_l[0] / 2, a_l[0] / 2],
            [a_l[0] / 2, 0, a_l[0] / 2],
            [a_l[0] / 2, a_l[0] / 2, 0]
        ]),
        "n_occup": 4,
        "kx": np.arange(-2, 4) * 1 / 6,
        "Ecut": 12.5,
        "path_VextR": origin_dir + "VPS_loc.dat",
        "path_vnl": origin_dir + 'vnl.dat',
        "densR": np.ones((24 ** 3)) * 8 / 10.263087 ** 3 / 4,
        "output": "test",
        "dict_vnl": True,
    }
    #-----------------------------------------------------------------------------------------------------

    input_dict["output"] = "lda"
    mylogger = Logger(input_dict["output"])

    si = System(input_dict)
    si.config = mylogger.timeit(si.config)
    si.config()

    ham_instance = HamiltonianTemplate(si)
    with open("./Si_ham_template", "wb") as f:
        pickle.dump(ham_instance, f)
        # ham_instance = pickle.load(f)

    xc_func = LdaFunctional(xc_type="LDA")
    xc_func.update_xc = mylogger.timeit(xc_func.update_xc)

    ks_solv = KsSolver(si)

    for n_it in range(1, 999):
        # If you need to store n_0r, add it to xc_func 
        # xc_func.n_0r = si.densR
        xc_func.update_xc(si)
        ks_solv.get_dens_parallel(si, xc_func, ham_instance, mixing=0.75, njobs=1)
        mylogger.reporting(n_it, si, ks_solv, xc_func)

if __name__ == "__main__":
    main()
    
