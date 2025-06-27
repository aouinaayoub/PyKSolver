import numpy as np
from joblib import Parallel, delayed, cpu_count
from scipy.sparse.linalg import eigsh
from typing import Any, List, Tuple

number_of_cpu = cpu_count()


class KsSolver:
    """
    Kohn-Sham Solver for plane-wave DFT calculations.
    Handles the self-consistent field (SCF) loop for updating densities and solving the KS equations.
    """
    def __init__(self, system: Any) -> None:
        """
        Initialize the Kohn-Sham solver.
        Args:
            system: The system object containing grid, density, and structure information.
        """
        # Outputs
        self.bands_list: List[List[float]] = []
        self.densR_history: List[np.ndarray] = [np.copy(system.densR)]
        self.potxcR_history: List[np.ndarray] = []
    #################
    """## Kinetic energy """
    @staticmethod
    def get_kinetic(k: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Construct the kinetic energy matrix for a given k-point and G-vectors.
        Args:
            k: k-point vector.
            G: Array of G-vectors.
        Returns:
            Kinetic energy matrix (diagonal).
        """
        k, G = np.array(k), np.array(G)
        absKGm = [np.linalg.norm(k + g) ** 2 / 2 for g in G]
        T = np.diagflat(absKGm)
        return T
    @staticmethod
    def get_Ham_ik(
        K_ik: np.ndarray,
        dict_vnl_ik: np.ndarray,
        G_k_abinit_ik: np.ndarray,
        matrixGk_dict_ik: np.ndarray,
        potHartreeG: np.ndarray,
        potxcG: np.ndarray,
        V_ext: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the Hamiltonian matrix for a given k-point.
        """
        vnlk = dict_vnl_ik 
        T = KsSolver.get_kinetic(K_ik, G_k_abinit_ik)
        H_k = T + np.array(vnlk).T
        V_HxcExt = (
            potHartreeG[matrixGk_dict_ik]
            + potxcG[matrixGk_dict_ik]
            + V_ext[matrixGk_dict_ik]
        )
        H_k += V_HxcExt
        return H_k 
    
    @staticmethod
    def get_densk_KS(
        ik: int,
        dict_vnl_ik: np.ndarray,
        matrixGk_dict_ik: np.ndarray,
        potHartreeG: np.ndarray,
        potxcG: np.ndarray,
        system: Any,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Solve the KS equations for a single k-point and return the density and bands.
        """
        H_k = KsSolver.get_Ham_ik(
            system.K[ik],
            dict_vnl_ik,
            system.G_k_abinit[ik],
            matrixGk_dict_ik,
            potHartreeG,
            potxcG,
            system.V_ext,
        )
        bands: List[np.ndarray] = []
        ei_k, cim_k = np.linalg.eigh(H_k)
        bands.append(ei_k[: system.n_occup * 4])
        # Ensure nrk_tot is always an ndarray
        cr = 1 / np.sqrt(system.V_unitcell) * system.myfft_v_abinit(cim_k[:, 0], ik)
        nrk_tot = np.zeros_like(cr)
        for j in range(system.n_occup):
            cr = 1 / np.sqrt(system.V_unitcell) * system.myfft_v_abinit(cim_k[:, j], ik)
            nr = 2 * np.conjugate(cr) * cr
            nrk_tot += nr
        return nrk_tot, bands
    
    #@timeit
    def get_dens_parallel(
        self,
        system: Any,
        xc_functional: Any,
        Ham_instance: Any,
        update_system: bool = True,
        mixing: float = 0.75,
        njobs: int = number_of_cpu,
    ) -> None:
        """
        Update the density in parallel over k-points using the current Hamiltonian and XC functional.
        Args:
            system: The system object.
            xc_functional: The exchange-correlation functional object.
            Ham_instance: The Hamiltonian object.
            update_system: Whether to update the system's density.
            mixing: Mixing parameter for density update.
            njobs: Number of parallel jobs.
        """
        # Constructing the Hamiltonian 
        self.potxcG = xc_functional.potxcG 
        self.potHartreeG = system.densG * np.array(system.HP_nodens) 
        self.potxcR_history.append( xc_functional.potxcR ) 
        ##
        delayed_funcs = [delayed(self.get_densk_KS)(ik, Ham_instance.dict_vnl[ik], Ham_instance.matrixGk_dict[ik], self.potHartreeG, self.potxcG,system) for ik in range(len(system.K))]
        parallel_pool = Parallel(n_jobs=njobs, backend="threading" )  
        out= parallel_pool(delayed_funcs)
        ####        
        sumorbitk= np.sum(  np.array(out, dtype=object)[:,0], axis=0)/len(system.K)
        self.bands_list=[out[ik][1][0] for ik in range(len(system.K))]
        if update_system : 
            system.densR = mixing* sumorbitk.real + (1-mixing)*self.densR_history[-1]
            self.densR_history.append( system.densR )
            system.densG=system.myifft_dens_v(system.densR)