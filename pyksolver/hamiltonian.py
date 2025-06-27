import sys
import numpy as np
import logging

# Set up logging for warnings
logging.basicConfig(filename="warning_G_dens.txt", level=logging.WARNING, format='%(message)s')

MISSING_G_INDEX = -999

class HamiltonianTemplate:
    def __init__(self, Sys, load_vnl=True) -> None:
        """
        Construct the matrices for each G_k and optionally load vnl matrices.
        """
        def get_g_density_index(g_vector):
            g_vector_list = list(g_vector)
            index = Sys.G_dic.get(str(g_vector_list), MISSING_G_INDEX)
            if index == MISSING_G_INDEX:
                logging.warning("warning missing G in G_dens: %s", g_vector_list)
            return index

        self.matrixGk_dict = {}
        for i, G_k in enumerate(Sys.G_k_abinit_int):
            Gki = np.array(G_k)
            diff_matrix = Gki[:, None] - Gki[None, :]
            matrixGk = np.apply_along_axis(get_g_density_index, -1, diff_matrix)
            self.matrixGk_dict[i] = matrixGk

        if load_vnl:
            def get_vnl_k_opt(ik, Sys):
                len_Gk = Sys.G_k_abinit_lens[ik]
                n_skip = ik + sum(Sys.G_k_abinit_lens[:ik])
                vnl = np.genfromtxt(Sys.path_vnl, comments="ikpt", max_rows=len_Gk, skip_header=n_skip)
                res = vnl[:, 0::2] + 1j * vnl[:, 1::2]
                return res

            self.dict_vnl = {ik: get_vnl_k_opt(ik, Sys) for ik in range(len(Sys.K))}