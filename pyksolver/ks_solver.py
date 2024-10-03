import numpy as np
from joblib import Parallel, delayed, cpu_count
from scipy.sparse.linalg import eigsh

number_of_cpu= cpu_count() 


class KsSolver:
    def __init__(self,system) -> None:
        ### outputs 
        self.bands_list=[]
        self.densR_history=[np.copy(system.densR)] 
        self.potxcR_history=[] 
    #################
    """## Kinetic energy """
    @staticmethod
    def get_kinetic(k,G):
        k,G=np.array(k), np.array(G)
        absKGm=[]
        for g in G:
            ans=np.linalg.norm(k+g)**2/2
            absKGm.append(ans)
        absKGm=np.array(absKGm)
        T=np.diagflat(absKGm)
        return T
    @staticmethod
    def get_Ham_ik(K_ik, dict_vnl_ik, G_k_abinit_ik, matrixGk_dict_ik, potHartreeG,potxcG,V_ext):    
        vnlk= dict_vnl_ik 
        T=KsSolver.get_kinetic(K_ik,G_k_abinit_ik)
        H_k=T+np.array(vnlk).T
        V_HxcExt= potHartreeG[matrixGk_dict_ik] + potxcG[matrixGk_dict_ik] + V_ext[matrixGk_dict_ik]
        H_k+=V_HxcExt
        return H_k 
    
    @staticmethod
    def get_densk_KS(ik, dict_vnl_ik, matrixGk_dict_ik, potHartreeG, potxcG,system):
        H_k=KsSolver.get_Ham_ik(system.K[ik], dict_vnl_ik, system.G_k_abinit[ik], matrixGk_dict_ik, potHartreeG, potxcG, system.V_ext)
        #####
        bands=[] 
        ei_k, cim_k = np.linalg.eigh(H_k) #eigsh(H_k,system.n_occup*3, which='SA')   #np.linalg.eigh(H_k)  # 
        bands.append(ei_k[:system.n_occup*4])
        nrk_tot=0
        for j in range(system.n_occup):
            cr=1/np.sqrt(system.V_unitcell) *system.myfft_v_abinit(cim_k[:,j],ik)
            nr=2*np.conjugate(cr)*cr
            nrk_tot+=nr 
        return nrk_tot,bands 
    
    #@timeit
    def get_dens_parallel(self, system, xc_functional,Ham_instance,update_system=True,mixing=0.75,njobs=number_of_cpu): 
        # Constructing the Hamiltonian 
        self.potxcG=xc_functional.potxcG 
        self.potHartreeG = (system.densG * np.array(system.HP_nodens)) 
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