# Exchange-Correlation template 
import pyksolver.xc_functionals.heg as HEG
import numpy as np

class LdaFunctional: 
    def __init__(self, xc_type) -> None:
        self.potxcR=[]
        self.potxcG=[]
        self.xc_type=xc_type
        #### 
    def update_xc(self,system):        
        nx=np.sqrt(system.densR.real**2)
        potential=-(3./np.pi)**(1./3.)*nx**(1./3.) +HEG.p_correlation_PZ(nx)  ##the potential in the direct space 
        ##
        self.potxcR=potential
        self.potxcG=system.myifft_dens_v(potential)