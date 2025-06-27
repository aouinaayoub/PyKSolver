# Exchange-Correlation template 
import pyksolver.xc_functionals.heg as HEG
import numpy as np

class LdaFunctional: 
    """
    Local Density Approximation (LDA) exchange-correlation functional.
    """
    def __init__(self, xc_type: str) -> None:
        """
        Initialize the LDA functional with the given type.
        """
        self.potxcR=[]
        self.potxcG=[]
        self.xc_type=xc_type
        #### 
    def update_xc(self,system) -> None:        
        """
        Update the exchange-correlation potential in real and reciprocal space.
        """
        nx = np.abs(system.densR)
        potential=-(3./np.pi)**(1./3.)*nx**(1./3.) +HEG.p_correlation_PZ(nx)  ##the potential in the direct space 
        ##
        self.potxcR=potential
        self.potxcG=system.myifft_dens_v(potential)