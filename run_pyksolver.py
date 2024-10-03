import pyksolver
import numpy as np 
import pickle
from pyksolver.xc_functionals.lda import LdaFunctional


def main(): 
    #-----------------------------------------------------------------------------------------------------
    # nonlocal potenial 
    origin_dir= "./"
    a_l=np.array([10.263087]*3)
    input_dict={# Lattice parameters of the material in bohr 
                "a_l":a_l , 
                # change of basis from conventional to primitive cell 
                "MtR": np.array([[0,a_l[0]/2,a_l[0]/2],[a_l[0]/2, 0,a_l[0]/2],[a_l[0]/2, a_l[0]/2,0]]), 
                #number of occupied bands for semi-conductor and insulator 
                "n_occup": 4,
                # Grid of k points 
                "kx": np.arange(-2,4)*1/6 , 
                # Cut-off energ
                "Ecut":12.5,  
                "path_VextR": origin_dir+"VPS_loc.dat", 
                "path_vnl": origin_dir+'vnl.dat', 
                # intial guess for  density 
                "densR": np.ones((24**3))*8  /10.263087**3/4, 
                "output": "test", 
                "dict_vnl": True, } 
    #-----------------------------------------------------------------------------------------------------

    input_dict["output"]="lda"
    mylogger=pyksolver.logger.Logger(input_dict["output"])


    

    si=pyksolver.system.System(input_dict)
    si.config=mylogger.timeit(si.config)
    si.config() 
    
    ham_instance = pyksolver.hamiltonian.HamiltonianTemplate(si)
    with open("./Si_ham_template","wb") as f :
         pickle.dump(ham_instance,f) 
         #ham_instance= pickle.load(f)

    xc_func = LdaFunctional(si)
    xc_func.update_xc=mylogger.timeit(xc_func.update_xc)

    ###
    ks_solv = pyksolver.ks_solver.KsSolver(si) 
    


    for n_it in range(1,999):
        xc_func.n_0r= si.densR
        xc_func.update_xc(si) 
        ks_solv.get_dens_parallel(si,xc_func,ham_instance, mixing=0.75, njobs=1)
        mylogger.reporting(n_it,si, ks_solv, xc_func)

if __name__=="__main__": 
    main()
    
