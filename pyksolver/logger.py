import numpy as np 
import time  
from joblib import cpu_count 
from functools import wraps
number_of_cpu= cpu_count() 


class Logger:
    def __init__(self, output_file="log.txt"):
        self.output= output_file 
    
    def my_print(self,txt):
        with open("log_"+self.output+".txt", "a") as f:
            f.write(txt + "\n") 
        
    def timeit(self,func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            self.my_print(f'{func.__name__} Took {total_time/60:.4f} minutes')
            return result
        return timeit_wrapper  
        
    def reporting(self,n_it,system, ks_solver, xc_functional, nkpt_bands_print=1, save=True): 
        self.my_print(f"default_njob is {number_of_cpu}")
        self.my_print(f"# Iteration {n_it} diff_dens {np.mean(np.abs(ks_solver.densR_history[n_it]-ks_solver.densR_history[n_it-1])) } ")
        self.my_print('direct gap {} eV'.format((ks_solver.bands_list[0][system.n_occup]-ks_solver.bands_list[0][system.n_occup-1])*27.211) )
        #if xc_functional.xc_type.split('_')[0]=="connector": 
        #    self.my_print(f"connector n_0r {xc_functional.n_0r} ")  
        #    self.my_print(f"connector n_rrp {xc_functional.n_0r} ")
        #    self.my_print(f"connector sc {xc_functional.sc_con} ")          
        np.set_printoptions(precision=6)  
        for ik in range(nkpt_bands_print):
            self.my_print("ik="+str(ik)+" eigens "+str(ks_solver.bands_list[ik][:system.n_occup +1]) + str(" Ha"))
        self.my_print(f"############# Iteration {n_it} ####################### \n")
        if save: 
            np.savez(self.output, dens_history=ks_solver.densR_history, potxc_history=ks_solver.potxcR_history, bands_list=ks_solver.bands_list, 
                     ncon=xc_functional.ncon, n_0r=xc_functional.n_0r  )
        
    
 # try eighsh and sort
  
