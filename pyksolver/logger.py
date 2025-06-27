import numpy as np 
import time  
from joblib import cpu_count 
from functools import wraps
from typing import Callable

number_of_cpu= cpu_count() 


class Logger:
    """
    Logger class for timing, printing, and reporting simulation results.
    """
    def __init__(self, output_file: str = "log"):
        self.output = output_file

    def my_print(self, txt: str) -> None:
        """Append a line of text to the log file."""
        with open(f"{self.output}.txt", "a") as f:
            f.write(txt + "\n")

    def timeit(self, func: Callable) -> Callable:
        """Decorator to time a function and log the duration in minutes."""
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            self.my_print(f'{func.__name__} took {total_time/60:.4f} minutes')
            return result
        return timeit_wrapper

    def reporting(self, n_it: int, system, ks_solver, xc_functional, nkpt_bands_print: int = 1, save: bool = True) -> None:
        """
        Log iteration details, band gaps, and optionally save simulation data.
        """
        self.my_print(f"default_njob is {number_of_cpu}")
        diff_dens = np.mean(np.abs(ks_solver.densR_history[n_it] - ks_solver.densR_history[n_it - 1]))
        self.my_print(f"# Iteration {n_it} diff_dens {diff_dens}")
        direct_gap = (ks_solver.bands_list[0][system.n_occup] - ks_solver.bands_list[0][system.n_occup - 1]) * 27.211
        self.my_print(f'direct gap {direct_gap} eV')
        np.set_printoptions(precision=6)
        for ik in range(nkpt_bands_print):
            eigens = ks_solver.bands_list[ik][:system.n_occup + 1]
            self.my_print(f"ik={ik} eigens {eigens} Ha")
        self.my_print(f"############# Iteration {n_it} ####################### \n")
        if save:
            savez_kwargs = {
                'dens_history': ks_solver.densR_history,
                'potxc_history': ks_solver.potxcR_history,
                'bands_list': ks_solver.bands_list
            }
            ncon = getattr(xc_functional, 'ncon', None)
            n_0r = getattr(xc_functional, 'n_0r', None)
            if ncon is not None:
                savez_kwargs['ncon'] = ncon
            if n_0r is not None:
                savez_kwargs['n_0r'] = n_0r
            np.savez(self.output, **savez_kwargs)
        
    
 # try eighsh and sort
  
