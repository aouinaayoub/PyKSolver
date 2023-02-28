import pickle
import numpy as np 
from datetime import datetime
from input_mat import * 
from utils import *
from joblib import Parallel, delayed, cpu_count 

my_print("start config \n")
rlist, rlist_int, K, K_int, G_k_abinit_int, G_k_abinit , G_dens, G_dens_int, G_dic, gnorm, HP_nodens, V_ext,vHartree, vget_xc, vget_Vext, G_k_abinit_lens=config()

def get_matrixGki(ik):
    with open(matrix_dir+"/matrix_ik_{}.pkl".format(ik+1),"rb") as matfile:
        matrixGki=pickle.load(matfile)
        matfile.close()
    return matrixGki[0]
matrixGk_dict= { ik:get_matrixGki(ik) for ik in range(len(K)) } 
my_print("config done \n")


if continue_from_last_run: 
    dens_history=list(np.load(last_run)["dens_history"])
    potxc_history=list(np.load(last_run)["potxc_history"])
    densR, potxcR=dens_history[-1], potxc_history[-1]
else:
    dens_history=[densR]
    potxc_history=[]

bands_history=[]
densG= myifft_dens_v(densR,G_dens_int,rlist)
number_of_cpu =  cpu_count() 
diff_dens=999
diff_l=[]
n_iter=0 

my_print("njob is "+str(number_of_cpu ) +"\n")
my_print("Approx for xc potential is "+str(xc_type) +"\n")
while (diff_dens>2e-24):

    # xc potential     
    potxcR=get_vxc_potentials(xc_type,densG,densR,gnorm,rlist,rlist_int,G_dens,G_dens_int)
    potxcG= myifft_dens_v(potxcR, G_dens_int,rlist)
    start_time=datetime.now()
    ### 
    delayed_funcs = [delayed(get_densk_KS)(ik, vHartree, vget_xc, vget_Vext,G_k_abinit_lens,G_k_abinit[ik],G_k_abinit_int[ik],matrixGk_dict[ik], K,densG, HP_nodens,potxcG,V_ext,rlist) for ik in range(len(K))]
    parallel_pool = Parallel(n_jobs=number_of_cpu//2 )   # to reduce the required memory ? 
    out= parallel_pool(delayed_funcs)
    ####
    end_time=datetime.now()
    
    sumorbitk= np.sum(  np.array(out, dtype=object)[:,0], axis=0)/len(K)
    diff_dens=np.sqrt(np.mean((sumorbitk-densR)**2))
    diff_l.append(diff_dens)
    densR=sumorbitk.real 
    n_iter+=1 
    my_print("-----------------\n")
    my_print("# Iteration " + str(n_iter)+"\n")
    my_print('finished in '+str(end_time-start_time)[:7]+"\n")
    my_print('diff_dens {} \n'.format(diff_dens.real))
    bands_list=[]
    for ik in range( len(K)) :
        bands_ik= out[ik][1][0]
        if ik<3 : 
            np.set_printoptions(precision=6)
            my_print("ik="+str(ik)+" eigens "+str(bands_ik)+"\n")
        bands_list.append(bands_ik)
    bands_history.append(bands_list)
    
    
    # mixing 
    densR=(densR+dens_history[-1])/2
    dens_history.append(densR)
    densG=myifft_dens_v(densR,G_dens_int,rlist)
    
    potxc_history.append(potxcR)
    
    path_outdens=output_filename+".npz"
    np.savez(path_outdens ,potxc_history=potxc_history ,dens_history=dens_history, K=K, diff_l=diff_l,bands_history=bands_history)    
    
