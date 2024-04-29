import os
import numpy as np 

#-------------------------------------------------------------------------------------------------------------------------#
# intial guess for  density
densR= np.ones((24**3))*8  /10.263087**3/4 #np.genfromtxt("../dens_ref_lda.dat") #0.9* np.genfromtxt("dens_ref_lda.dat") 

# choice of the xc functional 
xc_type="LDA"

# choose output file name   
output_filename=xc_type 
# continue from last run 
continue_from_last_run=0 
last_run="invlda.npz" 

#-------------------------------------------------------------------------------------------------------------------------#

# parameters of the material in bohr 
a_l=np.array([10.263087]*3) 

# Volume of unit cell 
V_unitcell= np.dot(np.cross([a_l[0],0,0],[0,a_l[1],0]) , [0, 0,a_l[2]])/4 

# change of basis from conventional to primitive cell 
MtR=np.array([[0,a_l[0]/2,a_l[0]/2],[a_l[0]/2, 0,a_l[0]/2],[a_l[0]/2, a_l[0]/2,0]])

# number of occupied bands for semi-conductor and insulator
n_occup=4

# Grid of k 
kx=np.arange(-2,4)*1/6  #  sampling of K points 

# Cut-off energy 
Ecut=12.5

# nonlocal and local potenial directory 
origin_dir= "./"

# local potential  
V_extR =np.genfromtxt(origin_dir+"VPS_loc.dat")

# grid in real space, it depends on the cutoff 
n_rgrid1,n_rgrid2,n_rgrid3 = [round(len(V_extR)**(1/3))]*3


# wrap up
input_dict={"a_l":a_l, "V_unitcell":V_unitcell, "MtR":MtR, "n_occup":n_occup, "kx":kx, "Ecut":Ecut, 
            "path_VextR": origin_dir+"VPS_loc.dat", "path_vnl": origin_dir+'vnl.dat', 
            "densR": densR, "output": output_filename, "xc_type":xc_type, "dict_vnl": True, "continue_from_last_run":continue_from_last_run, "last_run":last_run} 
