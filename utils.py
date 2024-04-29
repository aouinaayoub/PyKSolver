import numpy as np 
from scipy.sparse.linalg import eigsh
import sys, time  
from joblib import Parallel, delayed, cpu_count 
from functools import wraps

number_of_cpu= cpu_count() 

# Timing        
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        my_print(f'{func.__name__} Took {total_time/60:.4f} minutes')
        return result
    return timeit_wrapper  
####  
class system: 
    def __init__(self,input_dict) -> None:
        self.a_l=input_dict["a_l"]
        self.MtR=input_dict["MtR"]
        self.n_occup=input_dict["n_occup"]
        self.V_unitcell= input_dict["V_unitcell"] 
        self.kx=input_dict["kx"]
        self.Ecut=input_dict["Ecut"]
        # local potential  
        self.V_extR =np.genfromtxt(input_dict["path_VextR"]) 
        # grid in real space, it depends on the cutoff 
        self.n_rgrid1,self.n_rgrid2,self.n_rgrid3 = [round(len(self.V_extR)**(1/3))]*3 
        self.rlist, self.rlist_int, self.K, self.K_int, self.G_k_abinit_int, self.G_k_abinit , self.G_dens, self.G_dens_int, self.G_dic, self.gnorm, self.HP_nodens, self.G_k_abinit_lens=self.config()
        # FFT the psp once for all 
        self.V_ext=self.myifft_dens_v(self.V_extR) # should be in KS solver or ham constructor 
        # The density (will be updated)
        self.densR=input_dict["densR"]   
        self.densG= self.myifft_dens_v(self.densR) 
        self.path_vnl= input_dict["path_vnl"]            
    ##################
    # Useful functions
    ##################
        """## Parameter of the material """
    @timeit
    def config(self):
        """ This function is to generate the G and k vectors, and the the Hartree potential part without the density """
        rlist, rlist_int=[],[] 
        for z in range(0,self.n_rgrid3): 
            for y in range(0,self.n_rgrid2): 
                for x in range(0,self.n_rgrid1):
                    xr, yr ,zr=x/self.n_rgrid1, y/self.n_rgrid2, z/self.n_rgrid3
                    rlist.append(np.dot(self.MtR.T,[xr,yr,zr]))
                    rlist_int.append([x,y,z])  

    #Change of basis 
        Mt=2*np.pi*(np.linalg.inv(self.MtR)) 
    #Mt_int=np.array([[-1,1,1],[1, -1,1],[1, 1,-1]])
        kx2=np.roll(self.kx,1+int(len(self.kx)/2))   
        K=[]
        K_int=[]
        for k in kx2:
            for j in kx2 : 
                for i in kx2 :
                    hold=np.dot(Mt,[i,j,k])
                    K.append(hold)
                    K_int.append([i,j,k])
        K=np.array(K)

        qmax=int(np.sqrt(2*self.Ecut)/np.linalg.norm(Mt[:,0]))
        qlist_abinit=np.roll(np.arange(-qmax-3,qmax+4),qmax+4) 

    # Grid of G accoridng to Ecut according to abinit
        G_k_abinit_int, G_k_abinit=[],[]
        for q in K: 
            G,G_int=[],[]
            for k in qlist_abinit: 
                for j in qlist_abinit: 
                    for i in qlist_abinit: 
                        g_orth=np.dot(Mt,[i,j,k])
                        if np.linalg.norm(q+g_orth)**2/2<self.Ecut : 
                            x,y,z=g_orth[0],g_orth[1],g_orth[2] 
                            G.append([x,y,z]),G_int.append([i,j,k])                 
            G_k_abinit.append(G), G_k_abinit_int.append(G_int)

    # Grid of G-density 
        G_dens,  G_dens_int=[],[]
        #qmax2=int(np.sqrt(8*Ecut)/b)
        qmax2_x=int(self.n_rgrid1/2)                        
        qmax2_y=int(self.n_rgrid2/2) 
        qmax2_z=int(self.n_rgrid3/2)
        G_dic={}
        count=0
        for i in range(-qmax2_x,qmax2_x): 
            for j in range(-qmax2_y,qmax2_y): 
                for k in range(-qmax2_z,qmax2_z): 
                    g_orth=np.dot(Mt,[i,j,k])
                    if np.linalg.norm(g_orth)**2/2<4*self.Ecut  : 
                        x=g_orth[0]
                        y=g_orth[1]
                        z=g_orth[2]
                        G_dens.append([x,y,z])
                        G_dic[str([i,j,k])]=count
                        G_dens_int.append([i,j,k])
                        count+=1
        # useful for connector 
        gnorm=np.linalg.norm(G_dens,axis=1) 

        HP_nodens=[]
        for v in G_dens:
            if np.linalg.norm(v)==0:
                HP_nodens.append(0)
            else: HP_nodens.append( 4*np.pi * 1/np.linalg.norm(v)**2)
        
        G_k_abinit_lens=[len(G_k_abinit[ik]) for ik in range(len(K)) ]
        return rlist, rlist_int, K, K_int, G_k_abinit_int, G_k_abinit , G_dens, G_dens_int, G_dic, gnorm, HP_nodens, G_k_abinit_lens 
    
    #########
    # My FFTs 
    #########
    # redefine fourier transform 
    # direct FFT, using the corresponding G-set for each k
    def myfft_v_abinit(self,fG,ik):
        """ Compute the FFT of orbital-like function from G-space to Real space

        Args:
            fG (numpy array): function in G-space 
            ik (integer): index of k in the k list 

        Returns:
            numpy array: FFT of fG 
        """
        normalized_fG=np.zeros((self.n_rgrid1*self.n_rgrid2*self.n_rgrid3))+0J
        max_index1,max_index2,max_index3=int(self.n_rgrid1/2),int(self.n_rgrid2/2), int(self.n_rgrid3/2)
        for ifG in range(len(fG)): 
            g_inv=self.G_k_abinit_int[ik][ifG]
            ix, iy ,iz =int(round(g_inv[0])),int(round(g_inv[1])), int(round(g_inv[2]))
            ind=(max_index3+ix)*self.n_rgrid2*self.n_rgrid1+(max_index2+iy)*self.n_rgrid1+((max_index1+iz))
            normalized_fG[ind]=fG[ifG]
        normalized_fG3D=np.reshape(normalized_fG,(self.n_rgrid3,self.n_rgrid2,self.n_rgrid1))
        fR=np.ravel(np.fft.ifftn(np.fft.ifftshift(normalized_fG3D)))*len(self.rlist) 
        return fR
    
    def myfft_dens_v(self, fG):
        """ Compute the FFT of density-like function from G-space to Real space

        Args:
            fG (numpy array): function in G-space 
        Returns:
            numpy array: FFT of fG 
        """
        normalized_fG=np.zeros((self.n_rgrid1*self.n_rgrid2*self.n_rgrid3))+0J
        max_index1,max_index2,max_index3=int(self.n_rgrid1/2),int(self.n_rgrid2/2), int(self.n_rgrid3/2)
        for ifG in range(len(fG)): 
            g_inv=self.G_dens_int[ifG]
            ix, iy ,iz=int(round(g_inv[0])),int(round(g_inv[1])), int(round(g_inv[2]))
            ind=(max_index3+ix)*self.n_rgrid2*self.n_rgrid1+(max_index2+iy)*self.n_rgrid1+((max_index1+iz))
            normalized_fG[ind]=fG[ifG]
        normalized_fG3D=np.reshape(normalized_fG,(self.n_rgrid3,self.n_rgrid2,self.n_rgrid1))
        fR=np.ravel(np.fft.ifftn(np.fft.ifftshift(normalized_fG3D)))*len(self.rlist) 
        return fR

    def myifft_dens_v(self, fR): # inverse FT of the density which is different from above because we use more G vectors
        """ Compute the iFFT of density-like function from Real space  to G-space

        Args:
            fR (numpy array): function in Real space 
        Returns:
            numpy array: FFT of fR 
        """
        fR3D=np.reshape(fR,(self.n_rgrid3,self.n_rgrid2,self.n_rgrid1))
        fR3D=fR3D
        normalized_fG= np.ravel(np.fft.fftshift(np.fft.fftn(fR3D))/len(self.rlist))
        max_index1,max_index2,max_index3=int(self.n_rgrid1/2), int(self.n_rgrid2/2), int(self.n_rgrid3/2) 
        fG=[]
        for g in self.G_dens_int: 
            g_inv=g
            ix, iy ,iz=int(round(g_inv[0])), int(round(g_inv[1])), int(round(g_inv[2]))
            ind=(max_index3+ix)*self.n_rgrid2*self.n_rgrid1+(max_index2+iy)*self.n_rgrid1+((max_index1+iz))
            if ind<=self.n_rgrid1*self.n_rgrid2*self.n_rgrid3: 
                fG.append(normalized_fG[ind])
            else: 
                fG.append(0)
                sys.stdout = open("warning_fft.txt", "a")
                print("warning",ind,g)
                sys.stdout.close()
        return np.array(fG)    
      
    
class xc_functional: 
    def __init__(self, xc_type) -> None:
        self.potxcR=[]
        self.potxcG=[]
        self.xc_type=xc_type
        #### 
        self.av_funct={"LDA":self.LDA}
    def update_xc(self,system):
        self.av_funct[self.xc_type](system)
    # XC functionals :      
    def LDA(self, system) : 
        nx=np.sqrt(system.densR.real**2)
        # the potential in the direct space
        potential=-(3./np.pi)**(1./3.)*nx**(1./3.) +self.p_correlation_PZ(nx)  
        self.potxcR=potential
        self.potxcG=system.myifft_dens_v(potential)
        #return potential    
    ###############
    # XC utilities 
    ##############
    def p_correlation_PZ(self,n):
        rs= (4*np.pi/3*n)**(-1/3)
        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334
        ######
        Au=0.0311
        Bu=-0.048
        Cu=0.0020
        Du=-0.0116
        filterlow=(rs<1)
        filterhigh=(rs>=1)
        reslow=Au*np.log(rs)+(Bu-1/3 * Au) + 2/3 * Cu*rs*np.log(rs) +1/3 *(2*Du-Cu)*rs
        v_cep_rs= gamma / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
        reshigh= v_cep_rs* (1+7/6*beta1*np.sqrt(rs)+4/3*beta2*rs)/(1+beta1*np.sqrt(rs)+ beta2*rs)
        return reslow*filterlow + reshigh*filterhigh

class vxc_inverter:
    def __init__(self) -> None:
        self.potxcR=[]
        self.potxcG=[]
        self.alpha=5e-5
    def update_xc(self,Sys):
        self.potxcR= (Sys.densR_ref+self.alpha)/(Sys.densR + self.alpha) * self.potxcR
        self.potxcG=Sys.myifft_dens_v(self.potxcR)

    
class Hamiltonian_template: 
    def __init__(self,Sys,load_vnl=True) -> None:
        self.matrixGk_dict={}
        self.dict_vnl={}
                
        """## Constructing the matrices for each G_k """
        #@timeit
        def get_matrix(i,G_k_abinit_int_i, G_dic):
            def myG_dens_index(v):
                v=list(v)
                res=G_dic.get(str(v),-999) 
                if res==-999:
                    sys.stdout = open("warning_G_dens.txt", "a")
                    print("warning missing G in G_dens")
                    sys.stdout.close()
                return res  
             
            Gki=np.array(G_k_abinit_int_i)
            c=(Gki[:,None]-Gki[None,:])
            matrixGk=np.apply_along_axis(myG_dens_index, -1,c) 
            n_pkgmat=1
            if (i+1)%n_pkgmat==0 :
                return matrixGk 
        print("constructing matrix")
        delayed_funcs = [ delayed(get_matrix)(i,Sys.G_k_abinit_int[i], Sys.G_dic)  for i in range( len(Sys.K) )  ] 
        parallel_pool = Parallel(n_jobs=20)
        out = parallel_pool(delayed_funcs) 
        self.matrixGk_dict = {i:out[i] for i in range( len(Sys.K)) }
        # Loading  vnl 
        if load_vnl: 
            def get_vnl_k_opt(ik,G_k_abinit_lens, path_vnl ): 
                len_Gk=G_k_abinit_lens[ik]
                n_skip=ik
                for it in range(0,ik):
                    n_skip+=G_k_abinit_lens[it]
                vnl= np.genfromtxt(path_vnl,comments="ikpt",max_rows=len_Gk,skip_header=n_skip)
                res=vnl[:,0::2] + 1J*vnl[:,1::2]
                return res 
            delayed_funcs2=[ delayed(get_vnl_k_opt)(ik,Sys.G_k_abinit_lens,Sys.path_vnl) for ik in range(len( Sys.K )) ]
            out_vnl = parallel_pool(delayed_funcs2)
            for ik in range(len(Sys.K)):  
                self.dict_vnl[ik]=out_vnl[ik]
                #print(f"done for vnl {ik} / {len(Sys.K)}" , end="\r")

class KS_solver:
    def __init__(self,system) -> None:
        ### outputs 
        self.bands_history=[]
        self.densR_history=[np.copy(system.densR)] 
        self.potxcR_history=[] 
        ########## 
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
        T=KS_solver.get_kinetic(K_ik,G_k_abinit_ik)
        H_k=T+np.array(vnlk).T
        V_HxcExt= potHartreeG[matrixGk_dict_ik] + potxcG[matrixGk_dict_ik] + V_ext[matrixGk_dict_ik]
        H_k+=V_HxcExt
        return H_k 
    
    @staticmethod
    def get_densk_KS(K_ik, dict_vnl_ik, G_k_abinit_ik,matrixGk_dict_ik, potHartreeG, potxcG, V_ext, n_occup,V_unitcell,myfft_v_abinit,ik):
        H_k=KS_solver.get_Ham_ik(K_ik, dict_vnl_ik, G_k_abinit_ik, matrixGk_dict_ik, potHartreeG, potxcG, V_ext)
        #####
        bands=[] 
        ei_k, cim_k = np.linalg.eigh(H_k) #eigsh(H_k,system.n_occup*3, which='SA')   #np.linalg.eigh(H_k)  # 
        bands.append(ei_k[:n_occup*4])
        nrk_tot=0
        for j in range(n_occup):
            cr=1/np.sqrt(V_unitcell)*myfft_v_abinit(cim_k[:,j],ik)
            nr=2*np.conjugate(cr)*cr
            nrk_tot+=nr 
        return nrk_tot,bands 
    
    @timeit
    def get_dens_parallel(self, system, xc_functional,Ham_instance,update_system=True,mixing=0.75,njobs=number_of_cpu): 
        # Constructing the Hamiltonian 
        self.potxcG=xc_functional.potxcG 
        self.potHartreeG = (system.densG * np.array(system.HP_nodens)) 
        self.potxcR_history.append( xc_functional.potxcR ) 
        ##
        delayed_funcs = [delayed(self.get_densk_KS)(system.K[ik], Ham_instance.dict_vnl[ik], system.G_k_abinit[ik] ,Ham_instance.matrixGk_dict[ik], self.potHartreeG, self.potxcG, system.V_ext,system.n_occup,system.V_unitcell,system.myfft_v_abinit ,ik) for ik in range(len(system.K))]
        parallel_pool = Parallel(n_jobs=njobs, backend="threading" )  
        out= parallel_pool(delayed_funcs)
        ####        
        sumorbitk= np.sum(  np.array(out, dtype=object)[:,0], axis=0)/len(system.K)
        self.bands_history.append( [out[ik][1][0] for ik in range(len(system.K))] ) 
        if update_system : 
            system.densR = mixing* sumorbitk.real + (1-mixing)*self.densR_history[-1]
            self.densR_history.append( system.densR )
            system.densG=system.myifft_dens_v(system.densR)

def get_Hartree_energy(Sys):
        a_l= Sys.a_l[0]
        ### Hartree energy 
        Sys.G_dens_int = - np.array(Sys.G_dens_int) 
        sym_part = Sys.myifft_dens_v(Sys.densR) 
        Sys.G_dens_int = - np.array(Sys.G_dens_int)
        Eh= 0.5*np.sum( Sys.myifft_dens_v(Sys.densR)* Sys.HP_nodens*sym_part) * a_l **3 
        return Eh.real
                  
# Reporting     
from input_mat import input_dict
def my_print(txt):
    with open("log_"+input_dict["output"]+".txt", "a") as f:
         f.write(txt + "\n")
########
def reporting(n_it,system, ks_solver, xc_functional, nkpt_bands_print=1, save=True): 
    my_print(f"default_njob is {number_of_cpu}")
    my_print(f"# Iteration {n_it} diff_dens {np.mean(np.abs(ks_solver.densR_history[n_it]-ks_solver.densR_history[n_it-1])) } ")
    my_print('direct gap {} eV'.format((ks_solver.bands_history[-1][0][system.n_occup]-ks_solver.bands_history[-1][0][system.n_occup-1])*27.211) )
    Hartree = get_Hartree_energy(Sys = system) 
    my_print("Hartree energy is   "+  str(Hartree* 27.211)  + " eV"  )
    
    my_print(f"xc functional is {xc_functional.xc_type}")
    if xc_functional.xc_type.split('_')[0]=="connector": 
        my_print(f"connector n_0r {xc_functional.n_0r} ")  
        my_print(f"connector n_rrp {xc_functional.n_0r} ")
        my_print(f"connector sc {xc_functional.sc_con} ")          
    np.set_printoptions(precision=6)  
    for ik in range(nkpt_bands_print):
        my_print("ik="+str(ik)+" eigens "+str(ks_solver.bands_history[-1][ik][:system.n_occup +1]) + str(" Ha"))
    my_print(f"############# Iteration {n_it} ####################### \n")
    if save: 
        np.savez(input_dict["output"], dens_history=ks_solver.densR_history, potxc_history=ks_solver.potxcR_history, bands_history=ks_solver.bands_history)       
        # For connector add:  ncon=xc_functional.ncon, nrrp=xc_functional.nrrp, n_0r=xc_functional.n_0r, nbar_sc0=xc_functional.nbar_sc0  )
      
  
