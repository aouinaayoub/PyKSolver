import numpy as np 
import sys
from typing import Any, Dict, List

class System: 
    """
    Represents a physical system for Kohn-Sham calculations, including grid setup, FFTs, and reciprocal space vectors.
    """
    def __init__(self,input_dict: Dict[str, Any]) -> None:
        """
        Initialize the System with parameters from input_dict.
        """
        self.a_l=input_dict["a_l"]
        self.MtR=input_dict["MtR"]
        self.n_occup=input_dict["n_occup"]
        self.V_unitcell= np.dot(np.cross([self.a_l[0],0,0],[0,self.a_l[1],0]) , [0, 0,self.a_l[2]])/4 
        self.kx=input_dict["kx"] 
        self.Ecut=input_dict["Ecut"]
        # local potential  
        self.V_extR =np.genfromtxt(input_dict["path_VextR"]) 
        # grid in real space, it depends on the cutoff 
        self.n_rgrid1,self.n_rgrid2,self.n_rgrid3 = [round(len(self.V_extR)**(1/3))]*3 
        #self.rlist, self.rlist_int, self.K, self.K_int, self.G_k_abinit_int, self.G_k_abinit , self.G_dens, self.G_dens_int, self.G_dic, self.gnorm, self.HP_nodens, self.G_k_abinit_lens=self.config()
        # FFT the psp once for all 
        # The density (will be updated)
        self.densR=input_dict["densR"]   
        self.path_vnl= input_dict["path_vnl"]
            
    ##################
    # Useful functions
    ##################
        """## Parameter of the material """
    #@timeit
    def config(self):
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
        #qlist=np.arange(-qmax-3,qmax+4)# this order is important for fourier transform #np.roll(np.arange(-qmax-3,qmax+4),qmax+4) 
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
        qmax2_x=int(self.n_rgrid1/2)                        #2int(np.sqrt(8*Ecut)/np.linalg.norm(Mt[:,0]))*2
        qmax2_y=int(self.n_rgrid2/2) 
        qmax2_z=int(self.n_rgrid3/2)
        G_dic={}
        count=0
        for i in range(-qmax2_x,qmax2_x): 
            for j in range(-qmax2_y,qmax2_y): 
                for k in range(-qmax2_z,qmax2_z): 
                    g_orth=np.dot(Mt,[i,j,k])
                    if np.linalg.norm(g_orth)**2/2<4*self.Ecut  : 
                        #print([i,j,k], '>>' , np.linalg.norm(q+[i,j,k])**2/2)
                        x=g_orth[0]#round(g_orth[0],5)
                        y=g_orth[1]#round(g_orth[1],5)
                        z=g_orth[2]#round(g_orth[2],5)
                        G_dens.append([x,y,z])
                        G_dic[str([i,j,k])]=count
                        G_dens_int.append([i,j,k])
                        count+=1

        gnorm=np.linalg.norm(G_dens,axis=1) ## useful for connector 

        HP_nodens=[]
        for v in G_dens:
            if np.linalg.norm(v)==0:
                HP_nodens.append(0)
            else: HP_nodens.append( 4*np.pi * 1/np.linalg.norm(v)**2)

        
        G_k_abinit_lens=[len(G_k_abinit[ik]) for ik in range(len(K)) ]
        
        self.rlist = rlist 
        self.rlist_int = rlist_int 
        self.K= K 
        self.K_int = K_int 
        self.G_k_abinit_int = G_k_abinit_int  
        self.G_k_abinit =  G_k_abinit
        self.G_dens = G_dens
        self.G_dens_int = G_dens_int
        self.G_dic = G_dic 
        self.gnorm =gnorm 
        self.HP_nodens = HP_nodens 
        self.G_k_abinit_lens = G_k_abinit_lens
        self.V_ext=self.myifft_dens_v(self.V_extR) # should be in KS solver or ham constructor  
        self.densG= self.myifft_dens_v(self.densR) 

        #return rlist, rlist_int, K, K_int, G_k_abinit_int, G_k_abinit , G_dens, G_dens_int, G_dic, gnorm, HP_nodens, G_k_abinit_lens 
        
    #########
    # My FFTs 
    #########
    # redefine fourier transform 
    ## direct FT, using the corresponding G-set for each k
    def myfft_v_abinit(self,fG: np.ndarray,ik: int) -> np.ndarray:
        """
        Direct Fourier transform using the corresponding G-set for each k.
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
    
    def myfft_dens_v(self, fG: np.ndarray) -> np.ndarray:
        """
        Fourier transform for the density using the G-density grid.
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

    def myifft_dens_v(self, fR: np.ndarray) -> np.ndarray:
        """
        Inverse Fourier transform for the density using the G-density grid.
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
                with open("warning_fft.txt", "a") as f:
                    f.write(f"warning {ind} {g}\n")
        return np.array(fG)    
#######################################      