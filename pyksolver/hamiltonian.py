import sys 
import numpy as np 


class HamiltonianTemplate: 
    def __init__(self,Sys,load_vnl=True) -> None:
        """## Constructing the matrices for each G_k """
        def myG_dens_index(v):
            v=list(v)
            res=Sys.G_dic.get(str(v),-999) 
            if res==-999:
                sys.stdout = open("warning_G_dens.txt", "a")
                print("warning missing G in G_dens")
                sys.stdout.close()
            return res   
        self.matrixGk_dict={}
        #count=1
        for i in range(0,len(Sys.K)): 
            Gki=np.array(Sys.G_k_abinit_int[i])
            c=(Gki[:,None]-Gki[None,:])
            matrixGk=np.apply_along_axis(myG_dens_index, -1,c) 
            n_pkgmat=1
            if (i+1)%n_pkgmat==0 :
                self.matrixGk_dict[i]=matrixGk 
                #count+=1
        # Loading  vnl 
        if load_vnl: 
            def get_vnl_k_opt(ik,Sys): 
                len_Gk=Sys.G_k_abinit_lens[ik]
                n_skip=ik
                for it in range(0,ik):
                    n_skip+=Sys.G_k_abinit_lens[it]
                vnl= np.genfromtxt(Sys.path_vnl,comments="ikpt",max_rows=len_Gk,skip_header=n_skip)
                res=vnl[:,0::2] + 1J*vnl[:,1::2]
                return res 
            
            self.dict_vnl={ik: get_vnl_k_opt(ik,Sys) for ik in range(len(Sys.K))}