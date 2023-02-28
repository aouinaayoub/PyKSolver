from input_mat import * 
import sys
import pickle 
from scipy.sparse.linalg import eigs, eigsh  

# to redirect prints to the log file 
def my_print(txt):
    with open("log_"+output_filename+".txt", "a") as f:
         f.write(txt)  

# redefine fourier transform 
## direct FT, using the corresponding G-set for each k
def myfft_v_abinit(fG,G_k_abinit_int_ik,rlist):
    normalized_fG=np.zeros((n_rgrid1*n_rgrid2*n_rgrid3))+0J
    max_index1,max_index2,max_index3=int(n_rgrid1/2),int(n_rgrid2/2), int(n_rgrid3/2)
    for ifG in range(len(fG)): 
      g_inv=G_k_abinit_int_ik[ifG]
      ix, iy ,iz =int(round(g_inv[0])),int(round(g_inv[1])), int(round(g_inv[2]))
      ind=(max_index3+ix)*n_rgrid2*n_rgrid1+(max_index2+iy)*n_rgrid1+((max_index1+iz))
      normalized_fG[ind]=fG[ifG]
    normalized_fG3D=np.reshape(normalized_fG,(n_rgrid3,n_rgrid2,n_rgrid1))
    fR=np.ravel(np.fft.ifftn(np.fft.ifftshift(normalized_fG3D)))*len(rlist) 
    return fR

def myfft_dens_v(fG,G_dens_int,rlist):
    normalized_fG=np.zeros((n_rgrid1*n_rgrid2*n_rgrid3))+0J
    max_index1,max_index2,max_index3=int(n_rgrid1/2),int(n_rgrid2/2), int(n_rgrid3/2)
    for ifG in range(len(fG)): 
      g_inv=G_dens_int[ifG]
      ix, iy ,iz=int(round(g_inv[0])),int(round(g_inv[1])), int(round(g_inv[2]))
      ind=(max_index3+ix)*n_rgrid2*n_rgrid1+(max_index2+iy)*n_rgrid1+((max_index1+iz))
      normalized_fG[ind]=fG[ifG]
    normalized_fG3D=np.reshape(normalized_fG,(n_rgrid3,n_rgrid2,n_rgrid1))
    fR=np.ravel(np.fft.ifftn(np.fft.ifftshift(normalized_fG3D)))*len(rlist) 
    return fR

def myifft_dens_v(fR,G_dens_int,rlist): # inverse FT of the density which is different from above because we use more G vectors
    fR3D=np.reshape(fR,(n_rgrid3,n_rgrid2,n_rgrid1))
    fR3D=fR3D
    normalized_fG= np.ravel(np.fft.fftshift(np.fft.fftn(fR3D))/len(rlist))
    max_index1,max_index2,max_index3=int(n_rgrid1/2), int(n_rgrid2/2), int(n_rgrid3/2) 
    fG=[]
    for g in G_dens_int: 
      g_inv=g
      ix, iy ,iz=int(round(g_inv[0])), int(round(g_inv[1])), int(round(g_inv[2]))
      ind=(max_index3+ix)*n_rgrid2*n_rgrid1+(max_index2+iy)*n_rgrid1+((max_index1+iz))
      if ind<=n_rgrid1*n_rgrid2*n_rgrid3: 
        fG.append(normalized_fG[ind])
      else: 
        fG.append(0)
        sys.stdout = open("warning_fft.txt", "a")
        print("warning",ind,g)
        sys.stdout.close()
    return np.array(fG)


"""## Kinetic energy """
def get_kinetic(k,G):
    k=np.array(k)
    G=np.array(G)
    absKGm=[]
    for g in G:
        ans=np.linalg.norm(k+g)**2/2
        absKGm.append(ans)
    absKGm=np.array(absKGm)
    T=np.diagflat(absKGm)
    return T

"""## Hartree potential """
def Hartree_v(densG, HP_nodens, k): 
  if k==-999: 
    potential=0
  else:
    potential=densG[k]*HP_nodens[k]
  return potential


def get_xc_v(k,potxcG):
    if k==-999:
        rt=0
    else: rt=potxcG[k]
    return rt
"""## The external potential"""

def get_Vext_k(k, V_ext): 
    if k==-999: #or np.linalg.norm(G_dens[k])>0.78383 : 
        r=0
    else: r=V_ext[k]
    return r

"""## Nonlocal potential """
def get_vnl(i, G_k_abinit_i):
        myham=[]
        kline = next(f).split()
        #print(kline)
        for j in range(len(G_k_abinit_i)):
            line2=next(f).split()
            v=[]
            for i in range(int(len(line2)/2)):
                v.append(float(line2[2*i])+1J*float(line2[2*i+1]))
            myham.append(v)
        return myham

"""## Parameter of the material """
def config(): #(a_l,n_rgrid1,n_rgrid2,n_rgrid3, kx, Ecut, n_pkgmat,flagmatrix, MtR,V_extR):
    rlist, rlist_int=[],[] 
    for z in range(0,n_rgrid3): 
        for y in range(0,n_rgrid2): 
            for x in range(0,n_rgrid1):
                xr, yr ,zr=x/n_rgrid1, y/n_rgrid2, z/n_rgrid3
                rlist.append(np.dot(MtR.T,[xr,yr,zr]))
                rlist_int.append([x,y,z])  

#Change of basis 
    Mt=2*np.pi*(np.linalg.inv(MtR)) 
#Mt_int=np.array([[-1,1,1],[1, -1,1],[1, 1,-1]])
    kx2=np.roll(kx,1+int(len(kx)/2))   
    K=[]
    K_int=[]
    for k in kx2:
        for j in kx2 : 
            for i in kx2 :
                hold=np.dot(Mt,[i,j,k])
                K.append(hold)
                K_int.append([i,j,k])
    K=np.array(K)


    qmax=int(np.sqrt(2*Ecut)/np.linalg.norm(Mt[:,0]))
    qlist=np.arange(-qmax-3,qmax+4)# this order is important for fourier transform #np.roll(np.arange(-qmax-3,qmax+4),qmax+4) 
    qlist_abinit=np.roll(np.arange(-qmax-3,qmax+4),qmax+4) 

# Grid of G accoridng to Ecut according to abinit
    G_k_abinit_int, G_k_abinit=[],[]
    for q in K: 
        G,G_int=[],[]
        for k in qlist_abinit: 
            for j in qlist_abinit: 
                for i in qlist_abinit: 
                    g_orth=np.dot(Mt,[i,j,k])
                    if np.linalg.norm(q+g_orth)**2/2<Ecut : 
                        x,y,z=g_orth[0],g_orth[1],g_orth[2] 
                        G.append([x,y,z]),G_int.append([i,j,k])                 
        G_k_abinit.append(G), G_k_abinit_int.append(G_int)

# Grid of G-density 
    G_dens,  G_dens_int=[],[]
    #qmax2=int(np.sqrt(8*Ecut)/b)
    qmax2_x=int(n_rgrid1/2)#2int(np.sqrt(8*Ecut)/np.linalg.norm(Mt[:,0]))*2
    qmax2_y=int(n_rgrid2/2) 
    qmax2_z=int(n_rgrid3/2)
    G_dic={}
    count=0
    for i in range(-qmax2_x,qmax2_x): 
        for j in range(-qmax2_y,qmax2_y): 
            for k in range(-qmax2_z,qmax2_z): 
                g_orth=np.dot(Mt,[i,j,k])
                if np.linalg.norm(g_orth)**2/2<4*Ecut  : 
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

    
    V_ext=myifft_dens_v(V_extR,G_dens_int, rlist)


     
    vHartree=np.vectorize(Hartree_v,excluded=[0,1])
    vget_xc=np.vectorize(get_xc_v,excluded=[1])
    vget_Vext=np.vectorize(get_Vext_k,excluded=[1])
    
    """## Constructing the matrices for each G_k """

    def myG_dens_index(v):
        v=list(v)
        res=G_dic.get(str(v),-999) 
        if res==-999:
            my_print("warning missing G in G_dens")
        return res   #G_dic[str(v)]     #direct implementattion but slow: G_dens.index(v)

    if flagmatrix==True: 
       my_print('Constructing the matrix \n')
       matrixGk=[]
       count=1
       for i in range(0,len(K)): 
           Gki=np.array(G_k_abinit_int[i])
           c=(Gki[:,None]-Gki[None,:])
           #c=c.astype(int)
           #c=np.round(c,decimals=5)
           matrixGk.append(np.apply_along_axis(myG_dens_index, -1,c)) # old version matrixGk.append(np.apply_along_axis(vwhereG, -1,c))
           #my_print(str(i)+'>> finished')
           if (i+1)%n_pkgmat==0 :
              matfile=open(matrix_dir+"/matrix_ik_{}.pkl".format(count),"wb") 
              pickle.dump(matrixGk,matfile)
              matfile.close()
              count+=1
              matrixGk=[]
    G_k_abinit_lens=[len(G_k_abinit[ik]) for ik in range(len(K)) ]
    return rlist, rlist_int, K, K_int, G_k_abinit_int, G_k_abinit , G_dens, G_dens_int, G_dic, gnorm, HP_nodens, V_ext,vHartree, vget_xc, vget_Vext, G_k_abinit_lens 
    
def get_densk_KS(ik, vHartree, vget_xc, vget_Vext,G_k_abinit_lens,G_k_abinit_ik,G_k_abinit_int_ik,matrixGki, K,densG, HP_nodens,potxcG,V_ext,rlist):
    #Print("the density:"+str(densR[:24]))  
    #my_print("KS loop for ", ik )
    def sort_eigen(ei_k,cim_k): 
        dic={}
        for i in range (len(ei_k)): 
            dic[ei_k[i]]=cim_k[:,i]
        ei_k_sorted=sorted(dic.keys() )
        cim_k_sorted=np.zeros_like(cim_k)
        for j in range(len(ei_k_sorted)) : 
            cim_k_sorted[:,j]=dic[ei_k_sorted[j]]
        return np.array(ei_k_sorted), np.array(cim_k_sorted)  
    bands=[]
    vnlk=get_vnl_k(ik, G_k_abinit_lens)
    T=get_kinetic(K[ik],G_k_abinit_ik)
    H_k=T+np.array(vnlk).T
    H_k+=vHartree(densG, HP_nodens,matrixGki) 
    H_k+=vget_xc(matrixGki,potxcG) 
    H_k+=vget_Vext(matrixGki, V_ext) 
    ei_k, cim_k = np.linalg.eigh(H_k) #eigsh(H_k,n_occup*3, which='SA')  
    bands.append(ei_k[:10])

    nrk_tot=0
    for j in range(n_occup):
        cr=1/np.sqrt(V_unitcell) *myfft_v_abinit(cim_k[:,j],G_k_abinit_int_ik,rlist)
        nr=2*np.conjugate(cr)*cr
        nrk_tot+=nr 
    return nrk_tot,bands 

def get_vnl_k(ik,G_k_abinit_lens): 
  len_Gk=G_k_abinit_lens[ik]
  n_skip=ik
  for it in range(0,ik):
      n_skip+=G_k_abinit_lens[it]
  vnl= np.genfromtxt(path_vnl,comments="ikpt",max_rows=len_Gk,skip_header=n_skip)
  vnlk2=[]
  for k in range(len_Gk): 
    vnlk=[vnl[k][i]+1J*vnl[k][i+1] for i in range(0,len(vnl[k]),2)]
    vnlk2.append(vnlk)
  return vnlk2  


#===================================
def p_correlation_PZ(n):
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

def xc_real(nx):
    nx=np.sqrt(nx.real**2)
    potential=-(3./np.pi)**(1./3.)*nx**(1./3.) +p_correlation_PZ(nx)  ##the potential in the direct space 
    return potential

dict_vxc={ "LDA": (xc_real, ["densR"])}

def get_vxc_potentials(xc_type,densG,densR,gnorm,rlist,rlist_int,G_dens,G_dens_int,sc_num=3):
    dict_vars={"densG": densG, "densR":densR, "gnorm":gnorm,
            "rlist": rlist, "rlist_int":rlist_int, "G_dens":G_dens, 
            "G_dens_int":G_dens_int}
    try: 
        vxc_func, args_name= dict_vxc[xc_type]
        arg=[dict_vars[key] for key in args_name]
        return vxc_func(*arg)
    
    except: 
        print("XC functional not available")

