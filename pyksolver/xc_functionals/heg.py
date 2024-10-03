import numpy as np 

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
    potential=-(3./np.pi)**(1./3.)*nx**(1./3.) + p_correlation_PZ(nx)  ##the potential in the direct space 
    return potential


# corradini PZ
def diffv_cep(n):
    # Multiplied v_cep from above with rs and differentiated wrt rs
    third = 1. / 3.
    rs = (3. / (4.0 * np.pi * n)) ** third
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    # res = gamma * (beta1 * np.sqrt(rs) + 2) / \
    #    (2 * (beta1 * np.sqrt(rs) + beta2 * rs + 1) ** 2)
    res = (beta1 * gamma * np.sqrt(rs)) / \
        (2 * (beta1 * np.sqrt(rs) + beta2 * rs + 1) ** 2) + gamma / \
        (beta1 * np.sqrt(rs) + beta2 * rs + 1) ** 2
    return res

def diffvc(n):
    # from dp-code
    third = 1. / 3.
    a = 0.0311
    # b = -0.0480  #  never used?
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    rs = (3. / (4.0 * np.pi * n)) ** third

    stor1 = (1.0 + beta1 * np.sqrt(rs) + beta2 * rs) ** (-3.0)
    stor2 = -0.41666667 * beta1 * \
            (rs ** (-0.5)) - 0.5833333 * (beta1 ** 2) - 0.66666667 * beta2
    stor3 = -1.75 * beta1 * beta2 * np.sqrt(rs) - 1.3333333 * rs * \
            (beta2 ** 2)
    reshigh = gamma * stor1 * (stor2 + stor3)
    reslow = a / rs + 0.66666667 * (c * np.log(rs) + d) + 0.33333333 * c

    reshigh = reshigh * (-4.0 * np.pi / 9.0) * (rs ** 4)
    reslow = reslow * (-4.0 * np.pi / 9.0) * (rs ** 4)

    filterlow=(rs<1)
    filterhigh=(rs>=1)
    return reslow*filterlow + reshigh*filterhigh

def Hfxc(n):
    res1=diffvc(n)
    res2=-(3/np.pi)**(1/3) *1/3 *n**(-2/3)
    return res1+res2

def fxcr_smooth_pz(n,r):
    third=1./3. 
    rs = (3. / (4.0 * np.pi * n)) ** third
    k_F = (3. * np.pi ** 2. * n) ** (1.0 / 3.0)
    # Q = q
    e = 1                       # atomic units
#   diff_mu = 1                 # How to model this for HEG?
#                              This should be d \mu_c / d n_0
    diff_mu = diffvc(n)
    A = 1. / 4. - (k_F ** 2.) / (4. * np.pi * e ** 2.) * diff_mu
#   diff_rse = 1                # How to model this for HEG? e_c(rs) !!
#                                 This should be d(rs * e_c) / d rs
    diff_rse = diffv_cep(n)
    C = np.pi / (2. * e ** 2. * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    x = rs ** (1. / 2.)
    B = (1. + a1 * x + a2 * x ** 3.) / (3. + b1 * x + b2 * x ** 3.)
    g = B / (A - C)
    alpha = 1.5 / (rs ** (1. / 4.)) * A / (B * g)
    beta = 1.2 / (B * g)
    fxcr=  alpha*k_F * np.pi**1.5/(4. * np.pi**2 *beta**2.5)*(k_F**2 * r**2/(2*beta)-3) * np.exp(-k_F**2 * r**2/(4*beta))
    return fxcr

def fxcr_residual_pz(n,r,integral=0):
    third=1./3. 
    rs = (3. / (4.0 * np.pi * n)) ** third
    k_F = (3. * np.pi ** 2. * n) ** (1.0 / 3.0)
    # Q = q
    e = 1                       # atomic units
#   diff_mu = 1                 # How to model this for HEG?
#                              This should be d \mu_c / d n_0
    diff_mu = diffvc(n)
    A = 1. / 4. - (k_F ** 2.) / (4. * np.pi * e ** 2.) * diff_mu
#   diff_rse = 1                # How to model this for HEG? e_c(rs) !!
#                                 This should be d(rs * e_c) / d rs
    diff_rse = diffv_cep(n)
    C = np.pi / (2. * e ** 2. * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    x = rs ** (1. / 2.)
    B = (1. + a1 * x + a2 * x ** 3.) / (3. + b1 * x + b2 * x ** 3.)
    g = B / (A - C)
    alpha = 1.5 / (rs ** (1. / 4.)) * A / (B * g)  # unused!
    beta = 1.2 / (B * g)                            # unused !
    g = B / (A - C)
    if integral:
        return -B*r*np.exp(-np.sqrt(g)*k_F*r) # 
    else: 
        return -B*np.exp(-np.sqrt(g)*k_F*r)/r