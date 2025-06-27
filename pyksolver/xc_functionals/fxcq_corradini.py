import numpy as np 
import pyksolver.xc_functionals.heg as HEG 

def Ac(n: float) -> float: 
  """
  Compute the A_c coefficient for a given density n.
  """
  ach=(np.log(2)-1)/(2*np.pi**2)
  bch=20.456 
  KF= (3*np.pi**2*n)**(1/3)
  rs= (4*np.pi/3*n)**(-1/3)
  drs=(4*np.pi/3)**(-1/3)*(-1/3)*n**(-4/3)
  denom = 1 + bch / rs + bch / rs ** 2 + 1e-18
  dnec= ach*np.log(denom) + ach*n*(-bch/rs**2 -2*bch/rs**3)/denom*drs
  A=1/4 -KF**2/(4*np.pi) * dnec
  return A 

def Cc(rs: float) -> float: 
   """
   Compute the C_c coefficient for a given Wigner-Seitz radius rs.
   """
   ach=(np.log(2)-1)/(2*np.pi**2)
   bch=20.456 
   n=rs**(-3)*3/(4*np.pi)
   KF= (3*np.pi**2*n)**(1/3)
   denom = 1 + bch / rs + bch / rs ** 2 + 1e-18
   drsec=ach*np.log(denom) + ach*rs*(-bch/rs**2 -2*bch/rs**3)/denom
   C=-np.pi/(2*KF + 1e-18) * drsec
   return C 

def gc(rs: float) -> float: 
  """
  Compute the g_c coefficient for a given Wigner-Seitz radius rs.
  """
  n=rs**(-3)*3/(4*np.pi) 
  denominator = Ac(n) - Cc(rs) + 1e-18
  g=Bc(rs)/denominator
  return g

def Bc(rs: float) -> float: 
  """
  Compute the B_c coefficient for a given Wigner-Seitz radius rs.
  """
  x=np.sqrt(rs)
  a1=2.15
  a2=0.435
  b1=1.57
  b2=0.409 
  denom = 3 + b1 * x + b2 * x ** 3 + 1e-18
  res=(1+a1*x+a2*x**3)/denom
  return res

def alphac(rs: float) -> float: 
  """
  Compute the alpha_c coefficient for a given Wigner-Seitz radius rs.
  """
  n=rs**(-3)*3/(4*np.pi)
  denominator = Bc(rs) * gc(rs) + 1e-18
  alpha=1.5/rs**(1/4) *Ac(n)/denominator
  return alpha 

def betac(rs: float) -> float:
  """
  Compute the beta_c coefficient for a given Wigner-Seitz radius rs.
  """
  denominator = Bc(rs) * gc(rs) + 1e-18
  beta=1.2/denominator
  return beta 

def fxcq_ana(n: float, q: float) -> float:
  """
  Analytical expression for f_xc(q) as a function of density n and wavevector q.
  """
  third=1./3. 
  rs = (3. / (4.0 * np.pi * n)) ** third
  q=q+1e-18
  KF= (3*np.pi**2*n)**(1/3)
  Q=q/KF 
  G= Cc(rs)*Q**2 + Bc(rs)*Q**2/(gc(rs)+Q**2) + alphac(rs)*Q**4 *np.exp(-betac(rs)*Q**2)
  return G