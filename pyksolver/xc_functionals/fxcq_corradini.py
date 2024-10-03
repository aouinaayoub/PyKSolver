import numpy as np 
import pyksolver.xc_functionals.heg as HEG 

def Ac(n): 
  ach=(np.log(2)-1)/(2*np.pi**2)
  bch=20.456 
  KF= (3*np.pi**2*n)**(1/3)
  rs= (4*np.pi/3*n)**(-1/3)
  drs=(4*np.pi/3)**(-1/3)*(-1/3)*n**(-4/3)
  dnec= ach*np.log((1+bch/rs+bch/rs**2)) + ach*n*(-bch/rs**2 -2*bch/rs**3)/(1+bch/rs+bch/rs**2)*drs
  A=1/4 -KF**2/(4*np.pi) * dnec
  return A 

def Cc(rs): 
   ach=(np.log(2)-1)/(2*np.pi**2)
   bch=20.456 
   n=rs**(-3)*3/(4*np.pi)
   KF= (3*np.pi**2*n)**(1/3)
   drsec=ach*np.log((1+bch/rs+bch/rs**2)) + ach*rs*(-bch/rs**2 -2*bch/rs**3)/(1+bch/rs+bch/rs**2)
   C=-np.pi/(2*KF) * drsec
   return C 

def gc(rs): 
  n=rs**(-3)*3/(4*np.pi) 
  g=Bc(rs)/(Ac(n)-Cc(rs))
  return g

def Bc(rs): 
  x=np.sqrt(rs)
  a1=2.15
  a2=0.435
  b1=1.57
  b2=0.409 
  res=(1+a1*x+a2*x**3)/(3+b1*x+b2*x**3)
  return res

def alphac(rs): 
  n=rs**(-3)*3/(4*np.pi)
  alpha=1.5/rs**(1/4) *Ac(n)/(Bc(rs)*gc(rs))
  return alpha 

def betac(rs):
  beta=1.2/(Bc(rs)*gc(rs))
  return beta 

def fxcq_ana(n,q):
  third=1./3. 
  rs = (3. / (4.0 * np.pi * n)) ** third
  q=q+1e-18
  KF= (3*np.pi**2*n)**(1/3)
  Q=q/KF 
  G= Cc(rs)*Q**2 + Bc(rs)*Q**2/(gc(rs)+Q**2) + alphac(rs)*Q**4 *np.exp(-betac(rs)*Q**2)