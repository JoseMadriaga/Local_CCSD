"""
ccwfn.py: CC T-amplitude Solver
This was an attempt to understand which local basis is being run through, I've only expanded the amplitudes and integrals to the local basis whiletransforming them into the semicanonical local basis at the correction section; however, this doesn't seem to work ... might revisit when I have a better understanding 
"""

if __name__ == "__main__":
    raise Exception("This file cannot be invoked on its own.")


import psi4
import time
import numpy as np
from opt_einsum import contract
from utils import helper_diis
from hamiltonian import Hamiltonian
from local import Local


class lccwfn_test1(object):
    """
    An RHF-CC wave function and energy object.

    Attributes
    ----------
    ref : Psi4 SCF Wavefunction object
        the reference wave function built by Psi4 energy() method
    eref : float
        the energy of the reference wave function (including nuclear repulsion contribution)
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, and the spin-adapted ERIs (L)
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace
    Dia : NumPy array
        one-electron energy denominator
    Dijab : NumPy array
        two-electron energy denominator
    t1 : NumPy array
        T1 amplitudes
    t2 : NumPy array
        T2 amplitudes
    ecc | float
        the final CC correlation energy

    Methods
    -------
    solve_cc()
        Solves the CC T amplitude equations
    residuals()
        Computes the T1 and T2 residuals for a given set of amplitudes and Fock operator
    """

    def __init__(self, scf_wfn, **kwargs):
        """
        Parameters
        ----------
        scf_wfn : Psi4 Wavefunction Object
            computed by Psi4 energy() method

        Returns
        -------
        None
        """

        time_init = time.time()

        valid_cc_models = ['CCD', 'CC2', 'CCSD', 'CCSD(T)', 'CC3']
        model = kwargs.pop('model','CCSD')
        if model not in valid_cc_models:
            raise Exception("%s is not an allowed CC model." % (model))
        self.model = model
        
        # models requiring singles
        self.need_singles = ['CCSD', 'CCSD(T)']

        # models requiring T1-transformed integrals
        self.need_t1_transform = ['CC2', 'CC3']

        valid_local_models = [None, 'LPNO', 'PAO','LPNOpp']
        local = kwargs.pop('local', None)
        # TODO: case-protect this kwarg
        if local not in valid_local_models:
            raise Exception("%s is not an allowed local-CC model." % (local))
        self.local = local
        self.local_cutoff = kwargs.pop('local_cutoff', 1e-5)

        valid_local_MOs = ['PIPEK_MEZEY', 'BOYS']
        local_MOs = kwargs.pop('local_mos', 'PIPEK_MEZEY')
        if local_MOs not in valid_local_MOs:
            raise Exception("%s is not an allowed MO localization method." % (local_MOs))
        self.local_MOs = local_MOs

        self.ref = scf_wfn
        self.eref = self.ref.energy()
        self.nfzc = self.ref.frzcpi()[0]                # assumes symmetry c1
        self.no = self.ref.doccpi()[0] - self.nfzc      # active occ; assumes closed-shell
        self.nmo = self.ref.nmo()                       # all MOs/AOs
        self.nv = self.nmo - self.no - self.nfzc        # active virt
        self.nact = self.no + self.nv                   # all active MOs

        print("NMO = %d; NACT = %d; NO = %d; NV = %d" % (self.nmo, self.nact, self.no, self.nv))

        # orbital subspaces
        self.o = slice(0, self.no)
        self.v = slice(self.no, self.nmo)

        # For convenience
        o = self.o
        v = self.v

        # Get MOs
        C = self.ref.Ca_subset("AO", "ACTIVE")
        npC = np.asarray(C)  # as numpy array
        self.C = C

        # Localize occupied MOs if requested
        if (local is not None):
            C_occ = self.ref.Ca_subset("AO", "ACTIVE_OCC")
            LMOS = psi4.core.Localizer.build(self.local_MOs, self.ref.basisset(), C_occ)
            LMOS.localize()
            npL = np.asarray(LMOS.L)
            npC[:,:self.no] = npL
            C = psi4.core.Matrix.from_array(npC)
            self.C = C
          
        self.H = Hamiltonian(self.ref, self.C, self.C, self.C, self.C)
        
        if local is not None:
            self.Local = Local(local, self.C, self.nfzc, self.no, self.nv, self.H, self.local_cutoff)        
            self.transform_integral(o,v)
              
        # denominators
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        self.Dia = eps_occ.reshape(-1,1) - eps_vir
        self.Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        print("mp2 energy without truncation")
        t2_test = self.H.ERI[o,o,v,v]/self.Dijab

        print(contract('ijab,ijab->', t2_test, self.H.L[o,o,v,v]))

        # first-order amplitudes
        self.t1 = np.zeros((self.no, self.nv))

        if local is not None:
            t2_ij = []
            emp2 = 0
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i * self.no + i
                 
                X = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.H.ERI[i,j,v,v] @ self.Local.Q[ij] @ self.Local.L[ij] 
                t2_ij.append( -1*X/ (self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])) 
                
                t2_ij[ij] = self.Local.L[ij] @ t2_ij[ij] @ self.Local.L[ij].T 
                L_ij = 2.0 * t2_ij[ij] - t2_ij[ij].T
                mp2_ij = np.sum(np.multiply(self.ERIoovv_ij[ij][i,j], L_ij))
                emp2 += mp2_ij
                
                #t2_ij[ij] = self.Local.L[ij] @ t2_ij[ij] @ self.Local.L[ij].T

            print("mp2 energy in the local basis")    
            print(emp2)
            self.t2_ij = t2_ij
         
        print("CC object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_localcc(self, e_conv=1e-7,r_conv=1e-7,maxiter=1,max_diis=8,start_diis=8):
        o = self.o 
        v = self.v        
        F = self.H.F
        L = self.H.L
        Dia = self.Dia
        Dijab = self.Dijab

        emp2 = 0
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
 
            L_ij = 2.0 * self.t2_ij[ij] - self.t2_ij[ij].T
            mp2_ij = np.sum(np.multiply(self.ERIoovv_ij[ij][i,j], L_ij))
            emp2 += mp2_ij

        print("mp2 energy in the local basis but not in the semicanonical basis for t2 right before")    
        print(emp2)
        
        ecc = self.localcc_energy(o,v,F,self.Loovv_ij,self.t1,self.t2_ij)
        print("CC Iter %3d: CC Ecorr = %.15f dE = % .5E MP2" % (0,ecc,-ecc)) 
        for niter in range(1, maxiter+1):

            ecc_last = ecc  

            r1, r2_ij = self.local_residuals(self.Fov_ii, self.t1, self.t2_ij)
            
            self.t1 += r1/Dia
         
            rms = 0
            cool = np.zeros((self.no,self.nv))
            coolest = np.zeros((self.no,self.nv))
            cooler = np.zeros((self.no, self.no, self.nv, self.nv))
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i*self.no +i 
  
                self.t2_ij[ij] = self.Local.L[ij].T @ self.t2_ij[ij] @ self.Local.L[ij]          
                self.t2_ij[ij] -= r2_ij[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])
                self.t2_ij[ij] = self.Local.L[ij] @ self.t2_ij[ij] @ self.Local.L[ij].T                  
                #cooler[i,j] = self.Local.Q[ij] @ self.Local.L[ij] @ r2_ij[ij] @ self.Local.L[ij].T @ self.Local.Q[ij].T 

                rms += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])
            
            rms = np.sqrt(rms)
            #print("cool") 
            #print(cool)
            #print("cooler")
            #print(cooler)
            #print("coolest")
            #print(coolest)           
            ecc = self.localcc_energy(o,v,F,self.Loovv_ij,self.t1,self.t2_ij)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                #print("\nCC has converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(%s) = %20.15f" % (self.model, ecc))
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                self.ecc = ecc
                return ecc        
   
    def transform_integral(self,o,v):
        
        Q = self.Local.Q
        L = self.Local.L
        Fov_ii = []
        Fvv_ij = []
        
        ERIoovo_ij = []
        ERIooov_ij = []
        ERIovvv_ij = []
        ERIvvvv_ij = []
        ERIoovv_ij = []

        Loovv_ij = []
        Lovvv_ij = []
        Looov_ij = []
        Loovo_ij = []
        Lovvo_ij = [] 
        #contraction notation i,j,a,b typically MO; A,B,C,D virtual PNO; Z,X,Y virtual semicanonical PNO
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            ii = i*self.no +i

            Fov_ii.append(self.H.F[o,v] @ Q[ii])
            Fvv_ij.append(Q[ij].T @ self.H.F[v,v] @ Q[ij])

            ERIoovo_ij.append(contract('ijak,aA->ijAk', self.H.ERI[o,o,v,o],Q[ij]))
            ERIooov_ij.append(contract('ijka,aA->ijkA', self.H.ERI[o,o,o,v],Q[ij]))
            ERIoovv_ij.append(contract('ijab,aA,bB->ijAB', self.H.ERI[o,o,v,v],Q[ij],Q[ij]))
            tmp = contract('iabc,aA->iAbc',self.H.ERI[o,v,v,v], Q[ij])
            tmp1 = contract('iAbc,bB->iABc',tmp, Q[ij])
            ERIovvv_ij.append(contract('iABc,cC->iABC',tmp1, Q[ij]))            

            Loovo_ij.append(contract('ijak,aA->ijAk', self.H.L[o,o,v,o],Q[ij]))
            Loovv_ij.append(contract('ijab,aA,bB->ijAB', self.H.L[o,o,v,v],Q[ij],Q[ij]))
            tmp = contract('iabc,aA->iAbc',self.H.L[o,v,v,v], Q[ij])
            tmp1 = contract('iAbc,bB->iABc',tmp, Q[ij])
            Lovvv_ij.append(contract('iABc,cC->iABC',tmp1, Q[ij]))
            Looov_ij.append(contract('iabc,cC->iabC',self.H.L[o,o,o,v], Q[ij]))
            Lovvo_ij.append(contract('iabj,aA,bB->iABj', self.H.L[o,v,v,o],Q[ij],Q[ij]))

        self.Fov_ii = Fov_ii
        self.Fvv_ij = Fvv_ij

        self.ERIoovo_ij = ERIoovo_ij
        self.ERIooov_ij = ERIooov_ij
        self.ERIovvv_ij = ERIovvv_ij 
        self.ERIvvvv_ij = ERIvvvv_ij
        self.ERIoovv_ij = ERIoovv_ij

        self.Loovv_ij = Loovv_ij 
        self.Lovvv_ij = Lovvv_ij
        self.Looov_ij = Looov_ij 
        self.Loovo_ij = Loovo_ij
        self.Lovvo_ij = Lovvo_ij


    def local_residuals(self, Fov_ii, t1, t2_ij):
        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        ERI = self.H.ERI
        Fae_ij = []

        r2_ij = []
        Fae_ij = self.build_localFae(o, v, self.Fvv_ij,Fov_ii, self.Lovvv_ij, self.Loovv_ij, t1, t2_ij)

        Fae = self.build_Fae(o, v, F, L, t1, t2_ij)
        Fmi = self.build_Fmi(o, v, F, L, t1, t2_ij)
        Fme = self.build_Fme(o, v, F, L, t1)

        r1 = self.r_T1(o, v, F, ERI, L, t1, t2_ij, Fae , Fme, Fmi)
        r2_ij = self.localr_T2(o, v, r2_ij,self.ERIoovv_ij, t2_ij, Fae_ij)
        return r1, r2_ij
    
    def build_Fae(self, o, v, F, L, t1, t2_ij):
        Fae = F[v,v].copy()
        #Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
        #Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
        #Fae = Fae - contract('mnaf,mnef->ae', self.build_tau(t1, t2_ij, 1.0, 0.5), L[o,o,v,v])
        return Fae

    def build_localFae(self, o, v, Fvv_ij,Fov_ii, Lovvv_ij, Loovv_ij, t1, t2_ij):
        Fae_ij = []
        Q = self.Local.Q
        L = self.Local.L
        for ij in range(self.no*self.no): 
            i = ij // self.no
            j = ij % self.no 
            ii = i*self.no + i 
    
            Fae = 0 
            Fae1 = 0 
            Fae2 = 0 
            Fae3 = 0 
            #first term of Fae 
            Fae = Fvv_ij[ij]

            #second term of Fae 
            for m in range(self.no):
                mm = m*self.no + m 
    
                #Sijmm = L[ij].T @ Q[ij].T @ Q[mm] @ L[mm]

                Sijmm = Q[ij].T @ Q[mm] 

                X = self.t1[m] @ self.Local.Q[mm]
                #Y = X @ self.Local.L[mm]    

                Fae1 -= 0.5* contract('eE,E,A,aA->ae',Sijmm,Fov_ii[mm][m],X,Sijmm)    
    
                #third term of Fae 
                #Fae2 += contract('F,afe,fF->ae',X,Lovvv_ij[ij][m],Sijmm)
                Fae2 += contract('F,AFE,eE,aA->ae',X, Lovvv_ij[mm][m],Sijmm,Sijmm)
    
                #fourth term of Fae 
                for n in range(self.no):
                    mn = m*self.no +n
                    Sijmn = Q[ij].T @ Q[mn] 
          
       #try for tau to have it in mn without any overlaps then just project the mn to ij here instead of in the tau function

                    Fae3 -= contract('aA,AF,EF,eE->ae',Sijmn, self.build_localtau(mn,mn,t1,t2_ij,1.0,0.5),Loovv_ij[mn][m,n],Sijmn)
            Fae_ij.append(Fae)#+ Fae1 + Fae2) # + Fae3)
        return Fae_ij 

    def build_Fmi(self, o, v, F, L, t1, t2_ij):
        Fmi = F[o,o].copy()
        Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
        Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
        Fmi = Fmi + contract('inef,mnef->mi', self.build_tau(t1, t2_ij, 1.0, 0.5), L[o,o,v,v])
        return Fmi


    def build_Fme(self, o, v, F, L, t1):
        Fme = F[o,v].copy()
        Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Fme



    def build_tau(self,t1,t2_ij,fact1=1.0, fact2=1.0):
        cooler = np.zeros((self.no, self.no, self.nv, self.nv))
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            cooler[i,j] = self.Local.Q[ij] @ t2_ij[ij] @ self.Local.Q[ij].T
        
        return fact1 * cooler

    def build_localtau(self,ij,mn,t1,t2_ij,fact1=1.0, fact2=1.0):
        return fact1 * t2_ij[ij]

    def r_T1(self, o, v, F, ERI, L, t1, t2_ij, Fae , Fme, Fmi):
        if self.model == 'CCD':
            r_T1 = np.zeros_like(t1)
        else:
            cooler = np.zeros((self.no,self.no,self.nv,self.nv))
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no

                cooler[i,j] = self.Local.Q[ij]  @ t2_ij[ij] @ self.Local.Q[ij].T
    
            r_T1 = F[o,v].copy()
            r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
            r_T1 = r_T1 - contract('ma,mi->ia', t1, Fmi)
            r_T1 = r_T1 + contract('imae,me->ia', (2.0*cooler - cooler.swapaxes(2,3)), Fme)
            r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
            r_T1 = r_T1 + contract('mief,maef->ia', (2.0*cooler - cooler.swapaxes(2,3)), ERI[o,v,v,v])
            r_T1 = r_T1 - contract('mnae,nmei->ia', cooler, L[o,o,v,o])
        return r_T1

    def localr_T2(self, o, v, r2_ij,ERIoovv_ij, t2_ij, Fae_ij): #, Fme_ii, Fmi):
        Q = self.Local.Q
        L = self.Local.L
        
        r2 = 0
        r2_1 = 0
        for ij in range(self.no*self.no):
            i = ij //self.no
            j = ij % self.no

            #first term
            r2 = 0.5 * ERIoovv_ij[ij][i,j]

            #second term
            r2_1 = contract('ae,be->ab', t2_ij[ij], Fae_ij[ij])           
            #r2_ij.append(tmp + contract('ae,be->ab', t2_ij[ij], Fae_ij[ij]))

            r2_ij.append(r2) # + r2_1)
        return r2_ij

    def localcc_energy(self,o,v,F,Loovv_ij,t1,t2_ij):
        ecc_ij = 0
        ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
        
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            
            #L_ij = 2.0 * self.t2_ij[ij] - self.t2_ij[ij].T
            #mp2_ij = np.sum(np.multiply(self.ERIoovv_ij[ij][i,j], L_ij))

            #ecc_ij = ecc_ij + contract('ab,ab->',t2_ij[ij],(2*self.ERIoovv_ij[ij][i,j] - self.ERIoovv_ij[ij][i,j].T))
            ecc_ij = np.sum(np.multiply(t2_ij[ij],Loovv_ij[ij][i,j]))             
            
            ecc += ecc_ij

        return ecc

