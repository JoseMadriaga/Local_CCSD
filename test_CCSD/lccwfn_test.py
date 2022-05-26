"""
ccwfn.py: CC T-amplitude Solver
This is where I am currently working on the implementation of local CCSD of the doubles where the singles are in the canonical form, the only timethe singles are transformed into the local form is where contracting within the doubles residuals ... vice versa for the doubles amplitude where the  
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


class lccwfn_test(object):
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

        valid_local_models = [None, 'PNO', 'PAO','PNO++']
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

        valid_init_t2 = [None,'OPT']
        init_t2 = kwargs.pop('init_t2', None)
        # TODO: case-protect this kwarg
        if init_t2 not in valid_init_t2:
            raise Exception("%s is not an allowed initial t2 amplitudes." % (init_t2))
        self.init_t2 = init_t2

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
            self.Local = Local(local, self.C, self.nfzc, self.no, self.nv, self.H, self.local_cutoff, self.init_t2)        
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

                L_ij = 2.0 * t2_ij[ij] - t2_ij[ij].T
                mp2_ij = np.sum(np.multiply(self.ERIoovv_ij[ij][i,j], L_ij))
                emp2 += mp2_ij

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

        print("mp2 energy in the local basis right before")    
        print(emp2)
                
        ecc = self.localcc_energy(o,v,F,self.Loovv_ij,self.t1,self.t2_ij)
        print("CC Iter %3d: CC Ecorr = %.15f dE = % .5E MP2" % (0,ecc,-ecc)) 
        for niter in range(1, maxiter+1):

            ecc_last = ecc  

            r1, r2_ij = self.local_residuals(self.Fov_ii, self.t1, self.t2_ij)
            
            self.t1 += r1/Dia
         
            rms = 0
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i*self.no +i 
                 
                #print(np.shape(self.t2_ij[ij]),np.shape(r2_ij[ij]))
                self.t2_ij[ij] -= r2_ij[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])
                                  
                rms += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])
            
            rms = np.sqrt(rms)
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
        ERIovvo_ij = []

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

            Fov_ii.append(self.H.F[o,v] @ Q[ii] @ L[ii])
            Fvv_ij.append(L[ij].T @ Q[ij].T @ self.H.F[v,v] @ Q[ij] @ L[ij])

            ERIoovo_ij.append(contract('ijak,aA,AZ->ijZk', self.H.ERI[o,o,v,o],Q[ij],L[ij]))
            ERIooov_ij.append(contract('ijka,aA,AZ->ijkZ', self.H.ERI[o,o,o,v],Q[ij],L[ij]))
            ERIoovv_ij.append(contract('ijab,aA,AZ,bB,BY->ijZY', self.H.ERI[o,o,v,v],Q[ij],L[ij],Q[ij],L[ij]))
            
            tmp = contract('iabc,aA,AZ->iZbc',self.H.ERI[o,v,v,v], Q[ij], L[ij])
            tmp1 = contract('iZbc,bB,BY->iZYc',tmp, Q[ij],L[ij])
            ERIovvv_ij.append(contract('iZYc,cC,CX->iZYX',tmp1, Q[ij], L[ij]))            

            tmp2 = contract('abcd,aA,AZ->Zbcd',self.H.ERI[v,v,v,v], Q[ij], L[ij])
            tmp3 = contract('Zbcd,bB,BY->ZYcd',tmp2, Q[ij], L[ij])
            tmp4 = contract('ZYcd,cC,CX->ZYXd',tmp3, Q[ij], L[ij])
            ERIvvvv_ij.append(contract('ZYXd,dD,DW->ZYXW',tmp4, Q[ij], L[ij]))         

            tmp5 = contract('iabj,aA,AZ->iZbj',self.H.ERI[o,v,v,o], Q[ij],L[ij]) 
            ERIovvo_ij.append(contract('iZbj,bB,BY->iZYj',tmp5,Q[ij], L[ij]))

            Loovo_ij.append(contract('ijak,aA,AZ->ijZk', self.H.L[o,o,v,o],Q[ij],L[ij]))
            Loovv_ij.append(contract('ijab,aA,AZ,bB,BY->ijZY', self.H.L[o,o,v,v],Q[ij],L[ij],Q[ij],L[ij]))
            tmp = contract('iabc,aA,AZ->iZbc',self.H.L[o,v,v,v], Q[ij], L[ij])
            tmp1 = contract('iZbc,bB,BY->iZYc',tmp, Q[ij],L[ij])
            Lovvv_ij.append(contract('iZYc,cC,CX->iZYX',tmp1, Q[ij], L[ij]))
            Looov_ij.append(contract('iabc,cC,CX->iabX',self.H.L[o,o,o,v], Q[ij],L[ij]))
            Lovvo_ij.append(contract('iabj,aA,AZ,bB,BY->iZYj', self.H.L[o,v,v,o],Q[ij],L[ij],Q[ij],L[ij]))

        self.Fov_ii = Fov_ii
        self.Fvv_ij = Fvv_ij

        self.ERIoovo_ij = ERIoovo_ij
        self.ERIooov_ij = ERIooov_ij
        self.ERIovvv_ij = ERIovvv_ij 
        self.ERIvvvv_ij = ERIvvvv_ij
        self.ERIoovv_ij = ERIoovv_ij
        self.ERIovvo_ij = ERIovvo_ij

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
        Wmbej_ij = []     
 
        r2_ij = []
        Fae_ij = self.build_localFae(o, v, self.Fvv_ij,Fov_ii, self.Lovvv_ij, self.Loovv_ij, t1, t2_ij)
        lFmi = self.build_lFmi(o, v, F, self.Fov_ii, self.Looov_ij, self.Loovv_ij, t1, t2_ij)

        Fae = self.build_Fae(o, v, F, L, t1, t2_ij)
        Fmi = self.build_Fmi(o, v, F, L, t1, t2_ij)
        Fme = self.build_Fme(o, v, F, L, t1)
        Wmnij = self.build_Wmnij(o, v, ERI, t1, t2_ij)
        Zmbij = self.build_Zmbij( o, v, ERI, t1, t2_ij)
        Wmbej_ij = self.build_Wmbej( o, v, self.ERIovvo_ij,self.ERIoovo_ij , self.ERIovvv_ij, L, t1, t2_ij)

        r1 = self.r_T1(o, v, F, ERI, L, t1, t2_ij, Fae , Fme, Fmi)
        r2_ij = self.localr_T2(o, v, r2_ij,self.ERIoovv_ij, self.ERIvvvv_ij, t2_ij, Fae_ij, Fmi, Fme,Wmnij, Zmbij, Wmbej_ij)
        return r1, r2_ij
    
    def build_Fae(self, o, v, F, L, t1, t2_ij):
        Fae = F[v,v].copy()
        Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
        Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
        Fae = Fae - contract('mnaf,mnef->ae', self.build_tau(t1, t2_ij, 1.0, 0.5), L[o,o,v,v])
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

            #For consistency at the moment, I have kept all the contraction with singles in the MO basis then transforming them 
            #to the semicanonical local basis ... but for Fae1 and Fae2, the local implementation works!!!
        
            #second term of Fae
            #for m in range(self.no):
                #mm = m*self.no + m

            tmp = 0
            tmp1 = 0
                #Sijmm = L[ij].T @ Q[ij].T @ Q[mm] @ L[mm]

                #X = self.t1[m] @ self.Local.Q[mm]
                #Y = X @ self.Local.L[mm]

            tmp =  0.5 * contract('me,ma->ae', self.H.F[o,v], self.t1)
            Fae1 -= L[ij].T @ Q[ij].T @ tmp @ Q[ij] @ L[ij]
                
                #Fae1 -= 0.5* contract('eE,E,A,aA->ae',Sijmm,Fov_ii[mm][m],Y,Sijmm)

 
                #third term of Fae (both ways work, need to know which is more efficient: the first one?)
                #Fae2 += contract('F,afe,fF->ae',Y,Lovvv_ij[ij][m],Sijmm)
                #Fae2 += contract('F,AFE,eE,aA->ae',Y, Lovvv_ij[mm][m],Sijmm,Sijmm)

            tmp1 = contract('mf,mafe->ae', self.t1, self.H.L[o,v,v,v])   
            Fae2 += L[ij].T @ Q[ij].T @ tmp1 @ Q[ij] @ L[ij]

            #fourth term of Fae
            for mn in range(self.no*self.no): 
                m = mn // self.no
                n = mn % self.no
                Sijmn = L[ij].T @ Q[ij].T @ Q[mn] @ L[mn]

                #try for tau to have it in mn without any overlaps then just project the mn to ij here instead of in the tau function
                Fae3 -= contract('aA,AF,EF,eE->ae',Sijmn, self.build_localtau(mn,mn,t1,t2_ij,1.0,0.5),Loovv_ij[mn][m,n],Sijmn)

            Fae_ij.append(Fae + Fae1 + Fae2)# + Fae3)
        return Fae_ij


    #should I make this local? 
    def build_Fmi(self, o, v, F, L, t1, t2):
        Fmi = F[o,o].copy()
        Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
        Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
        Fmi = Fmi + contract('inef,mnef->mi', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fmi


    def build_lFmi(self, o, v, F, Fov_ii, Looov_ij, Loovv_ij, t1, t2_ij):
        Q = self.Local.Q 
        L = self.Local.L
        tmp = F[o,o].copy()
   
        Fmi1 = np.zeros_like(tmp)
        Fmi2 = np.zeros_like(tmp)
        Fmi3 = np.zeros_like(tmp)

        for ij in range(self.no*self.no):
            i = ij // self.no 
            j = ij % self.no
            ii = i*self.no + i            
 
            for m in range(self.no):
                X = self.t1[i] @ Q[ii]
                Y = X @ L[ii] 
                Fmi1[m,i] += 0.5 * contract('e,e->',Y,Fov_ii[ii][m])
            
                for n in range(self.no):
                    mn = m *self.no + n
                    nn = n *self.no + n
                    X = self.t1[n] @ self.Local.Q[nn]
                    Y = X @ self.Local.L[nn]
                    Fmi2[m,i] += contract('e,e->',Y,Looov_ij[nn][m,n,i,:])
                    
                    in_pair = i*self.no + n  
                    Fmi3[m,i] += contract('EF,EF->', self.build_localtau(in_pair,in_pair,t1,t2_ij,1.0,0.5),Loovv_ij[in_pair][m,n,:,:]) 
         
        Fmi = tmp + Fmi1 + Fmi2 + Fmi3
        return Fmi

    def build_Fme(self, o, v, F, L, t1):
        Fme = F[o,v].copy()
        Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Fme

    def build_Wmnij(self, o, v, ERI, t1, t2_ij):
        Wmnij = ERI[o,o,o,o].copy()
        Wmnij = Wmnij + contract('je,mnie->mnij', t1, ERI[o,o,o,v])
        Wmnij = Wmnij + contract('ie,mnej->mnij', t1, ERI[o,o,v,o])
        Wmnij = Wmnij + contract('ijef,mnef->mnij', self.build_tau(t1, t2_ij), ERI[o,o,v,v])
        return Wmnij

    def build_Zmbij(self, o, v, ERI, t1, t2_ij): 
        return contract('mbef,ijef->mbij', ERI[o,v,v,v], self.build_tau(t1, t2_ij))

    def build_Wmbej(self, o, v,ERIovvo_ij, ERIoovo_ij,ERIovvv_ij, L, t1, t2_ij):
        Q = self.Local.Q
        L = self.Local.L
        Wmbej_ij = []  
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            jj = j*self.no + j
            
            Y = contract('jf,fF,FZ->jZ',t1,Q[jj],L[jj])        
     
            Sijjj = L[ij].T @ Q[ij].T @ Q[jj] @ L[jj]
            Wmbej = ERIovvo_ij[ij]

            Wmbej1 = contract('jF,mbef,fF->mbej',Y, ERIovvv_ij[ij],Sijjj)
            #Wmbej1 = contract('jF,mBEF,bB,eE->mbej',Y, ERIovvv_ij[ij],Sijjj,Sijjj)

            Wmbej2 = np.zeros_like(Wmbej)
            Wmbej3 = np.zeros_like(Wmbej)
            t2_nj = np.zeros_like(Wmbej)
            for n in range(self.no):
                nn = n*self.no + n
                jn = j*self.no + n
                nj = n*self.no + j
        
                Sijnn = L[ij].T @ Q[ij].T @ Q[nn] @ L[nn]
                
                tmp = contract('nb,bB,BY->nY',t1,Q[nn],L[nn])
                                                      
                Wmbej2 -= contract('B,mej,bB->mbej',tmp[n], ERIoovo_ij[ij][:,n,:,:],Sijnn)   

                Sijjn = L[ij].T @ Q[ij].T @ Q[jn] @ L[jn]

                #incorrect since tau is in pair jn which j is a target index ... don't know how to contract
                #Wmbej3 -= contract('FB,mEF,eE,bB->mbe', self.build_localtau(jn,jn,t1, t2_ij, 0.5, 1.0), ERIoovv_ij[jn][:,n,:,:],Sijjn,Sijjn)
                #Wmbej3 -= contract('FB,mef,fF,bB->mbe', self.build_localtau(jn,jn,t1, t2_ij, 0.5, 1.0), ERIoovv_ij[ij][:,n,:,:],Sijjn,Sijjn)
               
                Sijnj = L[ij].T @ Q[ij].T @ Q[nj] @ L[nj]
                
                #same here as well t2 is in a pair nj
                #Wmbej4 += 0.5 * contract('FB,mnEF,bB,eE->mnbe', t2_ij[nj], Loovv_ij[ij],Sijnj,Sijnj)

            Wmbej_ij.append(Wmbej) # + Wmbej1 + Wmbej2)
        return Wmbej_ij


    def build_tau(self,t1,t2_ij,fact1=1.0, fact2=1.0):
        cooler = np.zeros((self.no, self.no, self.nv, self.nv))
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            cooler[i,j] = self.Local.Q[ij] @ self.Local.L[ij] @ t2_ij[ij] @ self.Local.L[ij].T @ self.Local.Q[ij].T
        
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

                cooler[i,j] = self.Local.Q[ij] @ self.Local.L[ij] @ t2_ij[ij] @ self.Local.L[ij].T @ self.Local.Q[ij].T
    
            r_T1 = F[o,v].copy()
            r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
            r_T1 = r_T1 - contract('ma,mi->ia', t1, Fmi)
            r_T1 = r_T1 + contract('imae,me->ia', (2.0*cooler - cooler.swapaxes(2,3)), Fme)
            r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
            r_T1 = r_T1 + contract('mief,maef->ia', (2.0*cooler - cooler.swapaxes(2,3)), ERI[o,v,v,v])
            r_T1 = r_T1 - contract('mnae,nmei->ia', cooler, L[o,o,v,o])
        return r_T1

    def localr_T2(self, o, v, r2_ij,ERIoovv_ij, ERIvvvv_ij, t2_ij, Fae_ij,Fmi,Fme, Wmnij, Zmbij, Wmbej_ij): #, Fme_ii, Fmi):
        Q = self.Local.Q
        L = self.Local.L
        
        r2 = 0
        r2_1 = 0
        r2_2 = 0
        r2_3 = 0
        r2_4 = 0
        r2_5 = 0
        r2_6 = 0
        r2_7 = 0
        r2_8 = 0
        r2_9 = 0
        #t2_mo = np.zeros((self.no, self.no, self.nv, self.nv))
          
        #for ij in range(self.no*self.no):
            #i = ij // self.no
            #j = ij % self.no
 
            #t2_mo[i,j] = self.Local.Q[ij] @ self.Local.L[ij] @ t2_ij[ij] @ self.Local.L[ij].T @ self.Local.Q[ij].T
        #tmp = - contract('imab,mj->ijab', t2_mo, Fmi)               
        
        #part of third term 
        tmp = contract('mb,me->be', self.t1, Fme)

        #part of tenth term
        tmp1 = - contract('ma,mbij->ijab', self.t1, Zmbij)
  
        #sixth term
        r2_5 = contract('je,me->jm', self.t1, Fme)

        for ij in range(self.no*self.no):
            i = ij //self.no
            j = ij % self.no
  
            #first term - add
            r2 = 0.5 * ERIoovv_ij[ij][i,j]

            #second term - add
            r2_1 = contract('ae,be->ab', t2_ij[ij], Fae_ij[ij])           
            #r2_ij.append(tmp + contract('ae,be->ab', t2_ij[ij], Fae_ij[ij]))

            #third term 
            r2_2 = L[ij].T @ Q[ij].T @ tmp @ Q[ij] @ L[ij]

            #fourth term -add
            r2_3  = - 0.5 * contract('ae,be->ab', t2_ij[ij], r2_2)

            #nineth term -add
            r2_8 = 0.5 * contract('ef,abef->ab', self.build_localtau(ij,ij,self.t1, t2_ij), ERIvvvv_ij[ij]) 
        
            #tenth term - add
            #r2_9 = np.zeros_like(r2_1)
            #r2_9 =  L[ij].T @ Q[ij].T @ tmp1[i,j] @ Q[ij] @ L[ij]

            #fifth term -add
            r2_4 = np.zeros_like(r2_1)
            r2_6 = np.zeros_like(r2_1)
            r2_7 = np.zeros_like(r2_1)
            r2_10 = np.zeros_like(r2_1)
            for m in range(self.no):
                im = i*self.no +m

                Sijim = L[ij].T @ Q[ij].T @ Q[im] @ L[im]
                cool = contract('aA,AB,bB->ab',Sijim, t2_ij[im], Sijim)
                r2_4 -= Fmi[m,j] * cool

            #r2_4 =  L[ij].T @ Q[ij].T @ tmp @ Q[ij] @ L[ij]               

                #seventh term - add  
                r2_6 -= 0.5 * r2_5[j,m] * cool
              
                #part of eleventh term 
                #r2_10 += contract('ae,be->ab', (cool - cool.swapaxes(0,1)), Wmbej_ij[ij][m,:,:,j])
  
                #eight term -add
                for n in range(self.no):
                    mn = m *self.no + n
                    Sijmn = L[ij].T @ Q[ij].T @ Q[mn] @ L[mn] 
                    r2_7 += 0.5 * Wmnij[m,n,i,j] * contract('aA,AB,bB->ab',Sijmn,self.build_localtau(mn,mn,self.t1,t2_ij),Sijmn)   

            r2_ij.append(r2 + r2_1 + r2_3  + r2_4 + r2_6 + r2_7 + r2_8) # + r2_10) #  + r2_9
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

