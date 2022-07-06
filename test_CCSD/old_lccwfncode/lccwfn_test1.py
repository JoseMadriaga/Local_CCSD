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
            t1_ii = []
            t2_ij = []
            emp2 = 0

            for i in range(self.no):
                ii = i*self.no + i
 
                X = self.Local.Q[ii].T @ self.t1[i]
                t1_ii.append(self.Local.L[ii].T @ X)

                for j in range(self.no):
                    ij = i*self.no+ j
 
                    X = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.H.ERI[i,j,v,v] @ self.Local.Q[ij] @ self.Local.L[ij] 
                    t2_ij.append( -1*X/ (self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])) 

                    L_ij = 2.0 * t2_ij[ij] - t2_ij[ij].T
                    mp2_ij = np.sum(np.multiply(self.ERIoovv_ij[ij][i,j], L_ij))
                    emp2 += mp2_ij

            print("mp2 energy in the local basis")    
            print(emp2)
            self.t1_ii = t1_ii
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
                
        ecc = self.lcc_energy(self.Fov_ij,self.Loovv_ij,self.t1_ii,self.t2_ij)
        print("CC Iter %3d: CC Ecorr = %.15f dE = % .5E MP2" % (0,ecc,-ecc)) 
        for niter in range(1, maxiter+1):

            ecc_last = ecc  

            r1_ii, r2_ij = self.local_residuals(self.t1_ii, self.t2_ij)
         
            rms = 0

            for i in range(self.no):
                ii = i*self.no + i 
                
                for a in range(self.Local.dim[ii]):
                    self.t1_ii[i][a] += r1_ii[i][a]/(self.H.F[i,i] - self.Local.eps[ii][a])               
                
                rms += contract('Z,Z->',r1_ii[i], r1_ii[i]) 

                for j in range(self.no):
                    ij = i*self.no + j

                    self.t2_ij[ij] -= r2_ij[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])
                    rms += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])
            
            rms = np.sqrt(rms)
            ecc = self.lcc_energy(self.Fov_ij,self.Loovv_ij,self.t1_ii,self.t2_ij)
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

        #contraction notation i,j,a,b typically MO; A,B,C,D virtual PNO; Z,X,Y virtual semicanonical PNO
        
        Fov_ij = []
        Fvv_ij = []

        ERIoovo_ij = []
        ERIooov_ij = []
        ERIovvv_ij = []
        ERIvvvv_ij = []
        ERIoovv_ij = []
        ERIovvo_ij = []
        ERIvvvo_ij = []
        ERIovov_ij = []        
        ERIovoo_ij = [] 

        Loovv_ij = []
        Lovvv_ij = []
        Looov_ij = [] 
        Loovo_ij = []
        Lovvo_ij = [] 

        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            Fov_ij.append(self.H.F[o,v] @ Q[ij] @ L[ij])
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
            tmp6 = contract('abci,aA,AZ->Zbci',self.H.ERI[v,v,v,o], Q[ij], L[ij]) 
            tmp7 = contract('Zbci,bB,BY->ZYci',tmp6, Q[ij], L[ij])
            ERIvvvo_ij.append(contract('ZYci,cC,CX->ZYXi',tmp7, Q[ij], L[ij]))
            tmp8 = contract('iajb,aA,AZ->iZjb',self.H.ERI[o,v,o,v], Q[ij], L[ij])
            ERIovov_ij.append(contract('iZjb,bB,BY->iZjY', tmp8, Q[ij], L[ij]))
            ERIovoo_ij.append(contract('iajk,aA,AZ->iZjk', self.H.ERI[o,v,o,o], Q[ij], L[ij]))

            Loovo_ij.append(contract('ijak,aA,AZ->ijZk', self.H.L[o,o,v,o],Q[ij],L[ij]))
            Loovv_ij.append(contract('ijab,aA,AZ,bB,BY->ijZY', self.H.L[o,o,v,v],Q[ij],L[ij],Q[ij],L[ij]))
            tmp9 = contract('iabc,aA,AZ->iZbc',self.H.L[o,v,v,v], Q[ij], L[ij])
            tmp10 = contract('iZbc,bB,BY->iZYc',tmp, Q[ij],L[ij])
            Lovvv_ij.append(contract('iZYc,cC,CX->iZYX',tmp1, Q[ij], L[ij]))
            Looov_ij.append(contract('ijka,aA,AZ->ijkZ',self.H.L[o,o,o,v], Q[ij],L[ij]))
            Lovvo_ij.append(contract('iabj,aA,AZ,bB,BY->iZYj', self.H.L[o,v,v,o],Q[ij],L[ij],Q[ij],L[ij]))

        self.Fov_ij = Fov_ij
        self.Fvv_ij = Fvv_ij

        self.ERIoovo_ij = ERIoovo_ij
        self.ERIooov_ij = ERIooov_ij
        self.ERIovvv_ij = ERIovvv_ij 
        self.ERIvvvv_ij = ERIvvvv_ij
        self.ERIoovv_ij = ERIoovv_ij
        self.ERIovvo_ij = ERIovvo_ij
        self.ERIvvvo_ij = ERIvvvo_ij
        self.ERIovov_ij = ERIovov_ij 
        self.ERIovoo_ij = ERIovoo_ij      

        self.Loovv_ij = Loovv_ij 
        self.Lovvv_ij = Lovvv_ij
        self.Looov_ij = Looov_ij 
        self.Loovo_ij = Loovo_ij
        self.Lovvo_ij = Lovvo_ij

    def local_residuals(self, t1_ii, t2_ij):
        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        ERI = self.H.ERI 

        Fae_ij = []
        Fme_ij = []
        #Fme_im = []

        Wmbej_ij = []
        Wmbje_ij = []
        Zmbij_ij = []        

        r1_ii = []
        r2_ij = []
       
        Fae_ij = self.build_lFae(Fae_ij, self.Fvv_ij,self.Fov_ij, self.Lovvv_ij, self.Loovv_ij, t1_ii, t2_ij)
        lFmi = self.build_lFmi(o, F, self.Fov_ij, self.Looov_ij, self.Loovv_ij, t1_ii, t2_ij)
        Fme_ij = self.build_lFme(Fme_ij, self.Fov_ij, self.Loovv_ij, t1_ii)
        lWmnij = self.build_lWmnij(o, ERI, self.ERIooov_ij, self.ERIoovo_ij, self.ERIoovv_ij, t1_ii, t2_ij)
        Zmbij = self.build_lZmbij(Zmbij_ij, self.ERIovvv_ij, t1_ii, t2_ij)
        Wmbej_ij = self.build_lWmbej(Wmbej_ij, self.ERIoovv_ij, self.ERIovvo_ij, self.ERIovvv_ij, self.ERIoovo_ij, self.Loovv_ij, t1_ii, t2_ij)
        Wmbje_ij = self.build_lWmbje(Wmbje_ij, self.ERIovov_ij, self.ERIovvv_ij, self.ERIoovv_ij, self.ERIooov_ij, t1_ii, t2_ij)        
        r1_ii = self.lr_T1(r1_ii, self.Fov_ij , self.ERIovvv_ij, self.Lovvo_ij, self.Loovo_ij, t1_ii, t2_ij, Fae_ij , Fme_ij, lFmi)
        r2_ij = self.lr_T2(r2_ij,self.ERIoovv_ij, self.ERIvvvv_ij, self.ERIovvo_ij, self.ERIovoo_ij, self.ERIvvvo_ij, self.ERIovov_ij, t1_ii, t2_ij, Fae_ij,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ij, Wmbje_ij)

        return r1_ii, r2_ij

    def build_lFae(self, Fae_ij, Fvv_ij,Fov_ij, Lovvv_ij, Loovv_ij, t1_ii, t2_ij):
        Q = self.Local.Q
        L = self.Local.L

        Fae = 0
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            Fae = Fvv_ij[ij] 

            Fae_1 = np.zeros_like(Fae)
            Fae_2 = np.zeros_like(Fae) 
            Fae_3 = np.zeros_like(Fae)
            for m in range(self.no):
                mm = m*self.no + m

                Sijmm = L[ij].T @ Q[ij].T @ Q[mm] @ L[mm]

                tmp = Sijmm @ t1_ii[m]
                Fae_1 -= 0.5* contract('e,a->ae',Fov_ij[ij][m],tmp)                
                
                tmp1 = contract('afe,fF->aFe',Lovvv_ij[ij][m],Sijmm)
                
                Fae_2 += contract('F,aFe->ae',t1_ii[m],tmp1)
                #if ij == 0:
                    #print("updating Fae_1", m)
                    #print(Fae_1)
                    #print("updating Fae_2", m)
                    #print(Fae_2)
                for n in range(self.no):
                    mn = m *self.no +n 
               
                    Sijmn = L[ij].T @ Q[ij].T @ Q[mn] @ L[mn] 
                   
                    tmp2 = contract('aA,AF->aF', Sijmn, self.build_ltau(mn,t1_ii,t2_ij,1.0,0.5))
                    # may need to construct the integrals to appropriate pair instead of projecting to one
                    tmp3 = contract('ef,fF->eF', Loovv_ij[ij][m,n], Sijmn)
                    Fae_3 -= contract('aF,eF->ae', tmp2, tmp3)
                    #if ij == 0:
                        #print("updating Fae_3", m, n)
                        #print(Fae_3)
            Fae_ij.append(Fae + Fae_1 + Fae_2 + Fae_3)   
        return Fae_ij 
              
    def build_lFmi(self, o, F, Fov_ij, Looov_ij, Loovv_ij, t1_ii, t2_ij):

        Q = self.Local.Q 
        L = self.Local.L

        Fmi = F[o,o].copy()
   
        Fmi_1 = np.zeros_like(Fmi)
        Fmi_2 = np.zeros_like(Fmi)
        Fmi_3 = np.zeros_like(Fmi)
        for i in range(self.no):
            ii = i*self.no + i            
 
            for m in range(self.no):

                Fmi_1[m,i] += 0.5 * contract('e,e->', t1_ii[i], Fov_ij[ii][m])
                #print("updating Fm1_1", m, i )
                #print(Fmi_1[m,i])
                #print(Fmi_1)
                for n in range(self.no):
                    mn = m *self.no + n
                    nn = n *self.no + n
                    _in = i*self.no + n 
                                         
                    Fmi_2[m,i] += contract('e,e->',t1_ii[n],Looov_ij[nn][m,n,i])

                    Fmi_3[m,i] += contract('EF,EF->',self.build_ltau(_in,t1_ii,t2_ij,1.0,0.5),Loovv_ij[_in][m,n,:,:]) 
                   
                    #print("updating Fmi_2", m, n)
                    #print(Fmi_2[m,i])
                    #print(Fmi_2)
                    #print("updating Fmi_3", m, n)
                    #print(Fmi_3[m,i])
                    #print(Fmi_3)

        Fmi_tot = Fmi + Fmi_1 + Fmi_2 + Fmi_3
        return Fmi_tot

    def build_lFme(self, Fme_ij, Fov_ij, Loovv_ij, t1_ii):
        Q = self.Local.Q
        L = self.Local.L

        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no 
            
            Fme = np.zeros((self.no,self.Local.dim[ij]))             
            Fme_1 = np.zeros_like(Fme)
            for m in range(self.no):
       
                Fme[m] = Fov_ij[ij][m]
                #if ij == 0:       
                    #print("updating Fme", m)
                    #print(Fme[m])
                    #print(Fme)
                for n in range(self.no):
                    nn = n*self.no + n
                                                       
                    Sijnn = L[ij].T @ Q[ij].T @ Q[nn] @ L[nn]
                    
                    #may need to construct integrals to appropriate pair instead of projecting to one
                    tmp = contract('fF,ef->eF',Sijnn, Loovv_ij[ij][m,n])
                    Fme_1[m] += contract('F,eF->e',t1_ii[n],tmp)
                    
                    #if ij == 0:
                        #print("updating Fme_1", m, n, ij)
                        #print(Fme_1[m])
                        #print(Fme_1)               
            Fme_ij.append(Fme + Fme_1)
        return Fme_ij

    def build_lWmnij(self, o, ERI, ERIooov_ij, ERIoovo_ij, ERIoovv_ij, t1_ii, t2_ij):
        Wmnij = ERI[o,o,o,o].copy()
 
        Wmnij_1 = np.zeros_like(Wmnij)
        Wmnij_2 = np.zeros_like(Wmnij)
        Wmnij_3 = np.zeros_like(Wmnij)
        for i in range(self.no):
            for j in range(self.no):
                jj = j *self.no + j
                ij = i*self.no + j
                ii = i*self.no + i 

                for m in range(self.no):
                    for n in range(self.no):
 
                        Wmnij_1[m,n,i,j] += contract('E,E->', t1_ii[j], ERIooov_ij[jj][m,n,i])
                        Wmnij_2[m,n,i,j] += contract('E,E->', t1_ii[i], ERIoovo_ij[ii][m,n,:,j])
                        Wmnij_3[m,n,i,j] += contract('ef,ef->',self.build_ltau(ij,t1_ii, t2_ij), ERIoovv_ij[ij][m,n])       
                        #if ij == 0:
                            #print("Updating Wmnij_1", m, n, i, j)
                            #print(Wmnij_1[:,:,i,j])
                            #print("Updating Wmnij_2", m, n, i, j)
                            #print(Wmnij_2[:,:,i,j])
                            #print("Updating Wmnij_3", m, n, i, j)
                            #print(Wmnij_3[:,:,i,j])
        Wmnij_tot = Wmnij + Wmnij_1 + Wmnij_2 + Wmnij_3 
        return Wmnij_tot

    def build_lZmbij(self, Zmbij_ij, ERIovvv_ij, t1_ii, t2_ij): 
        """
        Lots of zeros in these tensors since it's running over a specific i and j element therefore only really storing a matrix m by b_ij
        """
        for i in range(self.no):
            for j in range(self.no):
                ij = i*self.no+ j
                Zmbij = np.zeros((self.no,self.Local.dim[ij],self.no,self.no)) 
                for m in range(self.no):
                    Zmbij[m,:,i,j] += contract('bef,ef->b', ERIovvv_ij[ij][m], self.build_ltau(ij,t1_ii,t2_ij))
                    #if ij == 0:
                        #print("updating Zmbij", m)
                        #print(Zmbij[m])
                        #print(self.Local.dim[ij])
                        #print(Zmbij[:,:,i,j])
                Zmbij_ij.append(Zmbij)
        return Zmbij_ij
               
    def build_lWmbej(self, Wmbej_ij, ERIoovv_ij, ERIovvo_ij, ERIovvv_ij, ERIoovo_ij, Loovv_ij, t1_ii, t2_ij):
        """
        Lots of zeros in these tensors since it's running over a specific i and j element therefore only really storing a rank 3 tensor m by b_ij by e_ij  
        """
        Q = self.Local.Q
        L = self.Local.L

        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no 
            jj = j*self.no + j  
           
            Wmbej = ERIovvo_ij[ij]
  
            Wmbej_1 = np.zeros_like(Wmbej) 
            Wmbej_2 = np.zeros_like(Wmbej)
            Wmbej_3 = np.zeros_like(Wmbej)
            Wmbej_4 = np.zeros_like(Wmbej)
            for m in range(self.no):
                im = i*self.no + m 
  
                Sijjj = L[ij].T @ Q[ij].T @ Q[jj] @ L[jj]

                #may need to construct integrals to appropriate pair instead of projecting to one
                tmp = contract('bef,fF->beF',ERIovvv_ij[ij][m],Sijjj)
                Wmbej_1[m,:,:,j] += contract('F,beF->be', t1_ii[j], tmp)
                               
                #print("updating Wmbej_1", m, j, ij)
                #print(Wmbej_1[m,:,:,j])
                #print(Wmbej_1[:,:,:,j])
                
                for n in range(self.no):
                    nn = n*self.no + n 
                    jn = j*self.no + n 
                    nj = n*self.no + j
 
                    Sijnn = L[ij].T @ Q[ij].T @ Q[nn] @ L[nn]

                    tmp1 = Sijnn @ t1_ii[n] 
                    Wmbej_2[m,:,:,j] -= contract('b,e->be',tmp1,ERIoovo_ij[ij][m,n,:,j]) 
 
                    Sijjn = L[ij].T @ Q[ij].T @ Q[jn] @ L[jn]
            
                    tmp2 = self.build_ltau(jn,t1_ii,t2_ij, 0.5, 1.0) @ Sijjn.T 
                    #may need to construct integra;s to appropriate pair instead of projecting to one
                    tmp3 = contract('ef,fF->eF',ERIoovv_ij[ij][m,n],Sijjn)
                    Wmbej_3[m,:,:,j] -= contract('Fb,eF->be', tmp2, tmp3)

                    Sijnj = L[ij].T @ Q[ij].T @ Q[nj] @ L[nj]
                    
                    tmp4 = t2_ij[nj] @ Sijnj.T 
                    #may need to construct integra;s to appropriate pair instead of projecting to one
                    tmp5 = contract('ef,fF->eF',Loovv_ij[ij][m,n],Sijnj)
                    Wmbej_4[m,:,:,j] += 0.5 * contract('Fb,eF->be', tmp4, tmp5)
                    
                    
            Wmbej_ij.append(Wmbej + Wmbej_1 + Wmbej_2 + Wmbej_3 + Wmbej_4)
        return Wmbej_ij

    def build_lWmbje(self, Wmbje_ij, ERIovov_ij, ERIovvv_ij, ERIoovv_ij, ERIooov_ij, t1_ii, t2_ij):
        Q = self.Local.Q
        L = self.Local.L 

        for ij in range(self.no*self.no):
            i = ij // self.no 
            j = ij % self.no 
            jj = j*self.no + j 
 
            Wmbje = -1.0 * ERIovov_ij[ij] 

            Wmbje_1 = np.zeros_like(Wmbje)
            Wmbje_2 = np.zeros_like(Wmbje)
            Wmbje_3 = np.zeros_like(Wmbje)
            for m in range(self.no):
                Sijjj = L[ij].T @ Q[ij].T @ Q[jj] @ L[jj]
 
                #may need to construct integra;s to appropriate pair instead of projecting to one
                tmp = contract('bfe,fF->bFe',ERIovvv_ij[ij][m],Sijjj) 
                Wmbje_1[m,:,j,:] -= contract('f,bFe->be', t1_ii[j], tmp)
                
                for n in range(self.no):
                    nn = n*self.no + n
                    jn = j*self.no + n

                    Sijnn = L[ij].T @ Q[ij].T @ Q[nn] @ L[nn]

                    tmp1 = Sijnn @ t1_ii[n]
                    Wmbje_2[m,:,j,:] += contract('b,e->be',tmp1,ERIooov_ij[ij][m,n,j])

                    Sijjn = L[ij].T @ Q[ij].T @ Q[jn] @ L[jn]
                    
                    tmp2 = self.build_ltau(jn,t1_ii,t2_ij, 0.5, 1.0) @ Sijjn.T
                    #may need to construct integrals to appropriate pair instead of projecting to one 
                    tmp3 = contract('fe,fF->Fe',ERIoovv_ij[ij][m,n],Sijjn)
                    Wmbje_3[m,:,j,:] += contract('Fb,Fe->be', tmp2, tmp3)
             
            Wmbje_ij.append( Wmbje + Wmbje_1 + Wmbje_2 + Wmbje_3)
        return Wmbje_ij

    def build_ltau(self,ij,t1_ii,t2_ij,fact1=1.0, fact2=1.0):
        i = ij // self.no
        j = ij % self.no
        ii = i*self.no + i
        jj = j*self.no + j
  
        Sijii = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.Local.Q[ii] @ self.Local.L[ii]
        Sijjj = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.Local.Q[jj] @ self.Local.L[jj] 
        
        tmp = Sijii @ t1_ii[i]
        tmp1 = t1_ii[j] @ Sijjj.T       
        return fact1 * t2_ij[ij] + fact1 * contract('a,b->ab',tmp,tmp1)

    def lr_T1(self, r1_ii, Fov_ij , ERIovvv_ij, Lovvo_ij, Loovo_ij, t1_ii, t2_ij, Fae_ij , Fme_ij, lFmi):
        Q = self.Local.Q
        L = self.Local.L 

        r1 = 0 
        r1_1 = 0
        for i in range(self.no):
            ii = i*self.no + i 
 
            r1 = Fov_ij[ii][i]
            r1_1 = contract('e,ae->a', t1_ii[i], Fae_ij[ii])
     
            r1_2 = np.zeros_like(r1)
            r1_3 = np.zeros_like(r1) 
            r1_5 = np.zeros_like(r1)
            for m in range(self.no):
                mm = m*self.no + m

                Siimm = L[ii].T @ Q[ii].T @ Q[mm] @ L[mm]

                tmp =  Siimm @ t1_ii[m]
                r1_2 -= tmp * lFmi[m,i] 

                im = i*self.no + m
                Siiim = L[ii].T @ Q[ii].T @ Q[im] @ L[im]
 
                tmp1 = Siiim @ (2*t2_ij[im] - t2_ij[im].swapaxes(0,1))
                #may need to construct the components of the Wmbje_ij to appropriate pairs such that b is pair ij while e is pair im
                #since the projection of one pair for the "dressed" integrals introduces too much error
                r1_3 += contract('aE,E->a',tmp1, Fme_ij[im][m])
                
                mi = m*self.no + i 
                Siimi = L[ii].T @ Q[ii].T @ Q[mi] @ L[mi]

                #may need to construct integrals to appropriate pair instead of projecting to one
                tmp2 = contract('aef,eE,fF->aEF',ERIovvv_ij[ii][m],Siimi,Siimi) 
                r1_5 += contract('EF,aEF->a', (2.0*t2_ij[mi] - t2_ij[mi].swapaxes(0,1)), tmp2)

            r1_4 = np.zeros_like(r1)
            for n in range(self.no):
                nn = n*self.no + n 
                     
                Siinn = L[ii].T @ Q[ii].T @ Q[nn] @ L[nn]

                #may need to construct integrals to appropriate pair instead of projecting to one
                r1_4 += contract('F,af,fF->a', t1_ii[n], Lovvo_ij[ii][n,:,:,i],Siinn)
     
            r1_6 = np.zeros_like(r1)              
            for mn in range(self.no*self.no):
                m = mn // self.no 
                n = mn % self.no
                Siimn = L[ii].T @ Q[ii].T @ Q[mn] @ L[mn]

                tmp3 = Siimn @ t2_ij[mn] 
                r1_6 -= contract('aE,E->a',tmp3,Loovo_ij[mn][n,m,:,i])
                #if ii == 0: 
                    #print("Updating r1_6", m, n)
                    #print(r1_6)            

            r1_ii.append(r1 + r1_1 + r1_2 + r1_3 + r1_4 + r1_5 + r1_6)
     
        return r1_ii

    def lr_T2(self,r2_ij,ERIoovv_ij, ERIvvvv_ij, ERIovvo_ij, ERIovoo_ij, ERIvvvo_ij, ERIovov_ij, t1_ii, t2_ij, Fae_ij,lFmi,Fme_ij, lWmnij, Zmbij_ij, Wmbej_ij, Wmbje_ij):
        Q = self.Local.Q
        L = self.Local.L
        
        r2 = 0
        r2_1one = 0
        r2_4 = 0
        for ij in range(self.no*self.no):
            i = ij //self.no
            j = ij % self.no
            ii = i*self.no + i 
            jj = j*self.no + j
       
            r2 = 0.5 * ERIoovv_ij[ij][i,j]

            r2_1one = contract('ae,be->ab', t2_ij[ij], Fae_ij[ij])           

            r2_4 = 0.5 * contract('ef,abef->ab',self.build_ltau(ij,t1_ii,t2_ij),ERIvvvv_ij[ij])
 
            r2_1two = np.zeros_like(r2)
            r2_2one = np.zeros_like(r2)
            r2_2two = np.zeros_like(r2)
            r2_3 = np.zeros_like(r2)
            r2_5 = np.zeros_like(r2)
            r2_6 = np.zeros_like(r2) 
            r2_7 = np.zeros_like(r2)
            r2_8 = np.zeros_like(r2)
            r2_9 = np.zeros_like(r2)
            r2_10 = np.zeros_like(r2) 
            r2_11 = np.zeros_like(r2)
            r2_12 = np.zeros_like(r2)

            for m in range(self.no):
                mm = m *self.no + m
                Sijmm = L[ij].T @ Q[ij].T @ Q[mm] @ L[mm]
                
                tmp = Sijmm @ t1_ii[m]   
                tmp1 = contract('b,e->be', tmp, Fme_ij[ij][m])
                r2_1two -= 0.5* contract('ae,be->ab', t2_ij[ij], tmp1)

                im = i*self.no + m 
                Sijim = L[ij].T @ Q[ij].T @ Q[im] @ L[im]
      
                tmp2 = Sijim @ t2_ij[im] @ Sijim.T           
                r2_2one -= tmp2 * lFmi[m,j]

                tmp3 = contract('E,E->',t1_ii[j], Fme_ij[jj][m])
                r2_2two -= 0.5 * tmp2 * tmp3
                
                r2_5 -= contract('a,b->ab', Zmbij_ij[ij][m,:,i,j],tmp)

                tmp5 = Sijim @ (t2_ij[im] - t2_ij[im].swapaxes(0,1))
                #may need to construct integrals to appropriate pair instead of projecting to one
                tmp6 = contract('be,eE->bE',Wmbej_ij[ij][m,:,:,j],Sijim)
                r2_6 += contract('aE,bE->ab',tmp5, tmp6) 
                
                #may need to construct the components of the Wmbje_ij and Wmbej_ij to appropriate pairs such that b is pair ij while e is pair im 
                #since the projection of one pair for the "dressed" integrals introduces too much error

                tst = Wmbje_ij[ij].swapaxes(2,3) 
                tmp7 = contract('be,eE->bE',(Wmbej_ij[ij][m,:,:,j] + tst[m,:,:,j]),Sijim)
                r2_7 += contract('aE,bE->ab', tmp5, tmp7)

                mj = m*self.no + j 
                Sijmj = L[ij].T @ Q[ij].T @ Q[mj] @ L[mj]
                
                tmp8 = Sijmj @ t2_ij[mj] 
                tmp9 = contract('be,eE->bE',Wmbje_ij[ij][m,:,i,:],Sijmj) 
                r2_8 += contract('aE,bE->ab',tmp8,tmp9)
 
                Sijii = L[ij].T @ Q[ij].T @ Q[ii] @ L[ii]
                
                #may need to construct integrals to appropriate pair instead of projecting to one
                tmp10 = contract('be,eE->bE',ERIovvo_ij[ij][m,:,:,j],Sijii)
                tmp11 = contract ('E,a->Ea', t1_ii[i],tmp)
                r2_9 -= contract('Ea,bE->ab',tmp11, tmp10)
                
                #may need to construct integrals to appropriate pair instead of projecting to one
                tmp12 = contract('ae,eE->aE', ERIovov_ij[ij][m,:,j,:],Sijii)
                r2_10 -= contract('Eb,aE-> ab', tmp11,tmp12)

                #may need to construct integrals to appropriate pair instead of projecting to one
                tmp13 = contract('abe,eE->abE', ERIvvvo_ij[ij][:,:,:,j], Sijii)
                r2_11 += contract('E,abE->ab', t1_ii[i], tmp13) 

                r2_12 -= contract('a,b -> ab', tmp, ERIovoo_ij[ij][m,:,i,j])
        
                for n in range(self.no): 
                    mn = m*self.no + n
                    
                    Sijmn = L[ij].T @ Q[ij].T @ Q[mn] @ L[mn]
                    
                    tmp4 = Sijmn @ self.build_ltau(mn,t1_ii,t2_ij) @ Sijmn.T 
                    r2_3 += 0.5 * tmp4 * lWmnij[m,n,i,j]
                    #if ij == 0:
                        #print("updating r2_3", m,n) 
                        #print(r2_3)       

            r2_ij.append(r2 + r2_1one + r2_1two + r2_2one + r2_2two + r2_3 + r2_4 + r2_5 + r2_6 + r2_7 + r2_8 + r2_9 + r2_10 + r2_11 + r2_12)
        return r2_ij

    def lcc_energy(self,Fov_ij,Loovv_ij,t1_ii,t2_ij):
        ecc_ij = 0
        ecc_ii = 0
        ecc = 0
        
        for i in range(self.no):
            ii = i*self.no + i

            ecc_ii = np.sum(np.multiply(Fov_ij[ii][i],t1_ii[i])) 
            ecc += ecc_ii 
            
            for j in range(self.no):
                ij = i*self.no + j 
            
                ecc_ij = np.sum(np.multiply(t2_ij[ij],Loovv_ij[ij][i,j]))             
                ecc += ecc_ij

        return ecc

