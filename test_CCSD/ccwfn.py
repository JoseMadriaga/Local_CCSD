"""
ccwfn.py: CC T-amplitude Solver
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


class ccwfn(object):
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
            t1_ij = []
            t2_ij = []
            emp2 = 0
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i * self.no + i
                
                #t1_i.append(contract('ia,aA,AZ->iZ',self.t1,self.Local.Q[ij],self.Local.L[ij]))
                X = self.t1[o] @ self.Local.Q[ij]
                Y = X @ self.Local.L[ij]
                t1_ij.append(Y/ -1*(self.Local.eps[ii].reshape(1,-1) - self.H.F[i,i]))
                #print(t1_ij[ij])

                X = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.H.ERI[i,j,v,v] @ self.Local.Q[ij] @ self.Local.L[ij] 
                t2_ij.append( -1*X/ (self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])) 
                L_ij = 2.0 * t2_ij[ij] - t2_ij[ij].T
                mp2_ij = np.sum(np.multiply(self.ERIoovv_ij[ij][i,j], L_ij))
                emp2 += mp2_ij

            print("mp2 energy in the local basis")    
            print(emp2)
            self.t1_ij = t1_ij
            self.t2_ij = t2_ij
            #print("t2_ij")
            #print(np.shape(t2_ij))
            #self.t1, self.t2 = self.Local.filter_amps(self.t1, self.H.ERI[o,o,v,v])
        
        else:
            self.t1 = np.zeros((self.no, self.nv))
            self.t2 = self.H.ERI[o,o,v,v]/self.Dijab

        #print("mp2 energy with truncation")
        #print(contract('ijab,ijab->', self.t2 , self.H.L[o,o,v,v]))
        #print("cc iter 0th")
        #print(self.cc_energy(o, v, self.H.F, self.H.L, self.t1, self.t2))
 
        print("CC object initialized in %.3f seconds." % (time.time() - time_init))

    def solve_localcc(self, e_conv=1e-7,r_conv=1e-7,maxiter=1,max_diis=8,start_diis=8):
        o = self.o 
        v = self.v
        
        ecc = self.localcc_energy(o,v,self.Fov_ij,self.Loovv_ij,self.t1_ij,self.t2_ij)
        print("CC Iter %3d: CC Ecorr = %.15f dE = % .5E MP2" % (0,ecc,-ecc)) 
        for niter in range(1, maxiter+1):

            ecc_last = ecc  

            r1_i, r2_ij = self.local_residuals(self.Fov_ij, self.t1_ij, self.t2_ij)
            
            rms = 0
            cool = np.zeros((self.no,self.nv))
            coolest = np.zeros((self.no,self.nv))
            cooler = np.zeros((self.no, self.no, self.nv, self.nv))
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i*self.no +i 
                self.t1_ij[ij] -= r1_i[ij]/(self.Local.eps[ij].reshape(1,-1) - self.H.F[i,i])
                self.t2_ij[ij] -= r2_ij[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])
                  
                tmp = r1_i[ij] @ self.Local.L[ij].T
                cool[o] =  tmp @ self.Local.Q[ij].T
                
                cooler[i,j] = self.Local.Q[ij] @ self.Local.L[ij] @ r2_ij[ij] @ self.Local.L[ij].T @ self.Local.Q[ij].T 

                tmp1 = self.t1_ij[ij] @ self.Local.L[ij].T 
                coolest[o] = tmp1 @ self.Local.Q[ij].T 

                rms += contract('iZ,iZ->',r1_i[ij],r1_i[ij])
                rms += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])
            
            rms = np.sqrt(rms)
            #print("cool") 
            #print(cool)
            #print("cooler")
            #print(cooler)
            print("coolest")
            print(coolest)           
            ecc = self.localcc_energy(o,v,self.Fov_ij,self.Loovv_ij,self.t1_ij,self.t2_ij)
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
   
    def solve_cc(self, e_conv=1e-7, r_conv=1e-7, maxiter=100, max_diis=8, start_diis=8):
        """
        Parameters
        ----------
        e_conv : float
            convergence condition for correlation energy (default if 1e-7)
        r_conv : float
            convergence condition for wave function rmsd (default if 1e-7)
        maxiter : int
            maximum allowed number of iterations of the CC equations (default is 100)
        max_diis : int
            maximum number of error vectors in the DIIS extrapolation (default is 8; set to 0 to deactivate)
        start_diis : int
            earliest iteration to start DIIS extrapolations (default is 1)

        Returns
        -------
        ecc : float
            CC correlation energy
        """
        ccsd_tstart = time.time()

        o = self.o
        v = self.v
        F = self.H.F
        L = self.H.L
        Dia = self.Dia
        Dijab = self.Dijab

        ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
        print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  MP2" % (0, ecc, -ecc))

        diis = helper_diis(self.t1, self.t2, max_diis)

        for niter in range(1, maxiter+1):

            ecc_last = ecc

            r1, r2 = self.residuals(F, self.t1, self.t2)

            if self.local is not None:
                print("in the solve_cc")
                inc1, inc2 = self.Local.filter_amps(r1, r2)
                self.t1 += inc1
                self.t2 += inc2
                rms = contract('ia,ia->', inc1, inc1)
                rms += contract('ijab,ijab->', inc2, inc2)
                rms = np.sqrt(rms)
            else:
                self.t1 += r1/Dia
                self.t2 += r2/Dijab
                rms = contract('ia,ia->', r1/Dia, r1/Dia)
                rms += contract('ijab,ijab->', r2/Dijab, r2/Dijab)
                rms = np.sqrt(rms)

            ecc = self.cc_energy(o, v, F, L, self.t1, self.t2)
            ediff = ecc - ecc_last
            print("CC Iter %3d: CC Ecorr = %.15f  dE = % .5E  rms = % .5E" % (niter, ecc, ediff, rms))

            # check for convergence
            if ((abs(ediff) < e_conv) and rms < r_conv):
                print("\nCC has converged in %.3f seconds.\n" % (time.time() - ccsd_tstart))
                print("E(REF)  = %20.15f" % self.eref)
                print("E(%s) = %20.15f" % (self.model, ecc))
                print("E(TOT)  = %20.15f" % (ecc + self.eref))
                self.ecc = ecc
                return ecc

            diis.add_error_vector(self.t1, self.t2)
            if niter >= start_diis:
                self.t1, self.t2 = diis.extrapolate(self.t1, self.t2)
    
    def transform_integral(self,o,v):
        
        Q = self.Local.Q
        L = self.Local.L
        Fov_ij = []
        Fvv_ij = []
        
        ERIoovo_ij = []
        ERIooov_ij = []
        ERIovvv_ij = []
        ERIvvvv_ij = []
        ERIoovv_ij = []

        Loovv_ij = []
        Lovvv_ij = []
        Looov_ij = []
         
        #contraction notation i,j,a,b typically MO; A,B,C,D virtual PNO; Z,X,Y virtual semicanonical PNO
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            ii = i*self.no + i 
            
           # first trying out ij with o and v slices, then ij with specific occupied indices then ii or jj with o and v slices then ii or jj with specific occupied indices
            Fov_ij.append(self.H.F[o,v] @ Q[ij] @ L[ij])
            Fvv_ij.append(L[ij].T @ Q[ij].T @ self.H.F[v,v] @ Q[ij] @ L[ij])

            ERIoovo_ij.append(contract('ijak,aA,AZ->ijZk', self.H.ERI[o,o,v,o],Q[ij],L[ij]))
            ERIooov_ij.append(contract('ijka,aA,AZ->ijkZ', self.H.ERI[o,o,o,v],Q[ij],L[ij]))
            ERIoovv_ij.append(contract('ijab,aA,AZ,bB,BY->ijZY', self.H.ERI[o,o,v,v],Q[ij],L[ij],Q[ij],L[ij]))
            Loovv_ij.append(contract('ijab,aA,AZ,bB,BY->ijZY', self.H.L[o,o,v,v],Q[ij],L[ij],Q[ij],L[ij]))
            tmp = contract('iabc,aA,AZ->iZbc',self.H.L[o,v,v,v], Q[ij], L[ij])
            tmp1 = contract('iZbc,bB,BY->iZYc',tmp, Q[ij],L[ij])
            Lovvv_ij.append(contract('iZYc,cC,CX->iZYX',tmp1, Q[ij], L[ij]))
            Looov_ij.append(contract('iabc,cC,CX->iabX',self.H.L[o,o,o,v], Q[ij],L[ij]))

        self.Fov_ij = Fov_ij
        #print(Fov_ij[0][0])
        self.Fvv_ij = Fvv_ij
        self.ERIoovo_ij = ERIoovo_ij
        #print(np.shape(ERIoovo_ij))
        self.ERIooov_ij = ERIooov_ij
        self.ERIovvv_ij = ERIovvv_ij 
        self.ERIvvvv_ij = ERIvvvv_ij
        self.ERIoovv_ij = ERIoovv_ij
        self.Loovv_ij = Loovv_ij 
        #print(np.shape(Loovv_ij))
        self.Lovvv_ij = Lovvv_ij
        self.Looov_ij = Looov_ij 
                   
    def local_residuals(self, Fov_ij, t1_ij, t2_ij):
        o = self.o
        v = self.v
        Fae_ij = []
        Fme_ij = []
        Fmi_ij = []
        Wmnij_ij = []
        Wmbej_ij = []
        Wmbje_ij = []
        Zmbij_ij = []
        r1_i = []
        r2_ij = []

        Fae_ij = self.build_localFae(o, v, self.Fvv_ij,Fov_ij, self.Lovvv_ij, self.Loovv_ij, t1_ij, t2_ij)
        #Fmi_ij = self.build_localFmi(o, v, F, L, t1_ij, t2_ij)
        #Fme_ij = self.build_localFme(o, v, F, L, t1_ij)
        #Wmnij = self.build_localWmnij(o, v, ERI, t1_ij, t2_ij)
        #Wmbej_ij = self.build_localWmbej(o, v, ERI, L, t1_ij, t2_ij)
        #Wmbje_ij = self.build_localWmbje(o, v, ERI, t1_ij, t2_ij)
        #Zmbij_ij = self.build_localZmbij(o, v, ERI, t1_ij, t2_ij)

        r1_i = self.localr_T1(o,v, r1_i,Fov_ij,t1_ij,Fae_ij) #, Fme_ij, Fmi)
        r2_ij = self.localr_T2(o, v, r2_ij,self.ERIoovv_ij, t2_ij, Fae_ij) #, Fme_ij, Fmi, Wmnij, Wmbej_ij, Wmbje_ij, Zmbij_ij) 

        return r1_i, r2_ij
    
    def residuals(self, F, t1, t2):
        """
        Parameters
        ----------
        F: NumPy array
            Fock matrix
        t1: NumPy array
            Current T1 amplitudes
        t2: NumPy array
            Current T2 amplitudes

        Returns
        -------
        r1, r2: NumPy arrays
            New T1 and T2 residuals: r_mu = <mu|HBAR|0>
        """

        o = self.o
        v = self.v
        ERI = self.H.ERI
        L = self.H.L
        
        Fae = self.build_Fae(o, v, F, L, t1, t2)
        #Fmi = self.build_Fmi(o, v, F, L, t1, t2)
        #Fme = self.build_Fme(o, v, F, L, t1)
        #Wmnij = self.build_Wmnij(o, v, ERI, t1, t2)
        #Wmbej = self.build_Wmbej(o, v, ERI, L, t1, t2)
        #Wmbje = self.build_Wmbje(o, v, ERI, t1, t2)
        #Zmbij = self.build_Zmbij(o, v, ERI, t1, t2)

        r1 = self.r_T1(o, v, F, ERI, L, t1, t2, Fae) #, Fme, Fmi)
        r2 = self.r_T2(o, v, F, ERI, L, t1, t2, Fae) #, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)

        return r1, r2

    def build_localtau(self,ij,t1_ij,t2_ij,fact1=1.0, fact2=1.0):
        i = ij // self.no
        j = ij % self.no
        ii = i*self.no + i
        jj = j*self.no + j

        #print(np.shape(fact1 * t2_ij[ij] + fact2 * contract('Z,Y->ZY',t1_ij[ij][i],t1_ij[ij][j])))

        return fact1 * t2_ij[ij] + fact2 * contract('Z,Y->ZY',t1_ij[ij][i],t1_ij[ij][j])
                 
    def build_tau(self, t1, t2, fact1=1.0, fact2=1.0):
        return fact1 * t2 + fact2 * contract('ia,jb->ijab', t1, t1)

    def build_localFae(self, o, v, Fvv_ij,Fov_ij, Lovvv_ij, Loovv_ij, t1_ij, t2_ij):
        Fae_ij = []
        Q = self.Local.Q
        L = self.Local.L
        for ij in range(self.no*self.no): 
            i = ij // self.no
            j = ij % self.no 
            ii = i *self.no + i 
            
            #Fae = 0
            #first term of Fae
            #Fae = Fvv_ij[ij]
            Fae_ij.append(Fvv_ij[ij]) 
            #second term of Fae
          
            #for m in range(self.no):
                
                #mm = m*self.no + m 
                #Si_m = L[ii].T @ Q[ii].T @ Q[mm] @ L[mm] 
                #tmp = Si_m @ t1_ij[ij][m]
                #print("shape of Si_m @ t1_ij[ij][m]") 
                #print(np.shape(tmp))
                #print("Fov_ij[ij][m]")
                #print(np.shape(Fov_ij[ij][m]))
            #Fae -= 0.5* contract('mZ,mY->ZY',Fov_ij[ij],t1_ij[ij])
                #Fae_ij.append( Fae - 0.5* contract('Z,Y->ZY',Fov_ij[ij][m],t1_ij[ij][m]))             
                #third term of Fae 
                #for f in range(self.no):
                    #ff = f*self.no + f 
                    #Si_f = L[ii].T @ Q[ii].T @ Q[ff] @ L[ff]
                    #tmp1 = Si_f @ t1_ij[ij][f]
            #Fae += contract('fW,fZWY->ZY',t1_ij[ij],Lovvv_ij[ij])
                    
                    #fourth term of Fae
            #for m in range(self.no):
                #for n in range(self.no):
                    #mn = m*self.no + n
                    #Sij_mn = L[ij].T @ Q[ij].T @ Q[mn] @ L[mn]
                    #tmp1 = contract('rs,ar,bs->ab',self.build_localtau(ij,t1_ij,t2_ij,1.0,0.5),Sij_mn,Sij_mn) 
                    #Fae_ij.append(Fae - contract('ZW,YW->ZY',tmp1,Loovv_ij[ij][m,n]))         
        return Fae_ij 
            
    def build_Fae(self, o, v, F, L, t1, t2):    
        if self.model == 'CCD':
            Fae = F[v,v].copy()
            Fae = Fae - contract('mnaf,mnef->ae', t2, L[o,o,v,v])
        else:
            Fae = F[v,v].copy()
            Fae = Fae - 0.5 * contract('me,ma->ae', F[o,v], t1)
            Fae = Fae + contract('mf,mafe->ae', t1, L[o,v,v,v])
            Fae = Fae - contract('mnaf,mnef->ae', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fae

    def build_localFmi(self,o,v):
    # since it is Foo no need to have this in an ij form 
        
        #first term
        Fmi = F[o,o].copy()
        for ij in range(self.no*self.no):
            i = ij //self.no
            j = ij % self.no
            
            #second term
            Fmi += 0.5 * contract('iZ,mZ->mi', t1_i[ij], Fov_ij[ij])                           
            
            #third term                
            for n in range(self.no): 
                nn = n*self.no + n
                Si_n = L[ii].T @ Q[ii].T @ Q[nn] @ L[nn]
                tmp[i] = Si_n @ t1_ij[ij][n]                
                #Fmi += contract('Z,
        return Fmi 

    def build_Fmi(self, o, v, F, L, t1, t2):
        if self.model == 'CCD':
            Fmi = F[o,o].copy()
            Fmi = Fmi + contract('inef,mnef->mi', t2, L[o,o,v,v])
        else:
            Fmi = F[o,o].copy()
            Fmi = Fmi + 0.5 * contract('ie,me->mi', t1, F[o,v])
            Fmi = Fmi + contract('ne,mnie->mi', t1, L[o,o,o,v])
            Fmi = Fmi + contract('inef,mnef->mi', self.build_tau(t1, t2, 1.0, 0.5), L[o,o,v,v])
        return Fmi

    def build_Fme(self, o, v, F, L, t1):
        if self.model == 'CCD':
            return
        else:
            Fme = F[o,v].copy()
            Fme = Fme + contract('nf,mnef->me', t1, L[o,o,v,v])
        return Fme

    def build_Wmnij(self, o, v, ERI, t1, t2):
        if self.model == 'CCD':
            Wmnij = ERI[o,o,o,o].copy()
            Wmnij = Wmnij + contract('ijef,mnef->mnij', t2, ERI[o,o,v,v])
        else:
            Wmnij = ERI[o,o,o,o].copy()
            Wmnij = Wmnij + contract('je,mnie->mnij', t1, ERI[o,o,o,v])
            Wmnij = Wmnij + contract('ie,mnej->mnij', t1, ERI[o,o,v,o])
            Wmnij = Wmnij + contract('ijef,mnef->mnij', self.build_tau(t1, t2), ERI[o,o,v,v])
        return Wmnij

    def build_Wmbej(self, o, v, ERI, L, t1, t2):
        if self.model == 'CCD':
            Wmbej = ERI[o,v,v,o].copy()
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', 0.5*t2, ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        else:
            Wmbej = ERI[o,v,v,o].copy()
            Wmbej = Wmbej + contract('jf,mbef->mbej', t1, ERI[o,v,v,v])
            Wmbej = Wmbej - contract('nb,mnej->mbej', t1, ERI[o,o,v,o])
            Wmbej = Wmbej - contract('jnfb,mnef->mbej', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
            Wmbej = Wmbej + 0.5 * contract('njfb,mnef->mbej', t2, L[o,o,v,v])
        return Wmbej

    def build_Wmbje(self, o, v, ERI, t1, t2):
        if self.model == 'CCD':
            Wmbje = -1.0 * ERI[o,v,o,v].copy()
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', 0.5*t2, ERI[o,o,v,v])
        else:
            Wmbje = -1.0 * ERI[o,v,o,v].copy()
            Wmbje = Wmbje - contract('jf,mbfe->mbje', t1, ERI[o,v,v,v])
            Wmbje = Wmbje + contract('nb,mnje->mbje', t1, ERI[o,o,o,v])
            Wmbje = Wmbje + contract('jnfb,mnfe->mbje', self.build_tau(t1, t2, 0.5, 1.0), ERI[o,o,v,v])
        return Wmbje

    def build_Zmbij(self, o, v, ERI, t1, t2):
        if self.model == 'CCD':
            return
        else:
            return contract('mbef,ijef->mbij', ERI[o,v,v,v], self.build_tau(t1, t2))

    def localr_T1(self,o,v,r1_i,Fov_ij,t1_ij,Fae_ij):
        for ij in range(self.no*self.no):
            tmp = Fov_ij[ij].copy()
            #print("Fov_ij[ij]")
            #print(np.shape(tmp))
            #print("Fae_ij[ij]")
            #print(np.shape(Fae_ij[ij]))
            r1_i.append(tmp + contract('ie,ae->ia',t1_ij[ij],Fae_ij[ij]))
        #print("r1_i")
        #print(np.shape(r1_i)) 
        return r1_i

    def r_T1(self, o, v, F, ERI, L, t1, t2, Fae): #, Fme, Fmi):
        if self.model == 'CCD':
            r_T1 = np.zeros_like(t1)
        else:
            r_T1 = F[o,v].copy()
            r_T1 = r_T1 + contract('ie,ae->ia', t1, Fae)
            #r_T1 = r_T1 - contract('ma,mi->ia', t1, Fmi)
            #r_T1 = r_T1 + contract('imae,me->ia', (2.0*t2 - t2.swapaxes(2,3)), Fme)
            #r_T1 = r_T1 + contract('nf,nafi->ia', t1, L[o,v,v,o])
            #r_T1 = r_T1 + contract('mief,maef->ia', (2.0*t2 - t2.swapaxes(2,3)), ERI[o,v,v,v])
            #r_T1 = r_T1 - contract('mnae,nmei->ia', t2, L[o,o,v,o])
        return r_T1

    def localr_T2(self, o, v, r2_ij,ERIoovv_ij, t2_ij, Fae_ij):
        for ij in range(self.no*self.no):
            i = ij //self.no
            j = ij % self.no

            #r2_ij.append(0.5 * ERIoovv_ij[ij][i,j])
            tmp = 0.5 * ERIoovv_ij[ij][i,j]

            r2_ij.append(tmp + contract('ae,be->ab', t2_ij[ij], Fae_ij[ij]))
        #print("r2_ij")
        #print(np.shape(r2_ij))
        return r2_ij
            
    def r_T2(self, o, v, F, ERI, L, t1, t2, Fae): #, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):
        if self.model == 'CCD':
            r_T2 = 0.5 * ERI[o,o,v,v].copy()
            r_T2 = r_T2 + contract('ijae,be->ijab', t2, Fae)
            r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
            r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', t2, Wmnij)
            r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', t2, ERI[v,v,v,v])
            r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
            r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
            r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
        else:
            r_T2 = 0.5 * ERI[o,o,v,v].copy()
            r_T2 = r_T2 + contract('ijae,be->ijab', t2, Fae)
            #tmp = contract('mb,me->be', t1, Fme)
            #r_T2 = r_T2 - 0.5 * contract('ijae,be->ijab', t2, tmp)
            #r_T2 = r_T2 - contract('imab,mj->ijab', t2, Fmi)
            #tmp = contract('je,me->jm', t1, Fme)
            #r_T2 = r_T2 - 0.5 * contract('imab,jm->ijab', t2, tmp)
            #r_T2 = r_T2 + 0.5 * contract('mnab,mnij->ijab', self.build_tau(t1, t2), Wmnij)
            #r_T2 = r_T2 + 0.5 * contract('ijef,abef->ijab', self.build_tau(t1, t2), ERI[v,v,v,v])
            #r_T2 = r_T2 - contract('ma,mbij->ijab', t1, Zmbij)
            #r_T2 = r_T2 + contract('imae,mbej->ijab', (t2 - t2.swapaxes(2,3)), Wmbej)
            #r_T2 = r_T2 + contract('imae,mbej->ijab', t2, (Wmbej + Wmbje.swapaxes(2,3)))
            #r_T2 = r_T2 + contract('mjae,mbie->ijab', t2, Wmbje)
            #tmp = contract('ie,ma->imea', t1, t1)
            #r_T2 = r_T2 - contract('imea,mbej->ijab', tmp, ERI[o,v,v,o])
            #r_T2 = r_T2 - contract('imeb,maje->ijab', tmp, ERI[o,v,o,v])
            #r_T2 = r_T2 + contract('ie,abej->ijab', t1, ERI[v,v,v,o])
            #r_T2 = r_T2 - contract('ma,mbij->ijab', t1, ERI[o,v,o,o])

        r_T2 = r_T2 #+ r_T2.swapaxes(0,1).swapaxes(2,3)
        return r_T2

    def localcc_energy(self,o,v,Fov_ij,Loovv_ij,t1_ij,t2_ij):
        ecc_ij = 0
        ecc = 0
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            
            #print(np.shape(Fov_ij[ij]))
            #print(Fov_ij[ij])
            #print(np.shape(t1_i[ij]))
            #print(t1_i[ij][0])

            ecc_ij = 2.0 * contract('Z,Z->',Fov_ij[ij][i],t1_ij[ij][i])

            #print(ecc_ij)
            #print(np.shape(Loovv_ij[ij])) 

            #ecc_ij += contract('ZY,ZY->', self.build_localtau(ij,t1_ij,t2_ij),Loovv_ij[ij])
            
            ecc_ij += np.sum(np.multiply(self.build_localtau(ij,t1_ij,self.t2_ij),Loovv_ij[ij][i,j]))
            
            #print(ecc_ij) 
 
            ecc += ecc_ij

        return ecc

    def cc_energy(self, o, v, F, L, t1, t2):
        if self.model == 'CCD':
            ecc = contract('ijab,ijab->', t2, L[o,o,v,v])
        else:
            ecc = 2.0 * contract('ia,ia->', F[o,v], t1)
            singles = ecc
            #print("singles value")
            #print(singles)
            ecc = ecc + contract('ijab,ijab->', self.build_tau(t1, t2), L[o,o,v,v])
            #print("doubles")
            #print(ecc-singles)
        return ecc
