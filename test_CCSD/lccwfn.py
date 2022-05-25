"""
ccwfn.py: CC T-amplitude Solver
This contain my implementation of both the singles and doubles but I am currently working on lccwfn_test.py to work on doubles ... 
will come back to here once the doubles looks good 
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


class lccwfn(object):
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
        with open("t1_amps.out","r") as f1:
            content = f1.readlines()
            for line in content:
                words = line.split()
                self.t1[int(words[0]),int(words[1])] = float(words[2])

        if local is not None:
            t1_ii = []
            t2_ij = []
            emp2 = 0
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i * self.no + i
                
                #this is for all occ 
                X = self.t1[o] @ self.Local.Q[ii]
                Y = X @ self.Local.L[ii]
                t1_ii.append(Y)           
 
                #changing this to i and see if it fixes the problem for r1 
                X = self.t1[i] @ self.Local.Q[ii]
                Y = X @ self.Local.L[ii]
                t1_ii.append(Y)

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
        
        ecc = self.localcc_energy(o,v,self.Fov_ii,self.Loovv_ij,self.t1_ii,self.t2_ij)
        print("CC Iter %3d: CC Ecorr = %.15f dE = % .5E MP2" % (0,ecc,-ecc)) 
        for niter in range(1, maxiter+1):

            ecc_last = ecc  

            r1_i, r2_ij = self.local_residuals(self.Fov_ii, self.t1_ii, self.t2_ij)
            
            rms = 0
            cool = np.zeros((self.no,self.nv))
            coolest = np.zeros((self.no,self.nv))
            cooler = np.zeros((self.no, self.no, self.nv, self.nv))
            for ij in range(self.no*self.no):
                i = ij // self.no
                j = ij % self.no
                ii = i*self.no +i 

                # this two works ( i think but trying out something with the initial t1 amplitudes) , just need to see which one would be faster
                #self.t1_ii[ii] -= r1_i[ii]/(self.Local.eps[ii].reshape(1,-1) - self.H.F[i,i])
                #for a in range(self.Local.dim[ii]):
                    #self.t1_ii[ii][o,a] += r1_i[ii][o,a]/(self.H.F[i,i] - self.Local.eps[ii][a])
         
                #this is my attempt at fixing the r1
                for occ in range(self.no): 
                    for a in range(self.Local.dim[ii]):
                        self.t1_ii[ii][occ,a] += r1_i[ii][occ,a]/(self.H.F[occ,occ] - self.Local.eps[ii][a])
                
                self.t2_ij[ij] -= r2_ij[ij]/(self.Local.eps[ij].reshape(1,-1) + self.Local.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j])
                  
                tmp = r1_i[ii] @ self.Local.L[ii].T
                cool[o] =  tmp @ self.Local.Q[ii].T
                
                cooler[i,j] = self.Local.Q[ij] @ self.Local.L[ij] @ r2_ij[ij] @ self.Local.L[ij].T @ self.Local.Q[ij].T 

                tmp1 = self.t1_ii[ii] @ self.Local.L[ii].T 
                coolest[o] = tmp1 @ self.Local.Q[ii].T 

                rms += contract('iZ,iZ->',r1_i[ii],r1_i[ii])
                rms += contract('ZY,ZY->',r2_ij[ij],r2_ij[ij])
            
            rms = np.sqrt(rms)
            #print("cool") 
            #print(cool)
            #print("cooler")
            #print(cooler)
            #print("coolest")
            #print(coolest)           
            ecc = self.localcc_energy(o,v,self.Fov_ii,self.Loovv_ij,self.t1_ii,self.t2_ij)
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

            Fov_ii.append(self.H.F[o,v] @ Q[ii] @ L[ii])
            Fvv_ij.append(L[ij].T @ Q[ij].T @ self.H.F[v,v] @ Q[ij] @ L[ij])

            ERIoovo_ij.append(contract('ijak,aA,AZ->ijZk', self.H.ERI[o,o,v,o],Q[ij],L[ij]))
            ERIooov_ij.append(contract('ijka,aA,AZ->ijkZ', self.H.ERI[o,o,o,v],Q[ij],L[ij]))
            ERIoovv_ij.append(contract('ijab,aA,AZ,bB,BY->ijZY', self.H.ERI[o,o,v,v],Q[ij],L[ij],Q[ij],L[ij]))
            tmp = contract('iabc,aA,AZ->iZbc',self.H.ERI[o,v,v,v], Q[ij], L[ij])
            tmp1 = contract('iZbc,bB,BY->iZYc',tmp, Q[ij],L[ij])
            ERIovvv_ij.append(contract('iZYc,cC,CX->iZYX',tmp1, Q[ij], L[ij]))            

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

        self.Loovv_ij = Loovv_ij 
        self.Lovvv_ij = Lovvv_ij
        self.Looov_ij = Looov_ij 
        self.Loovo_ij = Loovo_ij
        self.Lovvo_ij = Lovvo_ij
        print(np.shape(Loovo_ij[0]))           
        
    def local_residuals(self, Fov_ii, t1_ii, t2_ij):
        o = self.o
        v = self.v
        Fae_ij = []
        Fme_ii = []
    
 
        Wmbej_ij = []
        Wmbje_ij = []
        Zmbij_ij = []
        r1_i = []
        r2_ij = []

        Fae_ij = self.build_localFae(o, v, self.Fvv_ij,Fov_ii, self.Lovvv_ij, self.Loovv_ij, t1_ii, t2_ij)
        Fmi = self.build_localFmi(o, v, Fov_ii, self.Loovv_ij, self.Looov_ij, t1_ii)
        Fme_ii = self.build_localFme(o, v, Fov_ii, self.Loovv_ij, t1_ii)
        #Wmnij = self.build_localWmnij(o, v, ERI, t1_ij, t2_ij)
        #Wmbej_ij = self.build_localWmbej(o, v, ERI, L, t1_ij, t2_ij)
        #Wmbje_ij = self.build_localWmbje(o, v, ERI, t1_ij, t2_ij)
        #Zmbij_ij = self.build_localZmbij(o, v, ERI, t1_ij, t2_ij)

        r1_i = self.localr_T1(o,v, r1_i,Fov_ii,t1_ii,Fae_ij,Fme_ii, Fmi,self.Lovvo_ij,t2_ij) #, Fme_ij, Fmi)
        r2_ij = self.localr_T2(o, v, r2_ij,self.ERIoovv_ij, t2_ij, Fae_ij,Fmi, Fme_ii) #, Fme_ii, Fmi, Wmnij, Wmbej_ij, Wmbje_ij, Zmbij_ij) 

        return r1_i, r2_ij
    
    def build_localtau(self,ij,mn,t1_ii,t2_ij,fact1=1.0, fact2=1.0):
        i = ij // self.no
        j = ij % self.no
        ii = i *self.no + i
        jj = j*self.no +j 
        m = mn //self.no
        n = mn % self.no
        mm = m*self.no +m 
        nn = n*self.no +n

        if (mn != ij):
            Smnmm = self.Local.L[mn].T @ self.Local.Q[mn].T @ self.Local.Q[mm] @ self.Local.L[mm]
            Smnnn = self.Local.L[mn].T @ self.Local.Q[mn].T @ self.Local.Q[nn] @ self.Local.L[nn]
 
            tmp = contract('aA,A,F,fF->af',Smnmm, t1_ii[mm][m],t1_ii[nn][n],Smnnn)          
            return fact1 * t2_ij[mn] + fact2 * tmp 
            #fact1 * tmp
        else:

        #only need this and thats it 
            Sijii = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.Local.Q[ii] @ self.Local.L[ii]
            Sijjj = self.Local.L[ij].T @ self.Local.Q[ij].T @ self.Local.Q[jj] @ self.Local.L[jj]
            return fact1 * t2_ij[ij] + fact2 * contract('aA,A,F,fF->af',Sijii,t1_ii[ii][m],t1_ii[jj][n],Sijjj)
             #fact1 * t2_ij[ij]

    def build_localFae(self, o, v, Fvv_ij,Fov_ii, Lovvv_ij, Loovv_ij, t1_ii, t2_ij):
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
       
                Sijmm = L[ij].T @ Q[ij].T @ Q[mm] @ L[mm]
                Fae1 -= 0.5* contract('eE,E,A,aA->ae',Sijmm,Fov_ii[mm][m],t1_ii[mm][m],Sijmm)           
                
                #third term of Fae 
                Fae2 += contract('F,afe,fF->ae',t1_ii[mm][m],Lovvv_ij[ij][m],Sijmm)
            #Fae_ij.append(Fae + Fae1 + Fae2)       
                #fourth term of Fae
                for n in range(self.no):
                    mn = m*self.no +n
                    Sijmn = L[ij].T @ Q[ij].T @ Q[mn] @ L[mn]
       
       #try for tau to have it in mn without any overlaps then just project the mn to ij here instead of in the tau function

                    Fae3 -= contract('aA,AF,EF,eE->ae',Sijmn, self.build_localtau(mn,mn,t1_ii,t2_ij,1.0,0.5),Loovv_ij[mn][m,n],Sijmn)
            Fae_ij.append(Fae) # + Fae1 + Fae2 + Fae3)
        return Fae_ij 
    
    def build_localFmi(self, o, v, Fov_ii, Loovv_ij, Looov_ij,t1_ii):
        
        #first term 
        tmp = self.H.F[o,o].copy()
        tmp1 = np.zeros_like(tmp)
        tmp2 = np.zeros_like(tmp) 
        tmp3 = np.zeros_like(tmp)
        for ij in range(self.no*self.no): 
            i = ij // self.no
            j = ij % self.no
            ii = i *self.no + i
   
            #second term
            tmp1 += 0.5 * contract('ie,me->mi', t1_ii[ii], Fov_ii[ii])
            
            #third term
            for n in range(self.no):
                nn = n*self.no + n
        #try doing it with n and without n 
                tmp2 += contract('ne,mnie->mi', t1_ii[nn], Looov_ij[nn]) 
             
            #fourth term
        # no clue 
                #in = i*self.no +n
                #tmp3[i] += contract('ef,mnef->m', self.build_tau(ij,in,t1_ii, self.t2_ij, 1.0, 0.5), Loovv_ij[ij])
                
        return tmp + tmp1 + tmp2         

    def build_localFme(self, o, v, Fov_ii, Loovv_ij, t1_ii):
        Fme_ii = []
        Q = self.Local.Q
        L = self.Local.L
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            ii = i*self.no+i

            Fme = Fov_ii[ii]
            Fme1 = 0
            for n in range(self.no):
                nn = n*self.no +n
                Siinn = L[ii].T @ Q[ii].T @ Q[nn] @ L[nn] 
                Fme1 = Fme1 + contract('nF,mnEF,eE->me', t1_ii[nn], Loovv_ij[nn],Siinn)
            Fme_ii.append(Fme + Fme1)
        return Fme_ii     

    def localr_T1(self,o,v,r1_i,Fov_ii,t1_ii,Fae_ij,Fme_ij, Fmi, Lovvo_ij,t2_ij):
        Q = self.Local.Q
        L = self.Local.L
        cooler = np.zeros((self.no,self.nv))

        for ij in range(self.no*self.no):
            i = ij // self.no
            ii = i*self.no + i 
             
            #first term
            tmp = Fov_ii[ii].copy()
            
            #second term
            Siiij = L[ii].T @ Q[ii].T @ Q[ij] @ L[ij]
            Fae_ii = Siiij @ Fae_ij[ij] @ Siiij.T

            # using overlap term to Fae_ij[ij] 
            tmp1 = t1_ii[ii] @ Fae_ii.T
 
            # using no overlap
            #tmp1 = t1_ii[ii] @ Fae_ij[ii].T  

            #doing contraction instead with overlap
            #tmp1 = contract('ie,AE,aA,eE->ia', t1_ii[ii], Fae_ij[ij],Siiij, Siiij)    

            #doing contraction without overlap 
            #tmp1 = contract('ie,ae->ia', t1_ii[ii], Fae_ii)

            #third term 
            tmp2 = np.zeros_like(tmp)
            
            #1st attempt
            #tmp2 = -contract('ma,mi->ia', t1_ii[ii], Fmi)
             
            #second attempt
            for m in range(self.no):
                mm = m*self.no+m 
                Siimm = L[ii].T @ Q[ii].T @ Q[mm] @ L[mm]
                tmp2 -= contract('aA,mA,mi->ia',Siimm, t1_ii[mm],Fmi)

                #tmp2[i] -= t1_ii[ii][m] * Fmi[m
           
            #fourth term
            # I'm having trouble going through this part of the expression where I break down t2_ij to (o,o,v,v) 

            #fifth term
            for n in range(self.no):
                nn = n*self.no +n
                Siinn = L[ii].T @ Q[ii].T @ Q[nn] @ L[nn]
                tmp4 =  contract('F,afi,fF->ia', t1_ii[nn][n], Lovvo_ij[ii][n],Siinn)

            #sixth term
            #for m in range(self.no):
                #im = i*self.no + m
                #Siiim = L[ii].T @ Q[ii].T @ Q[im] @ L[im]
                #tmp5 = contract('imEF,mAEF,aA->ia', (2.0*cool[i,m] - cool[i,m].swapaxes(2,3)), self.ERIovvv_ij[im],Siiim)
            
            #seventh term
            for mn in range(self.no*self.no):
                m = mn // self.no
                n = mn % self.no
                Siimn = L[ii].T @ Q[ii].T @ Q[mn] @ L[mn]
                tmp6  =  contract('AE,Ei,aA->ia', t2_ij[mn], self.Loovo_ij[mn][n,m],Siimn)
                                             
            r1_i.append(tmp + tmp1) # + tmp2) + tmp4 +  tmp6) 
        return r1_i

    def localr_T2(self, o, v, r2_ij,ERIoovv_ij, t2_ij, Fae_ij, Fme_ii, Fmi):
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

            #third and fourth term
            #for m in range(self.no):
                #mm = m*self.no + m
                #Siimm = L[ii].T @ Q[ii].T @ Q[mm] @ L[mm]
             
                #r2_2 = contract('mb,me->be', t1_ii[mm], Fme_ij)

            #fifth term
            #r2_4 = np.zeros_like(r2_1) 
            #for m in range(self.no):
                #im = i*self.no +m
                #print(ij,im)
                #print(np.shape(Q[ij]),np.shape(Q[im]))
                #Sijim = L[ij].T @ Q[ij].T @ Q[im] @ L[im]
                #print(np.shape(Sijim),np.shape(t2_ij[im]))
                #cool = contract('aA,AB,bB->ab',Sijim, t2_ij[im], Sijim)
                #print("Fmi[m,j]")
                #print(Fmi[m,j]) 
                #r2_4 -= Fmi[m,j] * cool

            r2_ij.append(r2) # + r2_1)# + r2_4)
        return r2_ij

    def localcc_energy(self,o,v,Fov_ii,Loovv_ij,t1_ii,t2_ij):
        ecc_ij = 0
        ecc = 0
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
            ii = i*self.no + i
            
            ecc_ij = 2.0 * contract('Z,Z->',Fov_ii[ii][i],t1_ii[ii][i])

            ecc_ij += np.sum(np.multiply(self.build_localtau(ij,ij,t1_ii,t2_ij),Loovv_ij[ij][i,j]))
             
            ecc += ecc_ij

        return ecc

