import psi4
import numpy as np
from opt_einsum import contract


class Local(object):
    """
    A Local object for simulating different (virtual-space) localization schemes.
	
    Attributes
    ----------
    C: Psi4.core.Matrix object
        localized active orbital coefficients
    nfzc: int
        number of frozen core orbitals
    no: int
        number of (active) occupied orbitals
    nv: int
        number of virtual orbitals
    H: PyCC Hamiltonian object
        Hamiltonian object with integrals, Fock matrices, and orbital coefficients
    cutoff: double
        main cutoff parameter for virtual space localizer
    core_cut: double (optional)
        used in PAO calculations for determining the norm-cutoff of the virtual space
    lindep_cut: double (optional)
        used in PAO calculations for determining linear dependence threshold

    Parameters
    ----------
    local: string
        type of local calculation ("PAO", "PNO", etc.)

    Methods
    -------
    filter_amps(): Returns t1 and t2 amplitude numpy arrays
        rotates amps to local vir space, applies CC step, and back-transforms
    filter_res(): Returns t1 and t2 amplitude numpy arrays
        applies forward and reverse localized-virtual transforms (no CC step)
    filter_t2amps(): Returns only t2 amplitude numpy arrays
        rotates t2 amps to local vir space, applies CC step, and back-transforms
    _build(): runs requested local build
    _build_PAO(): build PAO orbital rotation tensors
    _build_PNO(): build PNO(++) orbital rotation tensors   
    
    Notes
    -----
    -try using the simulation code to ignore certain amplitudes in the ccsd for NO
    -for now local MP2 is here; however in the future, can designate a class object for just local wfn (MP2, CC, etc.)
     to run local MP2, uncomment the necessary lines within the _build_"local" functions which are at the end 
    """

    def __init__(self, local, C, nfzc, no, nv, H, cutoff,
            core_cut=5E-2,
            lindep_cut=1E-6):

        self.cutoff = cutoff
        self.nfzc = nfzc
        self.no = no
        self.nv = nv
        self.H = H
        self.C = C.to_array()
        self.core_cut = core_cut
        self.lindep_cut = lindep_cut

        self._build(local)
 
    def _build(self,local):
        if local.upper() in ["PNO"] or local.upper() in ["PNOPP"]:
            self.local = local.upper()
            self._build_PNO()
        elif local.upper() in ["PAO"]:
            self._build_PAO()
        else:
            raise Exception("Not a valid local type!")

    def _build_PAO(self):
        """
        Attributes
        ----------
        Q: transform between canonical VMO and PAO spaces
        L: transform between PAO and semicanonical PAO spaces
        dim: dimension of PAO space
        eps: semicananonical PAO energies

        Notes
        -----
        - Equation numbers from Hampel & Werner 1996 [10.1063/1.471289]
        - D includes the sum over the inactive occ orbs (C_{pk}@C_{qk}^T) 
          because we still need to project frzn core out of the virtual space.
          everywhere else, use only active occ space C

        TODO
        ----
        - choose initial domains by charge (instead of just the first atom)
        - print domain atoms (not just # of functions)
        - simplify a2ao map generation
        """
        # SETUP 
        bs = self.H.basisset
        mints = psi4.core.MintsHelper(bs)
        no_all = self.no + self.nfzc
        D = self.H.C_all[:,:no_all] @ self.H.C_all[:,:no_all].T
        S = mints.ao_overlap().to_array() 
        nao = self.no + self.nv + self.nfzc

        # map the number of atoms, basis functions per atom, and their indices
        a2ao = {} # atom-to-nao dict
        for i in range(bs.nshell()): # initialize for every atom
            a2ao[str(bs.shell_to_center(i))] = []
        natom = len(list(a2ao.keys()))
        a2nao = [0]*natom
        for i in range(bs.nshell()): # find nbasis per atom
            a2nao[bs.shell_to_center(i)] += bs.shell(i).nfunction
        for i in range(natom): # store basis indices for each atom
            bsi = [b+sum(a2nao[:i]) for b in range(a2nao[i])]
            a2ao[str(i)] = bsi

        # now, a2ao is a dict (keys str(0-natom)) w/ vals as indices
        # of AOs on each atom

        # SINGLES DOMAINS
        atom_domains = [] # atom indices
        AO_domains = []   # AO basis function indices
        for i in range(0,self.no):
            # population matrix for orbital i
            charges = [0]*natom
            for j in range(natom):
                for k in a2ao[str(j)]:
                    SC = contract('l,l->',S[k,:],self.C[:,i])
                    charges[j] += (SC*self.C[k,i])

            print("Charge analysis for occupied orbital %3d:" % i)
            print(np.round(charges,2))

            atoms = [i for i in range(natom)]
            zipped = zip(np.abs(np.array(charges)),atoms)
            sort = sorted(zipped, reverse=True)
            tups = zip(*sort)
            charges,atoms = [list(t) for t in tups] # sorted by abs(charge)

            # choose which atoms belong to the domain based on charge
            atom_domains.append([])
#            charge = 0
#            for n in range(0,natom):
#                atom_domains[i].append(atoms.pop(0))
#                charge += charges.pop(0)
#                if charge>1.8:
#                    break
            atom_domains[i].append(atoms.pop(0))

            # AOs associated with domain atoms
            AOi = []
            for n in atom_domains[i]:
                AOi += a2ao[str(n)]
            AOi = sorted(AOi) # set low-high ordering so two spaces with 
                              # the same orbitals will have them in the same order

            chk = 1
            while chk > self.cutoff:
                # Eq 8, SRp = SR
                # let A == S, B == SR
                # form and solve ARp = B
                A = np.zeros((len(AOi),len(AOi)))
                SB = np.zeros((len(AOi),nao))
                for x,a in enumerate(AOi):
                    for y,b in enumerate(AOi):
                        A[x,y] = S[a,b]
                    for y,b in enumerate(range(nao)):
                        SB[x,y] = S[a,b]
                B = contract('mp,p->m',SB,self.C[:,i])
                Rp = np.linalg.solve(A,B)
    
                # Eq 9, completeness check
                chk = 1 - contract('m,mn,n->',Rp,SB,self.C[:,i])

                if chk > self.cutoff:
                    try:
                        n = atoms.pop(0)
                        atom_domains[i].append(n)
                        AOi += a2ao[str(n)]
                        AOi = sorted(AOi) 
                    except IndexError:
                        print("Ran out of atoms. How did that happen?")
                        if self.cutoff == 0:
                            print("Current BP value: {}".format(chk))
                            print("Cutoff = 0 ... continuing with full space")
                            chk = 0
                            continue
                        else:
                            raise IndexError
                else:
                    print("Completeness threshold fulfilled.")
                    print("BP completeness check: %.3f" % chk)
                    print("PAO domain %3d contains %3d/%3d orbitals." 
                            % (i,len(AOi),self.no+self.nv))
            AO_domains.append(AOi)

        # at this point, atom_domains contains the indices for each atom in the 
        # PAO space for each occupied orbital
        # and AO_domains contains the (sorted low->high) indices of the
        # AOs which make up the PAO space for each occupied orbital

        # Eq 3, total virtual-space projector
        Rt_full = np.eye(S.shape[0]) - contract('ik,kj->ij',D,S)

        # remove PAOs with negligible norms
        for i in range(nao):
            norm = np.linalg.norm(Rt_full[:,i]) # column-norm
            if norm < self.core_cut:
                print("Norm of orbital %4d = %20.12f... deleting" % (i,norm))
                Rt_full[:,i] = 0

        # Eq 5, first two terms RHS
        # R^+.S for virtual space, we will use this to compute the LMO->PAO 
        # transformation matrix
        RS = self.C[:,self.no:].T @ S

        Q = []  # MOij->PAOij
        L = []  # PAOij->PAOij(semi-canonical)
        eps = [] # semi-canonical orbital energies
        dim = [] # dimension of final PAO pair spaces
        for ij in range(self.no**2):
            i = ij // self.no
            j = ij % self.no
            ij_domain = list(set(AO_domains[i]+AO_domains[j]))

            # Eq 5, last term RHS for a given pair
            Rt = np.zeros((nao,len(ij_domain)))
            for x,a in enumerate(ij_domain):
                Rt[:,x] = Rt_full[:,a] # virtual-space projector for pair ij

            # Eq 73, MO -> PAO (redundant)
            # called the "Local residual vector" in psi3
            # equivalent to Q in PNO code
            # used to transform the LMO-basis residual matrix into the
            # projected (redundant, non-canonical) PAO basis
            V = contract('ap,pq->aq',RS,Rt)
            Q.append(V)

            # Eq 5, PAO -> semicanonical PAO
            # check for linear dependencies 
            St = contract('pq,pr,rs->qs',Rt,S,Rt)
            evals,evecs = np.linalg.eigh(St)
            toss = np.abs(evals) < self.lindep_cut 
            if sum(toss) > 0:
                print("%1d linearly dependent orbitals removed." % (sum(toss)))

            # Eq 53, normalized nonredundant transform 
            # (still not semi-canonical)
            Xt = np.delete(evecs,toss,axis=1)
            evals = np.delete(evals,toss)
            for c in range(Xt.shape[1]):
                Xt[:,c] = Xt[:,c] / evals[c]**0.5
            dim.append(Xt.shape[1])

            # just below Eq 51, redundant PAO Fock 
            Ft = contract('pq,pr,rs->qs',Rt,self.H.F_ao,Rt)

            # Eq 54, non-redundant PAO Fock
            # diagonalize to get semi-canonical space
            Fbar = contract('pq,pr,rs->qs',Xt,Ft,Xt)
            evals,evecs = np.linalg.eigh(Fbar)

            # Eq 51, W 
            # rotates the redundant PAO-basis amplitudes 
            # directly into the into the non-redundant, semi-canonical basis
            W = contract('pq,qs->ps',Xt,evecs)

            eps.append(evals)
            L.append(W)

            print("Pair domain (%1d,%1d) contains %3d/%3d orbitals." 
                                % (i,j,dim[-1],nao))

        print("Average PAO dimension: %.2f" % (np.average(dim)))
        print("Number of canonical VMOs: %d" % (self.nv))

        self.Q = Q  # transform between canonical VMO and PAO spaces
        self.L = L  # transform between PAO and semicanonical PAO spaces
        self.dim = dim  # dimension of PAO space
        self.eps = eps  # semicananonical PAO energies
 
        #print("Now doing PAO_mp2")
        #self._local_MP2_loop()
        #self._sim_MP2_loop()
 
    def _build_PNO(self):
        """
        Perform MP2 loop in non-canonical MO basis, then construct pair density based on unperturbed or perturbed amplitudes
        then build MP2-level PNOs(++) 

        Attributes
        ----------
        Q: transform between canonical VMO and LPNO spaces
        L: transform between LPNO and semicanonical LPNO spaces
        dim: dimension of LPNO space
        eps: semicananonical LPNO energies
        """

        o = slice(0, self.no)
        v = slice(self.no, self.no+self.nv)

        # Compute MP2 amplitudes in non-canonical MO basis
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # initial guess amplitudes
        t2 = self.H.ERI[o,o,v,v]/Dijab
        
        print("Initial MP2 energy:")
        print(contract('ijab,ijab->', t2, self.H.L[o,o,v,v]))

        # MP2 loop (optional)
        self._MP2_loop(t2,self.H.F,self.H.ERI,self.H.L,Dijab)
        
        # Construct pair density 
        T_ij = t2.copy().reshape((self.no*self.no, self.nv, self.nv))
        Tt_ij = 2.0 * T_ij - T_ij.swapaxes(1,2)
        D = np.zeros_like(T_ij)
        Q_full = np.zeros_like(T_ij)
        Q = []  # truncated LPNO list
        occ = np.zeros((self.no*self.no, self.nv))
        dim = np.zeros((self.no*self.no), dtype=int)  # dimension of LPNO space for each pair
        L = []
        eps = []
        T2_PNO = 0

        if self.local in ["PNO"]:
            print("Computing " + self.local +"s.  Canonical VMO dim: %d" % (self.nv))
            D = self.form_density(t2) 

        elif self.local in ["PNOPP"]:
            print("Computing " + self.local +"s.  Canonical VMO dim: %d" % (self.nv))
            
            # Hbar_ii  = f_ii + t_inef ( 2 * <in|ef> - <in|fe> )
            Hbar_oo = self.H.F[o,o].copy()
            Hbar_oo += contract ('inef,mnef->mi',t2, self.H.L[o,o,v,v])
            Hbar_ii = Hbar_oo.diagonal().copy()

            # Hbar_aa = f_aa - t_mnfa (2 * <mn|fa> - <mn|af> )
            Hbar_vv = self.H.F[v,v].copy()
            Hbar_vv -= contract ('mnfa,mnfe->ae',t2,self.H.L[o,o,v,v])
            Hbar_aa = Hbar_vv.diagonal().copy()

            denom_ia = Hbar_ii.reshape(-1,1) - Hbar_aa
            #need to add this for response
            #denom_ia += omega

            denom_ijab = Hbar_ii.reshape(-1, 1, 1, 1) + Hbar_ii.reshape(-1, 1, 1) - Hbar_aa.reshape(-1, 1) - Hbar_aa
            #going to ignore the omega for a moment such that it is a static case
            #denom_ijab += omega

            self.denom_tuple = (denom_ia, denom_ijab)

            # need to initialize the Abar terms
            A_list = {}
            X_guess = {}

            ## Here, perturbation is dipole moment
            pert = "mu"
            if pert == "mu":
                bs = self.H.basisset
                mints = psi4.core.MintsHelper(bs)
                dipole_array = mints.ao_dipole()
                dirn = ['X','Y','Z']
                
                for i, drn in enumerate(dirn):
                    A_list[drn] = np.einsum('uj,vi,uv', self.C, self.C, np.asarray(dipole_array[i]))
                    A = A_list[drn]
                
                    # Build guess Abar
                    # Abar_ijab = P_ij^ab (t_ij^eb A_ae - t_mj^ab A_mi)
                    Avvoo = contract('ijeb,ae->abij', t2, A[self.no:, self.no:])
                    Avvoo -= contract('mjab,mi->abij', t2, A[:self.no, :self.no])
                    Abar = Avvoo.swapaxes(0,2).swapaxes(1,3)
                    Abar += Abar.swapaxes(0,1).swapaxes(2,3)

                    # Build guess X's
                    # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb
                    X_guess[drn] = Abar.copy()
                    X_guess[drn] /= denom_ijab

                    D += self.form_density(X_guess[drn])

                D /= 3.0

        # Now obtain the Q and L
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            # Compute PNO(++) and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
            if (occ[ij] < 0).any(): # Check for negative occupation numbers
                neg = occ[ij][(occ[ij]<0)].min()
                print("Warning! Negative occupation numbers up to {} detected. \
                      Using absolute values - please check if your input is correct.".format(neg))
            dim[ij] = (np.abs(occ[ij]) > self.cutoff).sum()
            Q.append(Q_full[ij, :, (self.nv-dim[ij]):])

            # Compute semicanonical virtual space
            F = Q[ij].T @ self.H.F[v,v] @ Q[ij]  # Fock matrix in PNO(++) basis
            eval, evec = np.linalg.eigh(F)
            eps.append(eval)
            L.append(evec)
            T2_PNO += dim[ij] * dim[ij]
            print(self.local + " dimension of pair %d = %d" % (ij, dim[ij]))

        print("Average " + self.local + " dimension: %2.3f" % (np.average(dim)))
        print("T2 " +  self.local + ": %d" % (T2_PNO))
        T2_full = (self.no*self.no)*(self.nv*self.nv)
        print("T2 full: %d" % (T2_full))
        print("T2 Ratio: %3.12f" % (T2_PNO/T2_full))
        self.Q = Q  # transform between canonical VMO and PNO(++) spaces
        self.dim = dim  # dimension of PNO(++) space
        self.eps = eps  # semicananonical PNO(++) energies
        self.L = L  # transform between PNO(++) and semicanonical PNO(++) spaces
        
        #print("Now doing local mp2")
        #self._local_MP2_loop()
        #self._sim_MP2_loop()

    def _PNO_space(self, t2):
        # Build LPNOs and store transformation matrices
        print("Computing PNOs.  Canonical VMO dim: %d" % (self.nv))
        T_ij = t2.copy().reshape((self.no*self.no, self.nv, self.nv))
        Tt_ij = 2.0 * T_ij - T_ij.swapaxes(1,2)
        D = np.zeros_like(T_ij)
        Q_full = np.zeros_like(T_ij)
        Q = []  # truncated LPNO list
        occ = np.zeros((self.no*self.no, self.nv))
        dim = np.zeros((self.no*self.no), dtype=int)  # dimension of LPNO space for each pair
        L = []
        eps = []
        T2_PNO = 0
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
    
            # Compute pair density
            D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
            D[ij] *= 2.0/(1 + int(i == j))
    
            # Compute PNOs and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
            if (occ[ij] < 0).any(): # Check for negative occupation numbers
                neg = occ[ij][(occ[ij]<0)].min()
                print("Warning! Negative occupation numbers up to {} detected. \
                        Using absolute values - please check if your input is correct.".format(neg))
            dim[ij] = (np.abs(occ[ij]) > self.cutoff).sum()
            Q.append(Q_full[ij, :, (self.nv-dim[ij]):])
    
            # Compute semicanonical virtual space
            v = slice(self.no, self.no+self.nv)
            F = Q[ij].T @ self.H.F[v,v] @ Q[ij]  # Fock matrix in PNO basis
            eval, evec = np.linalg.eigh(F)
            eps.append(eval)
            L.append(evec)
            T2_PNO += dim[ij] * dim[ij]
            print("PNO dimension of pair %d = %d" % (ij, dim[ij]))

        print("Average PNO dimension: %d" % (np.average(dim)))
        print("Number of canonical VMOs: %d" % (self.nv))
        print("T2 PNO: %d" % (T2_PNO))
        T2_full = (self.no*self.no)*(self.nv*self.nv)
        print("T2 full: %d" % (T2_full))
        print("T2 Ratio: %3.12f" % (T2_PNO/T2_full))
        self.Q = Q  # transform between canonical VMO and LPNO spaces
        self.dim = dim  # dimension of LPNO space
        self.eps = eps  # semicananonical LPNO energies
        self.L = L  # transform between LPNO and semicanonical LPNO spaces

    def _build_LPNOpp(self):
        """
        Perform MP2 loop in non-canonical MO basis, then build MP2-level PNOpps

        Attributes
        ----------
        Q: transform between canonical VMO and PNO++ spaces
        L: transform between LPNO and semicanonical PNO++ spaces
        dim: dimension of PNO++ space
        eps: semicananonical PNO++ energies

        Notes
        -----
        from D'Cunha & Crawford 2021 [10.1021/acs.jctc.0c01086]
        """

        o = slice(0, self.no)
        v = slice(self.no, self.no+self.nv)
        
        # Compute MP2 amplitudes in non-canonical MO basis
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir
        
        # initial guess amplitudes
        t2 = self.H.ERI[o,o,v,v]/Dijab
        
        # MP2 loop (optional) 
        self._MP2_loop(t2,self.H.F,self.H.ERI,self.H.L,Dijab)
        
        # build MP2-level space
        self._PNOpp_space(t2)

        #print("Now doing LPNOpp_mp2")
        #self._local_MP2_loop()
        #self._sim_MP2_loop()

    def _PNOpp_space(self,t2):
        '''
         Constructing the approximated perturbed pair density
        
        Notes
        -----
        Modification of Ruhee's construction of the pno++ density 
        For now, not frequency dependent (No +omega in the energy denominators)
        pert = "mu" only

        '''
        o = slice(0, self.no)
        v = slice(self.no, self.no+self.nv)
        print("Perturbed density on. Preparing perturbed density PNOs.")

        # Hbar_ii  = f_ii + t_inef ( 2 * <in|ef> - <in|fe> )
        Hbar_oo = self.H.F[o,o].copy()
        Hbar_oo += contract ('inef,mnef->mi',t2, self.H.L[o,o,v,v])
        Hbar_ii = Hbar_oo.diagonal().copy() 
       
        # Hbar_aa = f_aa - t_mnfa (2 * <mn|fa> - <mn|af> )
        Hbar_vv = self.H.F[v,v].copy()
        Hbar_vv -= contract ('mnfa,mnfe->ae',t2,self.H.L[o,o,v,v])
        Hbar_aa = Hbar_vv.diagonal().copy()
        denom_ia = Hbar_ii.reshape(-1,1) - Hbar_aa

        #need to add this for response
        #denom_ia += omega
                
        denom_ijab = Hbar_ii.reshape(-1, 1, 1, 1) + Hbar_ii.reshape(-1, 1, 1) - Hbar_aa.reshape(-1, 1) - Hbar_aa

        #going to ignore the omega for a moment such that it is a static case
        #denom_ijab += omega

        self.denom_tuple = (denom_ia, denom_ijab)

        # need to initialize the Abar terms
        A_list = {}
        X_guess = {}
        D = np.zeros((self.no * self.no, self.nv, self.nv))
        Q_full = np.zeros_like(D)
        Q = []
        occ = np.zeros((self.no*self.no, self.nv))
        dim = np.zeros((self.no*self.no), dtype=int)  # dimension of LPNO space for each pair
        L = []
        eps = []
        T2_PNO = 0
        
        ## Here, perturbation is dipole moment
        pert = "mu"    
        if pert == "mu":
            bs = self.H.basisset
            mints = psi4.core.MintsHelper(bs) 
            dipole_array = mints.ao_dipole()
            dirn = ['X','Y','Z']
            for i, drn in enumerate(dirn):
                A_list[drn] = np.einsum('uj,vi,uv', self.C, self.C, np.asarray(dipole_array[i]))
            
                A = A_list[drn]
                # Build guess Abar
                # Abar_ijab = P_ij^ab (t_ij^eb A_ae - t_mj^ab A_mi)
                Avvoo = contract('ijeb,ae->abij', t2, A[self.no:, self.no:])
                Avvoo -= contract('mjab,mi->abij', t2, A[:self.no, :self.no])
                Abar = Avvoo.swapaxes(0,2).swapaxes(1,3)
                Abar += Abar.swapaxes(0,1).swapaxes(2,3)

                # Build guess X's
                # X_ijab = Abar_ijab / Hbar_ii + Hbar_jj - Hbar_aa _ Hbar_bb
                X_guess[drn] = Abar.copy()
                X_guess[drn] /= denom_ijab

                D += self.form_density(X_guess[drn])
  
            D /= 3.0

        # Now obtain the Q and L 
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no 

            # Compute PNOs and truncate
            occ[ij], Q_full[ij] = np.linalg.eigh(D[ij])
            if (occ[ij] < 0).any(): # Check for negative occupation numbers
                neg = occ[ij][(occ[ij]<0)].min()
                print("Warning! Negative occupation numbers up to {} detected. \
                      Using absolute values - please check if your input is correct.".format(neg))
            dim[ij] = (np.abs(occ[ij]) > self.cutoff).sum()
            Q.append(Q_full[ij, :, (self.nv-dim[ij]):])
            
            # Compute semicanonical virtual space
            F = Q[ij].T @ self.H.F[v,v] @ Q[ij]  # Fock matrix in PNO basis
            eval, evec = np.linalg.eigh(F)
            eps.append(eval)
            L.append(evec)
            T2_PNO += dim[ij] * dim[ij]
            print("PNO dimension of pair %d = %d" % (ij, dim[ij]))

        print("Average PNO dimension: %2.3f" % (np.average(dim)))
        print("T2 PNO: %d" % (T2_PNO))
        T2_full = (self.no*self.no)*(self.nv*self.nv)
        print("T2 full: %d" % (T2_full))
        print("T2 Ratio: %3.12f" % (T2_PNO/T2_full))
        self.Q = Q  # transform between canonical VMO and LPNO spaces
        self.dim = dim  # dimension of LPNO space
        self.eps = eps  # semicananonical LPNO energies
        self.L = L  # transform between LPNO and semicanonical LPNO spaces

    def form_density(self, t_ijab):
        # Create Tij and Ttij
        T_ij = t_ijab.copy().reshape((self.no * self.no, self.nv, self.nv))
        Tt_ij = 2.0 * T_ij.copy()
        Tt_ij -= T_ij.swapaxes(1, 2)

        # Form pair densities
        D = np.zeros((self.no * self.no, self.nv, self.nv))
        for ij in range(self.no * self.no):
            i = ij // self.no
            j = ij % self.no
            D[ij] = contract('ab,bc->ac', T_ij[ij], Tt_ij[ij].T) + contract('ab,bc->ac', T_ij[ij].T, Tt_ij[ij])
            D[ij] *= 2.0 / (1.0 + int(i==j))
            D[ij] += D[ij].T
            D[ij] *= 0.5
        return D

    def _MP2_loop(self,t2,F,ERI,L,Dijab):
        '''
        Perform the MP2 loop by minimization of the Hylleraas functional
    
        Parameters
        ----------
        t2: numpy array
            initial amplitudes
        ERI: numpy array
            two-electron repulsion integrals
        L: numpy array
            2*ERI - 2*ERI.swapaxes(2,3)
        Dijab: numpy array 
            Fock matrix eigenvalue denominator

        Notes
        -----
        Hylleraas functional form currently does not include overlap terms

        TODO
        ----
        Add optional overlap terms
        '''

        o = slice(0, self.no)
        v = slice(self.no, self.no+self.nv)
        emp2 = contract('ijab,ijab->', t2, L[o,o,v,v])
        print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E" % (0, emp2, -emp2))
        
        e_conv = 1e-7
        r_conv = 1e-7
        maxiter = 100
        ediff = emp2
        rmsd = 0.0
        niter = 0

        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            niter += 1
            elast = emp2
            r2 = 0.5 * ERI[o,o,v,v].copy()
            r2 += contract('ijae,be->ijab', t2, F[v,v])
            r2 -= contract('imab,mj->ijab', t2, F[o,o])
            r2 = r2 + r2.swapaxes(0,1).swapaxes(2,3)
            
            t2 += r2/Dijab

            rmsd = np.sqrt(contract('ijab,ijab->', r2/Dijab, r2/Dijab))

            emp2 = contract('ijab,ijab->', t2, L[o,o,v,v])
            ediff = emp2 - elast

            print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E  rmsd = % .5E" % (niter, emp2, ediff, rmsd))
    
    def _sim_MP2_loop(self):
        print("Now doing a comparison against simulation code")
        Q = self.Q
        L = self.L
        o = slice(0,self.no)
        v = slice(self.no, self.no + self.nv)
        
        # localized fock occupied matrix 
        F_occ = self.H.F[o,o]

        # approximated canonical orbital energies
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir

        # initial guess amplitudes
        t2 = self.H.ERI[o,o,v,v]/Dijab
        
        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no

            t2_ij = L[ij].T @ Q[ij].T @ t2[i,j] @ Q[ij] @ L[ij]
            t2_ij /= -1*(self.eps[ij].reshape(-1,1) + self.eps[ij].reshape(1,-1) - self.H.F[i,i] - self.H.F[j,j])       
            t2[i,j] = Q[ij] @ L[ij] @ t2_ij @ L[ij].T @ Q[ij].T
       
        t2_ij = 0
     
        emp2 = contract('ijab,ijab->', t2, self.H.L[o,o,v,v])
        print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E" % (0, emp2, -emp2))
      
        e_conv = 1e-7
        r_conv = 1e-7
        maxiter = 100
        ediff = emp2
        niter = 0
        
        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            niter += 1
            elast = emp2
            emp2 = 0
            rmsd = 0
 
            r2 = 0.5 * self.H.ERI[o,o,v,v].copy()
            r2 += contract('ijac,cb->ijab', t2, self.H.F[v,v])
            r2 -= contract('imab,mj->ijab', t2, self.H.F[o,o])
            r2 = r2 + r2.swapaxes(0,1).swapaxes(2,3)
            
            new_tijab = np.zeros((self.no, self.no, self.nv, self.nv))
             
            new_tijab = self.filter_t2amps(r2) 
 
            t2 += new_tijab
 
            emp2 = contract('ijab,ijab->', t2, self.H.L[o,o,v,v])

            ediff = emp2 - elast

            rmsd = np.sqrt(contract('ijab,ijab->', new_tijab, new_tijab))

            print("MP2 Iter %3d: MP2 Ecorr = %.15f  dE = % .5E  rmsd = % .5E" % (niter, emp2, ediff, rmsd))

    def _local_MP2_loop(self,local,S):
        """
        Performs the MP2 loop in the transformed local basis (PAO, PNO)

        Attributes
        ----------
        Q: transform between canonical VMO and local spaces
        L: transform between local and semicanonical local spaces
        eps: semicananonical local energies

        Notes
        -----

        """
        Q = self.Q
        L = self.L
        o = slice(0,self.no)
        v = slice(self.no, self.no + self.nv)
        
        # noting the occupied fock matrix
        F_occ = self.H.F[o,o]
       
        # ERI_ij = local_semi by local_semi
        # t2_ij = local_semi by local_semi

        # initial guess amplitude in the approximated MO basis, dont really need
        eps_occ = np.diag(self.H.F)[o]
        eps_vir = np.diag(self.H.F)[v]
        Dijab = eps_occ.reshape(-1,1,1,1) + eps_occ.reshape(-1,1,1) - eps_vir.reshape(-1,1) - eps_vir
        t2 = self.H.ERI[o,o,v,v]/Dijab
  
        ERI_ij = []
        t2_ij = []

        for ij in range(self.no*self.no):
            i = ij // self.no
            j = ij % self.no
                
            ERI_ij.append(L[ij].T @ Q[ij].T @ self.H.ERI[i,j,v,v] @ Q[ij] @ L[ij])
             
            t2_ij.append( -1.0 * ERI_ij[ij] / (self.eps[ij].reshape(1,-1) + self.eps[ij].reshape(-1,1) - self.H.F[i,i] - self.H.F[j,j]))
        
        # parameters for the MP2 loop
        e_conv = 1e-7
        r_conv = 1e-7
        maxiter = 100
                
        emp2 = contract('ijab,ijab->', t2,self.H.L[o,o,v,v])
        print(emp2)
        ediff = emp2
        niter = 0
        maxiter = 100

        while ((abs(ediff) > e_conv) or (abs(rmsd) > r_conv)) and (niter <= maxiter):
            niter += 1
            elast = emp2
            emp2 = 0
            rmsd = 0
           
           #modification to zach's dlpno-mp2 python code 
            for ij in range(self.no*self.no):  
                i = ij // self.no 
                j = ij % self.no
                    
           # self.eps[ij] contains the semicanonical orbital energies obtain through the diagonalization of the local fock matrix                    
                r2_ij = ERI_ij[ij] + (self.eps[ij].reshape(-1,1) + self.eps[ij].reshape(1,-1) - self.H.F[i,i] - self.H.F[j,j])* t2_ij[ij]

                for k in range(self.no):
                    if i != k: 
                        kj = self.no * k + j 
                        S_ijkj = L[ij].T @ Q[ij].T @ Q[kj] @ L[kj]
                        r2_ij -= F_occ[i,k] * contract('rs,ar,bs->ab',t2_ij[kj],S_ijkj,S_ijkj)
                    if j != k:
                        ik = self.no * i + k 
                        S_ijik = L[ij].T @ Q[ij].T @ Q[ik] @ L[ik]
                        r2_ij -= F_occ[k,j] * contract('rs,ar,bs->ab',t2_ij[ik],S_ijik,S_ijik)
                

                t2_ij[ij] -= r2_ij / (self.eps[ij].reshape(-1,1) + self.eps[ij].reshape(1,-1) - self.H.F[i,i] - self.H.F[j,j])
                #if ij == 0:
                        #print('inc from PNO')
                        #print(r2_ij / (self.eps[ij].reshape(-1,1) + self.eps[ij].reshape(1,-1) - self.H.F[i,i] - self.H.F[j,j]))
                          
                rmsd += np.sqrt(contract('ab,ab->', r2_ij, r2_ij))
                
                L_ij = 2.0 * t2_ij[ij] - t2_ij[ij].T
                e_ij_mp2 = np.sum(np.multiply(ERI_ij[ij], L_ij))
                emp2 += e_ij_mp2
   
                #t2[i,j] = Q[ij] @ L[ij] @ t2_ij[ij] @ L[ij].T @ Q[ij].T 

            #emp2 = contract('ijab,ijab->', t2, self.H.L[o,o,v,v])
                       
            ediff = emp2 - elast

            print("MP2 Iter %3d: MP2 Ecorr = %.15f dE = % .5E rmsd = % .5E" % (niter, emp2, ediff, rmsd))
    
    def filter_t2amps(self,r2):
        no = self.no
        nv = self.nv
        dim = self.dim

        t2 = np.zeros((no,no,nv,nv))
        for ij in range(no*no):
            i = ij // no
            j = ij % no

            X = self.Q[ij].T @ r2[i,j] @ self.Q[ij]
            Y = self.L[ij].T @ X @ self.L[ij]

            for a in range(dim[ij]):
                for b in range(dim[ij]):
                    Y[a,b] = Y[a,b]/(self.H.F[i,i] + self.H.F[j,j] - self.eps[ij][a] - self.eps[ij][b])

            X = self.L[ij] @ Y @ self.L[ij].T
            t2[i,j] = self.Q[ij] @ X @ self.Q[ij].T

        return t2

    def filter_amps(self, r1, r2):
        no = self.no
        nv = self.nv
        dim = self.dim

        t1 = np.zeros((no,nv))
        for i in range(no):
            ii = i * no + i

            X = self.Q[ii].T @ r1[i]
            Y = self.L[ii].T @ X

            for a in range(dim[ii]):
                Y[a] = Y[a]/(self.H.F[i,i] - self.eps[ii][a])

            X = self.L[ii] @ Y
            t1[i] = self.Q[ii] @ X

        t2 = np.zeros((no,no,nv,nv))
        for ij in range(no*no):
            i = ij // no
            j = ij % no
   
            X = self.Q[ij].T @ r2[i,j] @ self.Q[ij]
            Y = self.L[ij].T @ X @ self.L[ij]

            for a in range(dim[ij]):
                for b in range(dim[ij]):
                    Y[a,b] = Y[a,b]/(self.H.F[i,i] + self.H.F[j,j] - self.eps[ij][a] - self.eps[ij][b])

            X = self.L[ij] @ Y @ self.L[ij].T
            t2[i,j] = self.Q[ij] @ X @ self.Q[ij].T

        return t1, t2

    def filter_res(self, r1, r2):
        no = self.no
        nv = self.nv

        t1 = np.zeros((no,nv)).astype('complex128')
        for i in range(no):
            ii = i * no + i

            X = self.Q[ii].T @ r1[i]
            t1[i] = self.Q[ii] @ X

        t2 = np.zeros((no,no,nv,nv)).astype('complex128')
        for ij in range(no*no):
            i = ij // no
            j = ij % no

            X = self.Q[ij].T @ r2[i,j] @ self.Q[ij]
            t2[i,j] = self.Q[ij] @ X @ self.Q[ij].T

        return t1, t2
