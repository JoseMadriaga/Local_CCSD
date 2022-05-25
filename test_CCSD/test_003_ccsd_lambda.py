"""
Test CCSD Lambda equation solution using various molecule test cases.
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
from molecules import *
from ccwfn_main import *
from cchbar import * 
from cclambda import *
 


psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': 'cc-pvdz',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-13,
                      'd_convergence': 1e-13,
                      'r_convergence': 1e-13,
                      'diis': 1})
mol = psi4.geometry(moldict["H2O"])
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

maxiter = 75
e_conv = 1e-13
r_conv = 1e-13
max_diis = 8    

ccsd = ccwfn(rhf_wfn, local='PNO', local_cutoff=1e-5, init_t2 = 'OPT')
eccsd = ccsd.solve_cc(e_conv, r_conv, maxiter)
print(eccsd)
hbar = cchbar(ccsd)
cclambda = cclambda(ccsd, hbar)
lccsd = cclambda.solve_lambda(e_conv, r_conv, maxiter, max_diis)
print(lccsd)
