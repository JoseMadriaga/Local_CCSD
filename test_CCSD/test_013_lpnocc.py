"""
Test basic LPNO-CCSD energy and Lambda code
"""

# Import package, test suite, and other packages as needed
import psi4
import numpy as np
from molecules import *
from lccwfn_test import *
from ccwfn_old import *
from lccwfn import *

# Psi4 Setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
psi4.set_options({'basis': '6-31G',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      'freeze_core': 'false',
                      'e_convergence': 1e-12,
                      'd_convergence': 1e-10,
                      'r_convergence': 1e-12,
                      'diis': 8})
mol = psi4.geometry("""
        O -1.5167088799 -0.0875022822  0.0744338901
        H -0.5688047242  0.0676402012 -0.0936613229
        H -1.9654552961  0.5753254158 -0.4692384530
        symmetry c1
        noreorient
        nocom
""")
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)
emp2_ref = psi4.energy('mp2')
print(emp2_ref-rhf_e)
print(emp2_ref)

maxiter = 1000
e_conv = 1e-7
r_conv = 1e-3
max_diis = 8


ccsd = ccwfn_old(rhf_wfn,local='LPNO', local_cutoff=1e-7)
eccsd = ccsd.solve_cc(e_conv,r_conv,maxiter)
ccsd_local = lccwfn_test(rhf_wfn, local='LPNO', local_cutoff=1e-7)
eccsd_local = ccsd_local.solve_localcc(e_conv, r_conv, maxiter)
print(eccsd)
print(eccsd_local)
#assert (abs(epsi4 - eccsd) < 1e-7)
#assert (abs(lpsi4 - lccsd) < 1e-7)
