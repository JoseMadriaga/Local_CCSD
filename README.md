# Local_CCSD
Within this folder, the files to look for are:
lccwfn_test2.py -> This is where I am currently implementing local correlation to CCSD
             
ccwfn_old.py -> This is where I modified the old version of the ccwfn.py to only do local filter for the r2 while keeping the r1 completely in the MO basis
             including some locality that contracts with the t2 amplitudes as it iterates ...
          
The two files above work such that for every term I implement in lccwfn.py can be verified by the ccwfn_old.py 

test_013_lpnocc.py -> This is the input file I use to test my code 

PNOformofCCSD -> Documentation of the derivation and implementation of local CCSD
