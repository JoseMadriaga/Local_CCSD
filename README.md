# Local_CCSD
Within this folder, the files to look for are:
lccwfn_test.py -> This is where I am currently implementing local correlation to CCSD starting with the double residuals ...
             the single residuals are kept in the MO basis such that all the integrals, amplitudes (even the t2 amplitude which is back-transform 
             to the MO basis to maintain the shape to allow contraction). For the implementation, I transform all the necessary integrals and the t2
             amplitudes into the semicanonical local basis and store them in a pair ij list. The overlap terms are constructed on the fly. 
             
ccwfn_old.py -> This is where I modified the old version of the ccwfn.py to only do local filter for the r2 while keeping the r1 completely in the MO basis
             including some locality that contracts with the t2 amplitudes as it iterates ...
          
The two files above work such that for every term I implement in lccwfn.py can be verified by the ccwfn_old.py 

test_013_lpnocc.py -> This is the input file I use to test my code 

Two other things to look at are: (1) log folder ( contains a "history" of what I have implemented and what things I have tried and what things work) 
(2) eqn folder (PNO expression of the local CCSD of what I have implemented )
