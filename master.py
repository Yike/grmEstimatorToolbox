#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import sys

def paraCom(N,Var):
    parent_comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args=['worker.py'],
                               maxprocs=4)
    
    N=np.array(N)
    parent_comm.Bcast([N, MPI.INT], root=MPI.ROOT)
    
    Var=np.array(Var)
    parent_comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)
    
    draws_sum = np.array(0.0)
    parent_comm.Reduce(None, [draws_sum, MPI.DOUBLE],
                op=MPI.SUM, root=MPI.ROOT)
    
    parent_comm.Disconnect()
    return draws_sum

ssss=paraCom(10000,0.222222)
print(ssss)