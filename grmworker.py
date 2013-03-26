#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

child_comm = MPI.Comm.Get_parent()
size = child_comm.Get_size()
rank = child_comm.Get_rank()
name = MPI.Get_processor_name()

N = np.array(0)
child_comm.Bcast([N, MPI.INT], root=0)
Var = np.array(0.0)
child_comm.Bcast([Var, MPI.DOUBLE], root=0)

number = N / size

np.random.seed(1234567)
draws = np.random.normal(0, scale=np.sqrt(Var), size=number)
draws_sum = np.sum(draws)
child_comm.Reduce([draws_sum, MPI.DOUBLE], None,
            op=MPI.SUM, root=0)

child_comm.Disconnect()
