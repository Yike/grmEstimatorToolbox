'''Calculate ATE, ATT and ATU by using the estimated parameters and simulated data
'''
import json
import grmReader
import numpy as np
from mpi4py import MPI
import sys


def evaluate():
    
    # read grmRslt.json and get the dictionary
    
    para=_getpara()    

    # get the estimated parameters from the dictionary
    
    Y1_beta=para['Y1_beta']
    
    Y0_beta=para['Y0_beta']
    
    U1_var=para['U1_var']
    
    U0_var=para['U0_var']
       
    # get the data from the .dat file
    
    data_=_getdata()
    
    # get numbAgents and numCovarsOut from the ini.file.
    initDict = grmReader.read()
    
    numAgents = initDict['numAgents']
    
    Y1_beta_=initDict['Y1_beta']
        
    numCovarsOut  = np.array(Y1_beta_).shape[0]

    # get the simulated X-covariates 
    
    X=data_[:,2:(numCovarsOut + 2)]
   
    # get the simulated decisions D
    
    D=data_[:,1]
    
    '''calculate ATE
    '''

    # calculate the level of Y_1 by using the estimated Y1_beta and simulated X
    Y1_level=np.dot(Y1_beta,X.T)
    
    # calculate the level of Y_0 by using the estimated Y0_beta and simulated X
    Y0_level=np.dot(Y0_beta,X.T)
    
    # checks
    assert (_checkdata(X,Y1_level,Y0_level,numAgents)==True)
    
    # get the number of people who are treated (D=1)
    numTreated=sum(D)
    
    # get the number of people who are untreated (D=0)
    numUntreated=numAgents-numTreated
       
    # simulate the sum of unobservables for people who are treated if they do get treated by using parallel computing
    sum_U1_tr=paraComp(numTreated,U1_var)
    
    # simulate the sum of unobservables for people who are treated if they don't get treated by using parallel computing
    sum_U0_tr=paraComp(numTreated,U0_var)
    
    # simulate the sum of unobservables for people who are untreated if they do get treated by using parallel computing 
    sum_U1_utr=paraComp(numUntreated,U1_var)
    
    # simulate the sum of unobservables for people who are untreated if they don't get treated by using parallel computing
    sum_U0_utr=paraComp(numUntreated,U0_var)

    
    # calculate ATE
    ATE = (sum(Y1_level)-sum(Y0_level)+sum_U1_tr+sum_U1_utr-sum_U0_tr-sum_U0_utr)/numAgents
    print "ATE = %s" % ATE
    
    '''calculate ATT
    '''
    
    # create an index indicating people who are treated (D=1)
    index_tr=np.where(D==1)[0]
    
    # get the X-covariates for people who are treated (D=1)
    X_tr=X[index_tr,:]
    
    # calculate the level of Y_1 for the treated agents (D=1)
    Y1_level_tr=np.dot(Y1_beta,X_tr.T)
    
    # calculate the level of Y_1 for the treated agents (D=1)
    Y0_level_tr=np.dot(Y0_beta,X_tr.T)
    
    # checks
    assert (_checkdata(X_tr,Y1_level_tr,Y0_level_tr,numTreated)==True)
    
    # calculate ATE
    ATT = (sum(Y1_level_tr)-sum(Y0_level_tr)+sum_U1_tr-sum_U0_tr)/numTreated
    print "ATT = %s" % ATT
    
    ''' calculate ATU
    '''

    # create an index indicating people who are untreated (D=0)
    index_utr=np.where(D==0)[0]
    
    # get the X-covariates for people who are untreated (D=0)
    X_utr=X[index_utr,:]
    
    # calculate the level of Y_1 for the untreated agents (D=0)
    Y1_level_utr=np.dot(Y1_beta,X_utr.T)
    
    # calculate the level of Y_1 for the untreated agents (D=0)
    Y0_level_utr=np.dot(Y0_beta,X_utr.T)
    
    # checks
    assert (_checkdata(X_utr,Y1_level_utr,Y0_level_utr,numUntreated)==True)
   
    # calculate ATU
    ATU = (sum(Y1_level_utr)-sum(Y0_level_utr)+sum_U1_utr-sum_U0_utr)/numUntreated
    print "ATU = %s" % ATU
        
    return 

def _getpara():
    '''read the .json file and export the saved dictionary
    '''
    json_data=open('grmRslt.json').read()
    
    para=json.loads(json_data)
    
    #check
    assert(isinstance(para,dict))
    
    return para

def _getdata():
    '''read the .dat file and export the simulated data
    '''
 
    # Process initialization file.
    initDict = grmReader.read()

    #read the data from the .dat file
    data_ = np.genfromtxt(initDict['fileName'], dtype = 'float')
        
    return data_

    
def _checkdata(X,Y1_level,Y0_level,numAgents):    
    
    # get numCovarsOut from the ini.file.
    initDict = grmReader.read()
    
    Y1_beta_=initDict['Y1_beta']
        
    numCovarsOut  = np.array(Y1_beta_).shape[0]
    
    # check X
    assert(isinstance(X,np.ndarray))
    assert(X.shape==(numAgents, numCovarsOut))
    assert(np.all(np.isfinite(X)))

    # check Y1_level
    assert(isinstance(Y1_level,np.ndarray))
    assert(Y1_level.shape==(numAgents, ))
    assert(np.all(np.isfinite(Y1_level)))
    
    # check Y0_level
    assert(isinstance(Y0_level,np.ndarray))
    assert(Y0_level.shape==(numAgents, ))
    assert(np.all(np.isfinite(Y0_level)))
    
    return True

def paraComp(N,Var):
    #print type(N)
    #print type(Var)
    parent_comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args=['grmworker.py'],
                               maxprocs=4)
    
    N=np.array(N,'i')
    parent_comm.Bcast([N, MPI.INT], root=MPI.ROOT)
    
    Var=np.array(Var,'d')
    parent_comm.Bcast([Var, MPI.DOUBLE], root=MPI.ROOT)
    
    draws_sum = np.array(0.0)
    parent_comm.Reduce(None, [draws_sum, MPI.DOUBLE],
                op=MPI.SUM, root=MPI.ROOT)
    
    parent_comm.Disconnect()
    return draws_sum

''' Executable.
'''
if __name__ == '__main__':
    
    evaluate()
