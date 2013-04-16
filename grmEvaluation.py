'''Calculate ATE, ATT and ATU by using the estimated parameters and simulated data
'''
import json
import grmReader
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def evaluate():
    
    # read grmRslt.json and get the dictionary
    
    para=_getpara()    

    # get the estimated parameters from the dictionary
    
    Y1_beta=para['Y1_beta']
    
    Y0_beta=para['Y0_beta']
    
    D_gamma=para['D_gamma']
    
    U1_var=para['U1_var']
    
    U0_var=para['U0_var']
    
    U1V_rho=para['U1V_rho']
    
    U0V_rho=para['U0V_rho']
    
    # normalization
    V_var=1
    
    # calculate the covariance between U1 and V
    U1V_cov=U1V_rho*np.sqrt(U1_var)*np.sqrt(V_var)
    
    # calculate covariance between U0 and V
    U0V_cov=U0V_rho*np.sqrt(U0_var)*np.sqrt(V_var)
    
    # get the data from the .dat file
    data_=_getdata()
    
    # get numbAgents, numCovarsOut, numCovarsCost and randomSeed from the ini.file.
    initDict = grmReader.read()
    
    numAgents = initDict['numAgents']
    
    Y1_beta_=initDict['Y1_beta']
        
    numCovarsOut  = np.array(Y1_beta_).shape[0]
    
    D_gamma_=initDict['D_gamma']
    
    numCovarsCost = np.array(D_gamma_).shape[0]
    
    randomSeed=initDict['randomSeed']
    
    #set random seed
    np.random.seed(randomSeed)

    # get the simulated X-covariates 
    
    X=data_[:,2:(numCovarsOut + 2)]
    
    # get the simulated Z-covariates
    
    Z=data_[:,-numCovarsCost:]
    
    # calculate the level of Y_1 by using the estimated Y1_beta and simulated X
    Y1_level=np.dot(Y1_beta,X.T)
    
    # calculate the level of Y_0 by using the estimated Y0_beta and simulated X
    Y0_level=np.dot(Y0_beta,X.T)
    
    # calculate the level of D by using the estimated D_gamma and simulated Z
    D_level=np.dot(D_gamma,Z.T)
    
    # simulate the unobservables based on the estimated distributions
    var_=[U1_var, U0_var, V_var]
    cov=np.diag(var_)
    cov[0,2]=U1V_cov
    cov[2,0]=cov[0,2]
    cov[1,2]=U0V_cov
    cov[2,1]=cov[1,2]
    
    U = np.random.multivariate_normal(np.tile(0.0,3), cov, numAgents)
    
    U1=U[:,0]
    U0=U[:,1]
    V=U[:,2]
  
    # simulate people's decisions
    D = np.array((Y1_level-Y0_level+U1-U0-D_level-V) > 0)
    
    # get the number of people who are treated (D=1)
    numTreated=sum(D)
    
    # get the number of people who are untreated (D=0)
    numUntreated=numAgents-numTreated
    
    # checks
    assert (_checkdata(X,Y1_level,Y0_level,numAgents)==True)
    
    '''calculate ATE
    '''
    ATE = (sum(Y1_level)-sum(Y0_level)+sum(U1)-sum(U0))/numAgents
    
    '''calculate ATT
    '''
    # create an index indicating people who are treated (D=1)
    index_tr=np.where(D==1)[0]
    
    # get the X-covariates for people who are treated (D=1)
    X_tr=X[index_tr,:]
    
    # get U1 for people who are treated (D=1)
    U1_tr=U1[index_tr,:]
    
    # get U0 for people who are treated (D=1)
    U0_tr=U0[index_tr,:]
    
    # calculate the level of Y_1 for the treated agents (D=1)
    Y1_level_tr=np.dot(Y1_beta,X_tr.T)
    
    # calculate the level of Y_0 for the treated agents (D=1)
    Y0_level_tr=np.dot(Y0_beta,X_tr.T)
    
    # checks
    assert (_checkdata(X_tr,Y1_level_tr,Y0_level_tr,numTreated)==True)
    
    # calculate ATT
    ATT = (sum(Y1_level_tr)-sum(Y0_level_tr)+sum(U1_tr)-sum(U0_tr))/numTreated

    
    ''' calculate ATU
    '''
    # create an index indicating people who are untreated (D=0)
    index_utr=np.where(D==0)[0]
    
    # get the X-covariates for people who are untreated (D=0)
    X_utr=X[index_utr,:]
    
    # get U1 for people who are untreated (D=0)
    U1_utr=U1[index_utr,:]
    
    # get U0 for people who are untreated (D=0)
    U0_utr=U0[index_utr,:]
    
    # calculate the level of Y_1 for the untreated agents (D=0)
    Y1_level_utr=np.dot(Y1_beta,X_utr.T)
    
    # calculate the level of Y_1 for the untreated agents (D=0)
    Y0_level_utr=np.dot(Y0_beta,X_utr.T)
    
    # checks
    assert (_checkdata(X_utr,Y1_level_utr,Y0_level_utr,numUntreated)==True)
   
    # calculate ATU
    ATU = (sum(Y1_level_utr)-sum(Y0_level_utr)+sum(U1_utr)-sum(U0_utr))/numUntreated

    #MPI
    TreatmentEffects = np.array(comm.gather([ATE,ATT,ATU],root=0))
    if rank==0:
        
        TreatmentEffects = np.mean(TreatmentEffects, axis=0)
        Treatments={}
        Treatments['ATE']  = TreatmentEffects[0]
        Treatments['ATT']  = TreatmentEffects[1]
        Treatments['ATU']= TreatmentEffects[2]
        
        print "ATE = %s" % Treatments['ATE']
        print "ATT = %s" % Treatments['ATT']
        print "ATU = %s" % Treatments['ATU']
    

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

''' Executable.
'''
if __name__ == '__main__':
    
    evaluate()
    
