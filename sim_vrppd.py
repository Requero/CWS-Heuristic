import os
from pathlib import Path
from io import StringIO
import sys
import cwsheuristic_vrppd as cws
import copy
import vrppd_objects as vrp
import time
import pandas as pd 
import lithops as lth
import numpy as np
#from lithops import Storage
import multiprocessing as mp
import itertools as it
import shutil
#from ibm_botocore.client import Config
#import ibm_boto3

def betaDistribution(mean, std):
    n = 1000
    mu = mean/n
    sigma = std/n
    alfa = abs(((1-mu)/(sigma**2)-(1/mu))*mu**2)
    beta = abs(alfa*((1/mu)-1))
    return 1000*np.random.beta(alfa, beta)

def generateStochasticValue(mean, desv):
    factor = mean**2/desv**2
    if factor < 15:
        return betaDistribution(mean, desv) 
    else:
        return np.random.normal(mean, desv)

def getNodeMatrixaWithStochasticDemandAndSuppy(nodeMatrix):
    nodeMatrixStochastic = []
    for node in nodeMatrix:
        demand_stochastic = 0 if node[0] == 0.0 else np.ceil(generateStochasticValue(node[3], node[4]))
        supply_stochastic = 0 if node[0] == 0.0 else np.ceil(generateStochasticValue(node[5], node[6]))
        nodeMatrixStochastic.append([node[0],node[1], node[2], demand_stochastic, supply_stochastic])
    return nodeMatrixStochastic

def VRPPDDeterministic(instanceData):
    #instanceName, veCap, nodeMatrix
    instanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    instanceCws.runCWSSol()
    return np.array([instanceData[0],instanceData[1], instanceCws.getCost()])

def VRPPDStochasticLithops(instanceData, beta): 
    nIterations = 10
    minCost = 100000
    nodeMatrixStochastic = getNodeMatrixaWithStochasticDemandAndSuppy(instanceData[2])
    localInstanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], nodeMatrixStochastic)
    bestSol = vrp.Solution()
    bestSol.cost = 1000000000
    #startTime = time.time()
    minCost = 100000
    for iteration in range(nIterations):
        localInstanceCws.runCWSSolGeneral(beta)
        sol = localInstanceCws.getSolution()
        if sol.cost < minCost:
            bestSol = copy.deepcopy( sol )
            minCost = bestSol.cost
    #deltaTime = time.time() - startTime
    return [instanceData[0], instanceData[1], beta, minCost]

def VRPPDStochastic(params): #instanceData, nIterations, beta):
    instanceData, beta = params[0], params[1]
    nIterations = 10
    minCost = 100000
    nodeMatrixStochastic = getNodeMatrixaWithStochasticDemandAndSuppy(instanceData[2])
    localInstanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], nodeMatrixStochastic)
    bestSol = vrp.Solution()
    bestSol.cost = 1000000000
    #startTime = time.time()
    minCost = 100000
    for iteration in range(nIterations):
        localInstanceCws.runCWSSolGeneral(beta)
        sol = localInstanceCws.getSolution()
        if sol.cost < minCost:
            bestSol = copy.deepcopy( sol )
            minCost = bestSol.cost
    #deltaTime = time.time() - startTime
    return [instanceData[0], instanceData[1], beta, minCost]

def readInstances(storage,params):
    #bucket_name, vehCaps, file
    fileinstance = storage.get_object(params[0], params[1]['Key'], stream=True)
    filecapacity = storage.get_object(params[0], params[2]['Key'], stream=True)
    instanceName = str(params[1]['Key']).replace('Instances3/', '').replace('_input_nodes.txt', '')
    i = 0
    nodeMatrix = []
    lines1 = fileinstance.read().decode('ISO-8859-1').split('\n')
    for line in lines1:
        if line[0]!='#':
            nums = [float(i) for i in line.split()]
            nodeMatrix.append(np.array([i, float(nums[0]), float(nums[1]), float(nums[2])]))
            i += 1
    lines2 = filecapacity.read().decode('ISO-8859-1').split('\n')
    for line in lines2:
        if line[0] !='#':
            nums = line.split()
            capacity = float(nums[0])
    return np.array([instanceName, capacity, nodeMatrix])

def generateGaps():
    df =  pd.read_csv('output/dfCostsCWSDeterm.csv')
    df = df.reset_index()
    #df.insert(loc=2, column = 'Deterministic', value = dfCostsDeterm['Deterministic'].to_numpy())
    for i in range(1,11):
        dftemp = pd.read_csv('output/dfCostsCWSStoch_'+str(i)+'.csv')
        df.insert(loc=len(df.columns), column = 'Stochastic '+str(i), value = dftemp['Stochastic'].to_numpy())
    for i in range(1,11):
        df.insert(loc=len(df.columns), column = 'gap'+str(i), value = 100*(df['Stochastic '+str(i)] - df['Deterministic'])/df['Deterministic'])
    df.to_csv(r'./output/dfCostsScenarios.csv', index=False)
    return df 

def generateAdditionalData(instanceData):
    instanceName = instanceData[0]
    capacity = instanceData[1]
    df = pd.DataFrame(instanceData[2], columns = ['node','x', 'y', 'demand_mean'])
    x = np.absolute(df['x'][1:].to_numpy())
    y = np.absolute(df['y'][1:].to_numpy())
    demand_mean = df['demand_mean'][1:].to_numpy()
    f1 = np.absolute(np.sqrt(x**2+y**2)/(x+y+1))
    f2 = (np.sqrt(x**2+y**2+demand_mean**2)/np.absolute(x+y+demand_mean+1))
    supply_mean = np.ceil(demand_mean * f1)
    demand_desv = np.ceil(demand_mean*np.absolute(1-f1))
    supply_desv = np.ceil(supply_mean*np.absolute(1-f2))
    df['demand_desv'] = np.insert(demand_desv, 0, 0)
    df['supply_mean'] = np.insert(supply_mean, 0, 0)
    df['supply_desv'] = np.insert(supply_desv, 0, 0)
    df.to_csv(r'./instancesmod/'+instanceName+'_'+str(int(capacity))+'.csv', index=False)

def readNewInstancesDeterminsticData(fileName):
    #bucket_name, vehCaps, file
    df = pd.read_csv('instancesmod/'+fileName)
    instanceName = fileName.split('_')[0]
    capacity = fileName.split('_')[1].split('.')[0]
    nodeMatrix = []
    rows = df.values.tolist()
    for row in rows:
        #nodeMatrix.append([i, float(data['x']), float(data['y']), float(data['demand'])])
        nodeMatrix.append([row[0], row[1], row[2], row[3], row[5]])
    return [instanceName, int(capacity), nodeMatrix]

def readNewInstancesStochasticData(fileName):
    #bucket_name, vehCaps, file
    df = pd.read_csv('instancesmod/'+fileName)
    instanceName = fileName.split('_')[0]
    capacity = fileName.split('_')[1].split('.')[0]
    #fileobject = storage.get_object(params[0], params[2]['Key'], stream=True)
    #instanceName = str(params[2]['Key']).replace('instances/', '').replace('_input_nodes.txt', '')
    #dfPoints=pd.read_csv(fileobject, sep='\t', names=['x', 'y', 'demand'])
    nodeMatrix = []
    rows = df.values.tolist()
    for row in rows:
        #nodeMatrix.append([i, float(data['x']), float(data['y']), float(data['demand'])])
        nodeMatrix.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
    return [instanceName, int(capacity), nodeMatrix]


def generateDeterministicInstances(fileName):
    #bucket_name, vehCaps, file
    df = pd.read_csv('instancesmod/'+fileName)
    instanceName = fileName.split('_')[0]
    capacity = fileName.split('_')[1].split('.')[0]
    #fileobject = storage.get_object(params[0], params[2]['Key'], stream=True)
    #instanceName = str(params[2]['Key']).replace('instances/', '').replace('_input_nodes.txt', '')
    #dfPoints=pd.read_csv(fileobject, sep='\t', names=['x', 'y', 'demand'])
    i = 0
    nodeMatrix = []
    rows = df.values.tolist()
    for row in rows:
        #nodeMatrix.append([i, float(data['x']), float(data['y']), float(data['demand'])])
        nodeMatrix.append([i, row[0], row[1], row[2], row[3], row[4], row[5]])
        i += 1
    return [instanceName, int(capacity), nodeMatrix]

def SimCWS(buckets):

    #Read instances from the cloud and generate additional data
    readAndGenerateModifiedInstances = False
    if readAndGenerateModifiedInstances:
        files = []
        spec_storage = Storage()
        for i in buckets:
            try:
                files = spec_storage.list_objects(bucket  =i, prefix='instances/')
                bucket = i
                print(i + ' available')
            except:
                print(i + ' not available')
        del files[0]
        runtime  = 'lithopscloud/ibmcf-python-v38:2021.01'
        #Read instances
        fexec = lth.FunctionExecutor(runtime=runtime)
        paramlist = [[bucket, vehCaps, file] for file in files]
        fexec.map(readInstances, paramlist)
        instanceData = fexec.get_result()
        fexec.plot(dst='lithops_plots/Read')
        fexec.clean()
        shutil.rmtree('./instancesmod')
        os.mkdir('./instancesmod')
        #Add supply and deviations
        for instance in instanceData:
            generateAdditionalData(instanceData)
             
    shutil.rmtree('./output')
    os.mkdir('./output')

    # Read csvs
    filesmod = os.listdir('instancesmod/')
    instanceDataDeterm = []
    for file in filesmod:
        instanceDataDeterm.append(readNewInstancesDeterminsticData(file))
    instanceDataStoch = []
    for file in filesmod:
        instanceDataStoch.append(readNewInstancesStochasticData(file))

    runtime  = 'lithopscloud/ibmcf-python-v38:2021.01'
    #Deterministic case
    withParalelization1  = True
    if withParalelization1:
        pool = mp.Pool()
        costsCWSDeterm = pool.map(VRPPDDeterministic,instanceDataDeterm)
    else:
        costsCWSDeterm= []
        for instance in instanceDataDeterm:
            costsCWSDeterm.append(VRPPDDeterministic(instance))
    dfCostsCWSDeterm = pd.DataFrame(costsCWSDeterm,columns = ["Instance", "Capacity", "Deterministic"])
    print(dfCostsCWSDeterm)
    dfCostsCWSDeterm.to_csv(r'./output/dfCostsCWSDeterm.csv', index=False)
    
    #Stochastic case
    betas = [0.5]
    paramlist1 = list(it.product(instanceDataStoch,  betas))
    isCloud = True
    if isCloud:
        for i in range(6, 11):
            runtime  = 'lithopscloud/ibmcf-python-v38:2021.01'
            fexec1 = lth.FunctionExecutor(runtime=runtime)
            fexec1.map(VRPPDStochasticLithops, paramlist1, timeout=3000)
            costsCWSStoch = fexec1.get_result()
            fexec1.plot(dst='lithops_plots/Stochastic')
            fexec1.clean()
            dfCostsCWSStoch = pd.DataFrame(costsCWSStoch,columns = ["Instance", "Capacity", "Beta", "Stochastic"])
            print(dfCostsCWSStoch)
            dfCostsCWSStoch.to_csv(r'./output/dfCostsCWSStoch_'+str(i)+'.csv', index=False)
    else: 
        withParalelization2 = True
        if withParalelization2:
            pool = mp.Pool()
            costsCWSStoch = pool.map(VRPPDStochastic,paramlist1)
        else:
            costsCWSStoch= []
            for instance in paramlist1:
                costsCWSStoch.append(VRPPDStochastic(instance))
        columns = ["Instance", "Capacity", "Beta"]
        for i in range(1,11):
            columns.append('Stochastic ' +str(i))
        dfCostsCWSStoch = pd.DataFrame(costsCWSStoch,columns = columns)
        print(dfCostsCWSStoch)
        dfCostsCWSStoch.to_csv(r'./output/dfCostsCWSStoch.csv', index=False)

    dfCostAndGaps =generateColumns()
    #Save results
    #file1 = open('./output/beta03.text', 'a')
    #file1.write(dfBetas[0].to_latex())
    #file1.close()

    #file2 = open('./output/beta05.text', 'a')
    #file2.write(dfBetas[1].to_latex())
    #file2.close()

    #file3 = open('./output/beta08.text', 'a')
    #file3.write(dfBetas[2].to_latex())
    #file3.close()
    return dfCostAndGaps

def main():
    buckets = ['bucketlithops', 'lithopsbucket3']
    dfs = SimCWS(buckets)
    
if __name__ == '__main__':
    main()
    
    
    
    
