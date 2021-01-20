import os
from pathlib import Path
from io import StringIO
import sys
import cwsheuristiclithops as cws
import copy
import vrp_objects as vrp
import time
import pandas as pd 
import lithops as lth
import numpy as np
from lithops import Storage
import multiprocessing as mp
import itertools as it
import shutil
from ibm_botocore.client import Config
import ibm_boto3


def testCWS(instanceData):
    #instanceName, veCap, nodeMatrix
    instanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    instanceCws.runCWSSol()
    return np.array([instanceData[0],instanceData[1], instanceCws.getCost()])

def testSRGCWS(instanceData, nIterations, beta, isRCU, splittingType):
    #instanceData, nIterations, betas,  isRCUs, splittingTypes
    localInstanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    bestSol = vrp.Solution()
    bestSol.cost = 1000000000
    #startTime = time.time()
    minCost = 100000
    for iteration in range(nIterations):
        localInstanceCws.runCWSSolGeneral(beta, isRCU, splittingType)
        sol = localInstanceCws.getSolution()
        if sol.cost < minCost:
            bestSol = copy.deepcopy( sol )
            minCost = bestSol.cost
    #deltaTime = time.time() - startTime
    return [instanceData[0], instanceData[1], beta, isRCU, 'Random' if splittingType == 'Null' else splittingType, minCost]


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

def generateColumns(dfsCostsSRGCWS, dfCostsCWS):
    del dfsCostsSRGCWS['Beta']
    #columns = ['Instance', 'Capacity','RCU','SplittingType', 'Cost']
    dfTemp = dfsCostsSRGCWS.pivot_table(index =['Instance','Capacity'], columns =['RCU', 'SplittingType'], values = 'Cost')
    dfTemp.columns = [''.join((col[1],'RCU') if col[0] == True else (col[1],'')) for col in dfTemp.columns]
    dfTemp = dfTemp.reset_index()
    dataframe = pd.merge(dfCostsCWS, dfTemp, on=['Instance','Capacity'], how='outer')
    dataframe['gap1'] = 100*(dataframe['Random'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap2'] = 100*(dataframe['RandomRCU'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap3'] = 100*(dataframe['TopBottom'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap4'] = 100*(dataframe['LeftRight'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap5'] = 100*(dataframe['Cross'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap6'] = 100*(dataframe['Star'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap7'] = 100*(dataframe['TopBottomRCU'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap8'] = 100*(dataframe['LeftRightRCU'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap9'] = 100*(dataframe['CrossRCU'] - dataframe['Original'])/dataframe['Original']
    dataframe['gap10'] = 100*(dataframe['StarRCU'] - dataframe['Original'])/dataframe['Original']
    return dataframe.reset_index()

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
    #print(demand_mean**2/demand_desv**2)
    #print(supply_mean**2/supply_desv**2)
    df['demand_desv'] = np.insert(demand_desv, 0, 0)
    df['supply_mean'] = np.insert(supply_mean, 0, 0)
    df['supply_desv'] = np.insert(supply_desv, 0, 0)
    df.to_csv(r'./instancesmod/'+instanceName+'_'+str(int(capacity))+'.csv', index=False)


def beta(mean, std):
    n = 1000
    mu = mean/n
    sigma = std/n
    alfa = abs(((1-mu)/(sigma**2)-(1/mu))*mu**2)
    beta = abs(alfa*((1/mu)-1))
    return 1000*np.random.beta(alfa, beta)

def costsStocksSimulation(instanceFile, cost_demand, cost_supply, isStochastic, nIterations):
    df = pandas.read_csv('/instancesmod/'+str(instanceFile)+'.csv', encoding='utf-8')
    dfCosts = pd.DataFrame(columns = ['costs'])
    dfCostsTemp = pd.DataFrame()
    if isStochastic:
        for j in range(0,nIterations):
            dfCostsTemp.insert(loc=0, column='cost_'+str(j))
            for i in len(df):
                demand_mean = df['demand_mean'][i]
                demand_desv = df['demand_desv'][i]
                supply_mean = df['supply_mean'][i]
                supply_desv = df['supply_desv'][i]
                factor1 = demand_mean**2/demand_desv**2
                factor2 = supply_mean**2/supply_desv**2
                if factor1 < 15:
                    demand = beta(demand_mean, demand_desv) 
                else:
                    demand = numpy.random.lognormal(demand_mean, demand_desv)
                if factor2 < 15:
                    supply = beta(supply_mean, supply_desv) 
                else:
                    supply = numpy.random.lognormal(supply_mean, supply_desv)
                dfCostsTemp['cost_'+str(j)][i] = demand * cost_demand + supply * cost_supply
        dfCosts = dfCostsTemp.mean(axis=1)    
    else:
        for i in len(df):
            if i == 0:
                dfCosts['costs'][i] = 0
            else:
                dfCosts['costs'][i] = df['demand_mean'][i]*cost_demand + df['supply_mean'][i]*cost_supply
    return dfCosts

def readNewInstancesLocal(fileName):
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

    if True == False:
        files = []
        spec_storage = Storage()
        #'bucketlithops',
        for i in buckets:
            try:
                files = spec_storage.list_objects(bucket  =i, prefix='instances/')
                bucket = i 
                print(i + ' available') 
            except:
                print(i + ' not available')
        #files = spec_storage.list_objects(bucket  =bucket, prefix='instances/')
        #print(files)
        del files[0]

        runtime  = 'lithopscloud/ibmcf-python-v38:2021.01' 
        #Read instances
        fexec = lth.FunctionExecutor(runtime=runtime)
        #paramlist = list(it.product(['bucketlithops'],[vehCaps], files))
        paramlist = [[bucket, vehCaps, file] for file in files]
        fexec.map(readInstances, paramlist)
        instanceData = fexec.get_result()
        #fexec.plot(dst='lithops_plots/Read') 
        fexec.clean()
        #print(futs[0].status())
        shutil.rmtree('./instancesmod')
        os.mkdir('./instancesmod')
        
    shutil.rmtree('./output')
    os.mkdir('./output')

    # Read csvs
    filesmod = os.listdir('instancesmod/')
    print(filesmod)
    instanceData = []
    for file in filesmod:
        instanceData.append(readNewInstancesLocal(file))

    isLocal = True
    if isLocal:
        pool =mp.Pool(processes=12)
        costsCWS = pool.map(testCWS, instanceData)
        dfCostsCWS = pd.DataFrame(costsCWS,columns = ["Instance", "Capacity", "Original"])
    else:
        #CWS problem
        fexec1 = lth.FunctionExecutor(runtime=runtime)
        fexec1.map(testCWS, instanceData)
        costsCWS = fexec1.get_result()
        fexec1.plot(dst='lithops_plots/CWS') 
        fexec1.clean()

    print(dfCostsCWS)
    #SRGCWS problem
    #Data
    nIterations = [100]
    #isRCUs = [False, True]
    rcPolicies = ['NoRefill', 'Refill1/4', 'Refill1/2', 'Refill3/4', 'RefillFull', 'RefillOpt']
    paramlist2 = list(it.product(instanceData, nIterations, betas,  isRCUs, splittingTypes))

    #Execution
    fexec2 = lth.FunctionExecutor(runtime=runtime)
    fexec2.map(testSRGCWS, paramlist2)
    costSRGCWS = fexec2.get_result()
    fexec2.plot(dst='lithops_plots/SRGCWS') 
    fexec2.clean()

    dfCostsSRGCWS = pd.DataFrame(costSRGCWS, columns = ['Instance', 'Capacity', 'Beta', 'RCU','SplittingType', 'Cost'])
    
    #Get one dataframe for each beta
    dfsCostsSRGCWS  = [dfCostsSRGCWS[dfCostsSRGCWS['Beta']==0.3], dfCostsSRGCWS[dfCostsSRGCWS['Beta']==0.5], dfCostsSRGCWS[dfCostsSRGCWS['Beta']==0.8]]

    dfsCostsSRGCWSBeta = []
    #Reshape dataframes
    for df in dfsCostsSRGCWS:
        dfsCostsSRGCWSBeta.append(generateColumns(df,dfCostsCWS))

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
    return [dfCostsCWS, dfsCostsSRGCWSBeta[0], dfsCostsSRGCWSBeta[1], dfsCostsSRGCWSBeta[2]]

def main():
    buckets = ['bucketlithops', 'lithopsbucket3']
    dfs = SimCWS(buckets)
    #dfs = VRP_SRCWS('bucketlithops')
    #dfs = VRP_SRCWS('lithopsbucket3')
    dfs[0].to_csv(r'./output/dfCostsCWS.csv', index=False)
    dfs[1].to_csv(r'./output/beta03.csv', index=False)
    dfs[2].to_csv(r'./output/beta05.csv', index=False)
    dfs[3].to_csv(r'./output/beta08.csv', index=False)
    
if __name__ == '__main__':
    main()
    
    
    
    
