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
import itertools as it
import shutil
from ibm_botocore.client import Config
import ibm_boto3


def testCWS(instanceData):
    #instanceName, veCap, nodeMatrix
    instanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    instanceCws.runCWSSol()
    return np.array([instanceData[0],instanceData[1], instanceCws.getCost()])

def iterationSRGCWS(iteration, instanceData, beta, isRCU, splittingType):
    localInstanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    localInstanceCws.runCWSSolGeneral(beta, isRCU, splittingType)
    return localInstanceCws.getSolution().cost

def testSRGCWSLithops(instanceData, nIterations, beta, isRCU, splittingType):
    rangeIterations = list(range(0,nIterations))
    paramlist = list(it.product(rangeIterations, [instanceData], [beta],[isRCU], [splittingType]))
    fexec = lth.FunctionExecutor()
    fexec.map(iterationSRGCWS, paramlist)
    minCosts = fexec.get_result()
    fexec.clean()
    minCost = min(minCosts)
    return [minCost]#[instanceData[0], instanceData[1], beta, isRCU, 'Random' if splittingType == 'Null' else splittingType, minCost] #, deltaTime

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
    df['demand_desv'] = np.insert(demand_desv, 0, 0)
    df['supply_mean'] = np.insert(supply_mean, 0, 0)
    df['supply_desv'] = np.insert(supply_desv, 0, 0)
    df.to_csv(r'./instancesmod/'+instanceName+'_'+str(int(capacity))+'.csv', index=False)
                    
def SimCWS(buckets):

    files = []
    spec_storage = Storage()
    for i in buckets:
        try:
            files = spec_storage.list_objects(bucket  =i, prefix='Instances3/')
            bucket = i 
            print(i + ' availiable') 
        except:
            print(i + ' not availiable')
    del files[0]
    
    runtime  = 'lithopscloud/ibmcf-python-v38:2021.01'
    
    #Read instances
    fexec = lth.FunctionExecutor(runtime=runtime)
    paramlist =[]
    files = np.reshape(files, (int(len(files)/2),2))
    for file in files:
        paramlist.append([bucket, file[0], file[1]])
    fexec.map(readInstances, paramlist)
    instanceData = fexec.get_result()
    fexec.clean()
    shutil.rmtree('./output')
    os.mkdir('./output')

    shutil.rmtree('./instancesmod')
    os.mkdir('./instancesmod')
    for i in instanceData:
        generateAdditionalData(i)
    #CWS problem
    fexec1 = lth.FunctionExecutor(runtime=runtime)
    fexec1.map(testCWS, instanceData)
    costsCWS = fexec1.get_result()
    fexec1.plot(dst='lithops_plots/CWS') 
    fexec1.clean()
    dfCostsCWS = pd.DataFrame(costsCWS,columns = ['Instance', 'Capacity', 'Original'])

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
    
    
    
    
