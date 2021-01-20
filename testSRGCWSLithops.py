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
from lithops import Storage
import itertools as it
import shutil
from ibm_botocore.client import Config
import ibm_boto3


def testCWS(instanceData):
    #instanceName, veCap, nodeMatrix
    instanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    instanceCws.runCWSSol()
    return [instanceData[0],instanceData[1], instanceCws.getCost()]

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

def testSRGCWS(instanceData, nIterations, beta):
    #instanceData, nIterations, betas,  isRCUs, splittingTypes
    localInstanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
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
    fileobject = storage.get_object(params[0], params[2]['Key'], stream=True)
    instanceName = str(params[2]['Key']).replace('instances/', '').replace('_input_nodes.txt', '')
    #dfPoints=pd.read_csv(fileobject, sep='\t', names=['x', 'y', 'demand'])
    i = 0
    nodeMatrix = []
    lines = fileobject.read().decode('ascii').split('\n')
    #for i, data in dfPoints.iterrows():
    for line in lines:
        nums = line.split('\t')
        #nodeMatrix.append([i, float(data['x']), float(data['y']), float(data['demand'])])
        nodeMatrix.append([i, float(nums[0]), float(nums[1]), float(nums[2])])
        i += 1
    return [instanceName, params[1][instanceName], nodeMatrix]

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

def VRP_SRCWS(buckets):
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

    shutil.rmtree('./output')
    os.mkdir('./output')

    filesmod = os.listdir('instancesmod/')
    print(filesmod)
    instances = []
    for file in filesmod:
        instances.append(readNewInstancesLocal(file))
    print(instances)
    #CWS problem
    fexec1 = lth.FunctionExecutor(runtime=runtime)
    fexec1.map(testCWS, instanceData)
    costsCWS = fexec1.get_result()
    fexec1.plot(dst='lithops_plots/CWS') 
    fexec1.clean()
    dfCostsCWS = pd.DataFrame(costsCWS,columns = ['Instance', 'Capacity', 'Original'])
    #print(dfCostsCWS)

    #SRGCWS problem
    #Data
    betas = [0.5] 
    nIterations = [100]
    paramlist2 = list(it.product(instanceData, nIterations, betas))

    #Execution
    fexec2 = lth.FunctionExecutor(runtime=runtime)
    fexec2.map(testSRGCWS, paramlist2)
    costSRGCWS = fexec2.get_result()
    fexec2.plot(dst='lithops_plots/SRGCWS') 
    fexec2.clean()
        

    dfCostsSRGCWS = pd.DataFrame(costSRGCWS, columns = ['Instance', 'Capacity', 'Beta', 'Cost'])
    #print(dfCostSRGCWS)
    
    #Get one dataframe for each beta
    dfsCostsSRGCWS  =  dfCostsSRGCWS[dfCostsSRGCWS['Beta']==0.5]

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
    return [dfCostsCWS, dfsCostsSRGCWSBeta[0]]

def main():
    buckets = ['bucketlithops', 'lithopsbucket3']
    dfs = VRP_SRCWS(buckets)
    #dfs = VRP_SRCWS('bucketlithops')
    #dfs = VRP_SRCWS('lithopsbucket3')
    dfs[0].to_csv(r'./output/dfCostsCWS.csv', index=False)
    dfs[2].to_csv(r'./output/beta05.csv', index=False)
    
if __name__ == '__main__':
    main()
    
    
    
    
