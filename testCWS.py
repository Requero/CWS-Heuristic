import os
from pathlib import Path
import sys
import cwsheuristic as cws
import copy
import vrp_objects as vrp
import time
import pandas as pd 
import multiprocessing as mp
import itertools as it


def testCWS(instanceData):
    #instanceName, veCap, nodeMatrix
    instanceCws = cws.HeuristicSequential(instanceData[0], instanceData[1], instanceData[2])
    instanceCws.runCWSSol()
    return [instanceData[0],instanceData[1], instanceCws.getCost()]

def testSRGCWS(params):
    #instanceData, beta, nIterations, isRCU, splittingType
    localInstanceCws = cws.HeuristicSequential(params[0][0], params[0][1], params[0][2])
    bestSol = vrp.Solution()
    bestSol.cost = 1000000000
    startTime = time.time()
    for iteration in range(params[1]):
        localInstanceCws.runCWSSolGeneral(params[2], params[3], params[4])
        sol = localInstanceCws.getSolution()
        if sol.cost < bestSol.cost:
            bestSol = copy.deepcopy( sol )
    deltaTime = time.time() - startTime
    return [params[0][0], params[0][1], params[2], params[3], params[4], bestSol.cost] #, deltaTime

def readNodes(filename):
    dirname, f = os.path.split(os.path.abspath(__file__))
    dirname = dirname + "\\instances"
    instanceName = str(filename).replace(dirname +"\\", '').replace('_input_nodes.txt', '')
    #replace('.txt')
    with open(filename) as instance:
        costs = []
        i = 0
        nodeMatrix = []
        for line in instance:
            # array  data with node data: x, y, demand
            data = [float(x) for x in line.split()]
            nodeMatrix.append([i, data[0], data[1], data[2]])
            i += 1
        #.update({instanceName : nodeMatrix})
    return (instanceName, nodeMatrix)

def generateColumns(dfs):
    dataframe = pd.DataFrame()
    #dataframe["Instance", "Capacity", "Original", "Random", "RandomRCU", "TopBottom","LeftRight", "Cross", "Star","TopBottomRCU", "LeftRightRCU","CrossRCU","StarRCU"]
    dataframe = dfs[1].copy()
    dataframe['Random'] = dfs[0].loc[(dfs[0]['RCU'] == False )&( dfs[0]['SplittingType'] == 'Null'), ['Cost']].values
    dataframe['RandomRCU'] = dfs[0].loc[(dfs[0]['RCU'] == True )&( dfs[0]['SplittingType'] == 'Null'), ['Cost']].values
    dataframe['TopBottom'] = dfs[0].loc[(dfs[0]['RCU'] == False )&( dfs[0]['SplittingType'] == 'TopBottom'), ['Cost']].values
    dataframe['LeftRight'] =  dfs[0].loc[(dfs[0]['RCU'] == False )&( dfs[0]['SplittingType'] == 'LeftRight'), ['Cost']].values
    dataframe['Cross'] =  dfs[0].loc[(dfs[0]['RCU'] == False )&( dfs[0]['SplittingType'] == 'Cross'), ['Cost']].values
    dataframe['Star'] =  dfs[0].loc[(dfs[0]['RCU'] == False )&( dfs[0]['SplittingType'] == 'Star'), ['Cost']].values
    dataframe['TopBottomRCU'] =  dfs[0].loc[(dfs[0]['RCU'] == True )&( dfs[0]['SplittingType'] == 'TopBottom'), ['Cost']].values
    dataframe['LeftRightRCU'] =  dfs[0].loc[(dfs[0]['RCU'] == True )&( dfs[0]['SplittingType'] == 'LeftRight'), ['Cost']].values
    dataframe['CrossRCU'] =  dfs[0].loc[(dfs[0]['RCU'] == True )&( dfs[0]['SplittingType'] == 'Cross'), ['Cost']].values
    dataframe['StarRCU'] =  dfs[0].loc[(dfs[0]['RCU'] == True )&( dfs[0]['SplittingType'] == 'Star'), ['Cost']].values
     
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
    
    dataframe.reset_index()
    return dataframe

def main():

    dirname, f = os.path.split(os.path.abspath(__file__))
    dirname = dirname + "\\instances"
    txt_folder = Path(dirname).rglob('*.txt')
    files = [x for x in txt_folder]
    
    dataframe = pd.DataFrame(columns = ["Instance", "Capacity", "Cost", "CostBeta1", "CostBeta2", "CostBeta3"])
    

    vehCaps = dict()
    vehCaps["A-n32-k5"]=100 
    vehCaps["A-n38-k5"]=100
    vehCaps["A-n45-k7"]=100
    vehCaps["A-n55-k9"]=100
    vehCaps["A-n60-k9"]=100
    vehCaps["A-n61-k9"]=100
    vehCaps["A-n65-k9"]=100
    vehCaps["A-n80-k10"]=100 # A 
    vehCaps["B-n50-k7"]=100
    vehCaps["B-n52-k7"]=100
    vehCaps["B-n57-k9"]=100
    vehCaps["B-n78-k10"]=100 # B
    vehCaps["E-n22-k4"]=6000
    vehCaps["E-n30-k3"]=4500
    vehCaps["E-n33-k4"]=8000
    vehCaps["E-n51-k5"]=160
    vehCaps["E-n76-k7"]=220
    vehCaps["E-n76-k10"]=140
    vehCaps["E-n76-k14"]=100 # E
    vehCaps["F-n45-k4"]=2010
    vehCaps["F-n72-k4"]=30000
    vehCaps["F-n135-k7"]=2210 # F
    vehCaps["M-n101-k10"]=200
    vehCaps["M-n121-k7"]=200 # M
    vehCaps["P-n22-k8"]=3000
    vehCaps["P-n40-k5"]=140
    vehCaps["P-n50-k10"]=100
    vehCaps["P-n55-k15"]=70
    vehCaps["P-n65-k10"]=130
    vehCaps["P-n70-k10"]=135
    vehCaps["P-n76-k4"]=350
    vehCaps["P-n76-k5"]=280
    vehCaps["P-n101-k4"]=400 # P
    print(vehCaps)
    
    pool =mp.Pool(processes=12)
    instancesNodes = pool.map(readNodes, files)
    instanceData = []
    for pair in instancesNodes:
        instanceData.append([pair[0],vehCaps[pair[0]],pair[1]])

    #Original problem
    costsCWS = pool.map(testCWS, instanceData)
    dfCostsCWS = pd.DataFrame(costsCWS,columns = ["Instance", "Capacity", "Original"])
    print(dfCostsCWS)

    # Initial data
    betas = [0.3, 0.5, 0.8] #Different betas to test differente behaviours (risky, normal, conservative)
    nIterations = [100]
    isRCUs = [False, True]
    splittingTypes = ['Null', 'TopBottom', 'LeftRight', "Cross", "Star"]
    paramlist = list(it.product(instanceData, nIterations, betas,  isRCUs, splittingTypes))

    # Costs
    costSRGCWS = pool.map(testSRGCWS, paramlist)
    dfCostSRGCWS = pd.DataFrame(costSRGCWS, columns = ["Instance", "Capacity", "Beta", "RCU","SplittingType", "Cost"])
    dfsCostSRGCWS  = [dfCostSRGCWS[dfCostSRGCWS["Beta"]==0.3], dfCostSRGCWS[dfCostSRGCWS["Beta"]==0.5], dfCostSRGCWS[dfCostSRGCWS["Beta"]==0.8]]

    paramlist1 = list(it.product(dfsCostSRGCWS, [dfCostsCWS]))
    dfBetas = pool.map(generateColumns, paramlist1)
    
    print(dfBetas[0])
    print(dfBetas[1])
    print(dfBetas[2])

    dfBetas[0].to_csv('output\\beta03.csv', header=None, index=False, mode='a')
    dfBetas[1].to_csv('output\\beta05.csv', header=None, index=False, mode='a')
    dfBetas[2].to_csv('output\\beta08.csv', header=None, index=False, mode='a')

    file1 = open('output\\beta03.text', 'a')
    file1.write(dfBetas[0].to_latex())
    file1.close()

    file2 = open('output\\beta05.text', 'a')
    file2.write(dfBetas[1].to_latex())
    file2.close()

    file3 = open('output\\beta08.text', 'a')
    file3.write(dfBetas[2].to_latex())
    file3.close()


if __name__ == '__main__':
    main()
    
    
    
    
