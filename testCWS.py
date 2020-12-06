import os
from pathlib import Path
import sys
import cwsheuristic as cws
import copy
import vrp_objects as vrp
import time
import pandas as pd 
import multiprocessing as mp
vehCap = 100.0 # update vehicle capacity for each instance
#instanceName = 'A-n80-k10' # name  of the instance
# txt file with the VRP instance data (nodeID, x, y, demand)
#fileName = 'instance/' + instanceName + '_input_nodes.txt'

def testBiasedStart(nIterations, beta, localInstanceCws, isPloted):
    
    bestSol = vrp.Solution()
    bestSol.cost = 1000000000
    startTime = time.time()
    for iteration in range(nIterations):
        localInstanceCws.runRandomSol(beta)
        sol = localInstanceCws.getSolution()
        if sol.cost < bestSol.cost:
            bestSol = copy.deepcopy( sol )
            #print( "New best sol: " + str(bestSol.cost) )
            
    deltaTime = time.time() - startTime
    #print( "Done in [" + "{:{}f}".format( (deltaTime),2) + "s]" )
    #localInstanceCws.printRouteCostsBestSolution( bestSol )
    if isPloted == True: 
        localInstanceCws.plotGraph()
    return bestSol.cost, deltaTime
    
    
def testBiasedStartWithRouteCache(nIterations, beta, localInstanceCws, isPloted):
    
    localInstanceCws.enableRCU()
    bestSol = vrp.Solution()
    bestSol.cost = 1000000000
    startTime = time.time()
    for iteration in range(nIterations):
        localInstanceCws.runRandomSol(beta)
        sol = localInstanceCws.getSolution()
        if sol.cost < bestSol.cost:
            bestSol = copy.deepcopy( sol )
            #print( "New best sol: " + str(bestSol.cost) )
    
    deltaTime = time.time() - startTime
    #print( "Done in [" + "{:{}f}".format( (deltaTime),2) + "s]" )
    #localInstanceCws.printRouteCostsBestSolution( bestSol )
    if isPloted == True: 
        localInstanceCws.plotGraph()
    return bestSol.cost
    
    
def testBiasedStartWithSplittingTechniques(nIterations, beta, localInstanceCws, isPloted):

    splittingTypes = ['TopBottom', 'LeftRight', "Cross", "Star"]
    for split in splittingTypes:
        #print( "----------" + split + "----------" )
        bestSol = vrp.Solution()
        bestSol.cost = 1000000000
        startTime = time.time()
        for iteration in range(nIterations):
            localInstanceCws.runSplittingSol(beta, split)
            sol = localInstanceCws.getSolution()
            if sol.cost < bestSol.cost:
                bestSol = copy.deepcopy( sol )
                #print( "New best solution: " + str(bestSol.cost) )
                
        deltaTime = time.time() - startTime
        #print( "Done in [" + "{:{}f}".format( (deltaTime),2) + "s]" )
        #localInstanceCws.printRouteCostsBestSolution( bestSol )
        if isPloted == True: 
            localInstanceCws.plotGraph()
        return bestSol.cost #, deltaTime
    
    
def testBiasedStartWithSplittingTechniquesAndRouteCache(nIterations, beta, localInstanceCws, isPloted):
    
    localInstanceCws.enableRCU()
    splittingTypes = ['TopBottom', 'LeftRight', "Cross", "Star"]
    for split in splittingTypes:
        #print( "----------" + split + "----------" )
        bestSol = vrp.Solution()
        bestSol.cost = 1000000000
        startTime = time.time()
        for iteration in range(nIterations):
            localInstanceCws.runSplittingSol(beta, split)
            sol = localInstanceCws.getSolution()
            if sol.cost < bestSol.cost:
                bestSol = copy.deepcopy( sol )
                #print( "New best solution: " + str(bestSol.cost) )
        
        deltaTime = time.time() - startTime
        #print( "Done in [" + "{:{}f}".format( (deltaTime),2) + "s]" )
        #localInstanceCws.printRouteCostsBestSolution( bestSol )
        if isPloted == True: 
            localInstanceCws.plotGraph()
        return bestSol.cost #, deltaTime
        
        
def main():
    vehCaps = {
        "A-n32-k5" : 100, 
        "A-n38-k5" : 100,
        "A-n45-k7" : 100,
        "A-n55-k9" : 100,
        "A-n60-k9" : 100,
        "A-n61-k9" : 100,
        "A-n65-k9" : 100,
        "A-n80-k10" : 100, # A 
        "B-n50-k7" : 100,
        "B-n52-k7" : 100,
        "B-n57-k9" : 100,
        "B-n78-k10" : 100, # B
        "E-n22-k4" : 6000,
        "E-n30-k3" : 4500,
        "E-n33-k4" : 8000,
        "E-n51-k5" : 160,
        "E-n76-k7" : 220,
        "E-n76-k10" : 140,
        "E-n76-k14" : 100, # E
        "F-n45-k4" : 2010,
        "F-n72-k4" : 30000,
        "F-n135-k7" : 2210, # F
        "M-n101-k10" : 200,
        "M-n121-k7" : 200, # M
        "P-n22-k8" : 3000,
        "P-n40-k5" : 140,
        "P-n50-k10" : 100,
        "P-n55-k15" : 70,
        "P-n65-k10" : 130,
        "P-n70-k10" : 135,
        "P-n76-k4" : 350,
        "P-n76-k5" : 280,
        "P-n101-k4" : 400 # P
    }
    dirname, f = os.path.split(os.path.abspath(__file__))
    dirname = dirname + "\\instances"
    txt_folder = Path(dirname).rglob('*.txt')
    betas = [0.3, 0.5, 0.8] #Different betas to test differente behaviours (risky, normal, conservative)
    files = [x for x in txt_folder]
    
    dataframe = pd.DataFrame(columns = ["Instance", "Capacity", "Cost", "CostBeta1", "CostBeta2", "CostBeta3"])
    
    for filename in files:  
        instanceName = str(filename).replace(dirname +"\\", '').replace('_input_nodes.txt', '')
        #replace('.txt')
        with open(filename) as instance:
            costs = []
            nodeMatrix = []
            i = 0
            for line in instance:
                # array  data with node data: x, y, demand
                data = [float(x) for x in line.split()]
                nodeMatrix.append([i, data[0], data[1], data[2]])
                i += 1
            instanceCws = cws.HeuristicSequential(instanceName, nodeMatrix, vehCaps[instanceName])
            instanceCws.startInstance()
            #instanceCopy = copy.deepcopy(instanceCws)
            # if instanceName == "E-n76-k10" or instanceName == "E-n76-k14" :
            #Ejecutamos primero la solucion básica
            startTime = time.time()
            instanceCws.runCWSSol()
            deltaTime = time.time() - startTime
            costs.append(instanceCws.getCost())
            
            instance1 = copy.deepcopy(instanceCws)
            instance2 = copy.deepcopy(instanceCws)
            instance3 = copy.deepcopy(instanceCws)
            
            N=100
            pool =mp.Pool(processes=3)
            tempcosts = pool.starmap(testBiasedStartWithSplittingTechniquesAndRouteCache, [
                (N, betas[0], instance1, False),
                (N, betas[1], instance2, False),
                (N, betas[2], instance3, False)])
            
            #for beta in betas:
                #f = open("output\\"+str(filename).split("\\")[-1].split("_")[0] + "_out_" + str(beta)+".txt","a")
                #f.truncate(0)
                #old_stdout = sys.stdout
                #sys.stdout = f
                
                #print( "Done in [" + "{:{}f}".format( (deltaTime),2) + "s]" )
                
                #instanceCws.printCost()
                #instanceCws.printRouteCosts()
                
                #N = 100;
                #Comentad todas las funciones menos el test que vayais a correr
                #costBS, dTimeBS = testBiasedStart(N, beta, instanceCwsBS, False)
                #costBSRC, dTimeBSRC = testBiasedStartWithRouteCache(N, beta, instanceCwsBSRC, False)
                #costBSST, dTimeBSST = testBiasedStartWithSplittingTechniques(N, beta, instanceCwsBSST, False)
                #costBSSTRC, dTimeBSSTRC = testBiasedStartWithSplittingTechniquesAndRouteCache(N, beta, instanceCws, False)
                
                #costs.append(costBSSTRC)
                #sys.stdout = old_stdout
                #f.close()
            print(instanceName)
            dataframe = dataframe.append(pd.Series([instanceName, vehCaps[instanceName], costs[0], tempcosts[0], tempcosts[1], tempcosts[2]], index=dataframe.columns), ignore_index=True)
    print(dataframe)

if __name__ == '__main__':
    main()