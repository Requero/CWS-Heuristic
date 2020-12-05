import os
from pathlib import Path
import sys
import cwsheuristic as cws
import copy
import vrp_objects as vrp

vehCap = 100.0 # update vehicle capacity for each instance
#instanceName = 'A-n80-k10' # name  of the instance
# txt file with the VRP instance data (nodeID, x, y, demand)
#fileName = 'instance/' + instanceName + '_input_nodes.txt'

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
betas = [0.3] #Different betas to test differente behaviours (risky, normal, conservative)
files = [x for x in txt_folder]
i = 0
for filename in files:  
    instanceName = str(filename).replace(dirname +"\\", '').replace('_input_nodes.txt', '')
    #replace('.txt')
    with open(filename) as instance:
        if i < 1:
            i += 1
            instanceCws = cws.HeuristicSequential(instanceName, instance, vehCaps[instanceName])
            # if instanceName == "E-n76-k10" or instanceName == "E-n76-k14" :
            for beta in betas:
                f = open("output\\"+str(filename).split("\\")[-1].split("_")[0] + "_out_" + str(beta)+".txt","a")
                f.truncate(0)
                old_stdout = sys.stdout
                sys.stdout = f
                #Ejecutamos primero la solucion básica
                instanceCws.runCWSSol()
                #instanceCws.runSplittingSol("Star")
                instanceCws.printCost()
                instanceCws.printRouteCosts()
                
                #Ahora viene la solucion randomizada con iteraciones
                bestSol = vrp.Solution()
                bestSol.cost = 1000000000
                N = 200;
                for i in range(N):
                    instanceCws.runRandomSol(beta)
                    sol = instanceCws.getSolution()
                    if sol.cost < bestSol.cost:
                        bestSol = copy.deepcopy( sol )
                        print( "New best sol: " + str(bestSol.cost) )
                        
                instanceCws.printRouteCostsBestSolution( bestSol )
                #instanceCws.printBestRoutes()
                instanceCws.plotGraph()
                
                sys.stdout = old_stdout
                f.close()