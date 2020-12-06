""" Clarke & Wright savings heuristic sfor the VRP """

import copy
from vrp_objects import Node, Edge, Route, Solution
import math
import operator
import networkx as nx
import matplotlib.pyplot as plt
import random

""" Read instance data fromm txt file """

class HeuristicSequential:
    # Fields
    # instanceName
    # vehCap
    # nodes
    # depot
    # savingsList
    # sol 
    
    def __init__(self, instanceName, nodeMatrix, vehCap):
        self.instanceName=instanceName
        self.vehCap = vehCap
        #with open(fileName) as instance:
        i = 0
        self.nodes = []
        for nodeData in nodeMatrix:
            # array  data with node data: x, y, demand
            aNode = Node(nodeData[0], nodeData[1], nodeData[2], nodeData[3])
            self.nodes.append(aNode)
        self.bestRoutes = {} # best routes regarding cost 
        self.enabledRouteCacheUsage = False
        
    def startInstance(self):
        self.constructEdges(self.nodes)
        self.sol = Solution()
        
    def enableRCU(self):
        self.enabledRouteCacheUsage = True
        
    def runCWSSol(self):
        self.constructDummySolution(self.nodes) 
        self.edgeSelectionRoutingMerging(self.savingsList)

    def runRandomSol(self, beta=0.3):
        self.constructDummySolution(self.nodes) 
        biasedList = self.generateBiasedSavingsList(beta)
        self.edgeSelectionRoutingMerging(biasedList)
        
        if( self.enabledRouteCacheUsage ):
            self.improveSolutionWithBestRoutesFound()

    def runSplittingSol(self, beta=0.3, splittingType="TopBottom" ):
        splittingTypes = ['TopBottom', 'LeftRight', "Cross", "Star"]
        if splittingType not in splittingTypes:
            raise ValueError("Invalid sim type. Expected one of: %s" % splittingTypes)
        if splittingType == "TopBottom":
            splt_Nodes = self.splitTopBottomNodes()
        if splittingType == "LeftRight":
            splt_Nodes = self.splitLeftRightNodes()
        if splittingType == "Cross":
            splt_Nodes = self.splitCrossNodes()
        if splittingType == "Star": #8 cuadrants
            splt_Nodes = self.splitStarNodes()
            
        self.sol = Solution()
        for splt_node in splt_Nodes:
            self.bestRoutes = {}
            self.constructEdges(splt_node)
            self.constructDummySolution(splt_node)
            biasedList = self.generateBiasedSavingsList(beta)
            self.edgeSelectionRoutingMerging(biasedList)
            
            if( self.enabledRouteCacheUsage ):
                self.improveSolutionWithBestRoutesFound()

    def splitTopBottomNodes(self): #Split the nodes depending on their position with respect to the Y axis.
        splitted_nodes = [[self.nodes[0]], [self.nodes[0]]] #Depot node is included in both splitted lists
        for node in self.nodes[1:]:
            if node.y >= 0: #Positive Y nodes will be added to the first list
                splitted_nodes[0].append(node)
            else: #Negative Y nodes will be added to the second list
                splitted_nodes[1].append(node)
        return splitted_nodes

    def splitLeftRightNodes(self): #Split the nodes depending on their position with respect to the X axis.
        splitted_nodes = [[self.nodes[0]], [self.nodes[0]]] #Depot node is included in both splitted lists
        for node in self.nodes[1:]:
            if node.x >= 0: #Positive X nodes will be added to the first list
                splitted_nodes[0].append(node)
            else: #Negative X nodes will be added to the second list
                splitted_nodes[1].append(node)
        return splitted_nodes
        
    def splitCrossNodes(self): #Split the nodes depending on their position with respect to the X  and Y axis.
        splitted_nodes = [[self.nodes[0]], [self.nodes[0]], [self.nodes[0]], [self.nodes[0]]]#Depot node is included in all splitted lists
        for node in self.nodes[1:]:
            if node.x >= 0 and node.y >=0: #Positive cuadrant nodes will be added to the first list
                splitted_nodes[0].append(node)
            if node.x >= 0 and node.y < 0: #Positive X and negative Y nodes will be added to the second list
                splitted_nodes[1].append(node)
            if node.x < 0 and node.y < 0: #Negative X and negative Y nodes will be added to the third list
                splitted_nodes[2].append(node)
            if node.x < 0 and node.y >=0: #Negative X and positive Y nodes will be added to the forth list
                splitted_nodes[3].append(node)
        return splitted_nodes

    def splitStarNodes(self): 
        splitted_nodes = [[self.nodes[0]], [self.nodes[0]], [self.nodes[0]], [self.nodes[0]], [self.nodes[0]], [self.nodes[0]], [self.nodes[0]], [self.nodes[0]]] #Depot node is included in all splitted lists
        for node in self.nodes[1:]:
            if node.x >= 0 and node.y >=0: #Positive cuadrant
                if node.x >= node.y:
                    splitted_nodes[0].append(node)
                else:
                    splitted_nodes[1].append(node)
            if node.x >= 0 and node.y < 0: #Positive X and negative Y
                if abs(node.x) >= abs(node.y):
                    splitted_nodes[2].append(node)
                else:
                    splitted_nodes[3].append(node)
            if node.x < 0 and node.y < 0: #Negative X and negative Y
                if abs(node.x) >= abs(node.y):
                    splitted_nodes[4].append(node)
                else:
                    splitted_nodes[5].append(node)
            if node.x < 0 and node.y >=0: #Negative X and positive Y
                if abs(node.x) >= abs(node.y):
                    splitted_nodes[6].append(node)
                else:
                    splitted_nodes[7].append(node)
        return splitted_nodes

    def getRouteHash(self, route ):
        routeIds = [0]
        for edge in route.edges:
            routeIds.append(edge.end.ID);
        routeIds.sort()
        
        routeKey = str(0)
        for x in routeIds[1:]:
            routeKey += "_" + str(x)
            
        return routeKey
        
    def improveSolutionWithBestRoutesFound(self):
        for route in self.sol.routes:
            routeKey = self.getRouteHash( route )
        
            if routeKey in self.bestRoutes.keys():
                if self.bestRoutes[routeKey].cost < route.cost:
                    self.sol.cost -= route.cost
                    self.sol.cost += self.bestRoutes[routeKey].cost
                    route = self.bestRoutes[routeKey]
            
        
    def generateBiasedSavingsList(self, beta):
        copySavings = self.savingsList.copy()
        biasedSavings = []
        for i in range( len(copySavings) ):
            index = int( math.log(random.random()) / math.log(1 - beta) )
            index = index % len(copySavings)
            biasedSavings.append( copySavings[index] )
            copySavings.pop( index )
        return biasedSavings
    
    
    def constructEdges(self, nodes):
        """ Construct edges with costs and savings list from self.nodes """
        self.depot = nodes[0] # node 0 is self.depot
        
        for node in nodes[1:]: # excludes the self.depot
            dnEdge = Edge(self.depot, node) # creates the (self.depot, node) edge (arc)
            ndEdge = Edge(node, self.depot)
            dnEdge.invEdge = ndEdge # sets the inverse edge (arc)
            ndEdge.invEdge = dnEdge
            # compute the Euclidean distance as cost
            dnEdge.cost = math.sqrt((node.x - self.depot.x)**2 + (node.y - self.depot.y)**2)
            ndEdge.cost = dnEdge.cost # assume symmetric costs
            # save in node a reference to the (self.depot, node) edge (arc)
            node.dnEdge = dnEdge
            node.ndEdge = ndEdge
        
        self.savingsList = []
        for i in range(1, len(nodes) - 1): # excludes the self.depot
            iNode = nodes[i]
            for j in range(i + 1, len(nodes)):
                jNode = nodes[j]
                ijEdge = Edge(iNode, jNode) # creates the (i, j) edge
                jiEdge = Edge(jNode, iNode)
                ijEdge.invEdge = jiEdge # sets the inverse edge (arc)
                jiEdge.invEdge = ijEdge
                # compute the Euclidean distance as cost
                ijEdge.cost = math.sqrt((jNode.x - iNode.x)**2 + (jNode.y - iNode.y)**2)
                jiEdge.cost = ijEdge.cost # assume symmetric costs
                # compute savings as proposed by Clark & Wright
                ijEdge.savings = iNode.ndEdge.cost + jNode.dnEdge.cost -ijEdge.cost
                jiEdge.savings = ijEdge.savings
                # save one edge in savings list
                self.savingsList.append(ijEdge)
        # sort the list of edges from higher to lower savings
        self.savingsList.sort(key = operator.attrgetter("savings"), reverse = True)
    
    def constructDummySolution(self, nodes):
        """ Construct the dummy solution """
        
        for node in nodes[1:]: # excludes the self.depot
            dnEdge = node.dnEdge # get the(self.depot, node) edge
            ndEdge = node.ndEdge
            dndRoute = Route() # construct the route (self.depot, node, self.depot)
            dndRoute.edges.append(dnEdge)
            dndRoute.demand += node.demand
            dndRoute.cost += dnEdge.cost
            dndRoute.edges.append(ndEdge)
            dndRoute.cost += ndEdge.cost 
            node.inRoute = dndRoute # save in node a reference to its current route
            node.isInterior = False # this node  is  currently exterior (connected to self.depot)
            self.sol.routes.append(dndRoute) # add this  route to the solution
            self.sol.cost += dndRoute.cost
            self.sol.demand += dndRoute.demand
            
            if( self.enabledRouteCacheUsage ):
                self.bestSol = copy.deepcopy( self.sol );
                #initial bestRoutes
                routeIds = [0, dnEdge.end.ID]
                routeIds.sort();
                routeKey = str(0) + "_" + str(dnEdge.end.ID)
                self.bestRoutes[routeKey] = copy.deepcopy(dndRoute)
    
    def checkMergingConditions(self, iNode, jNode, iRoute, jRoute):
        # condition 1 : iRoute and jRoute  are not the  same route  object
        if iRoute == jRoute: return False
        # conditions 2: both nodes are exteriornodes in their respective routes
        if iNode.isInterior == True or jNode.isInterior == True: return False
        # condition 3: demand after merging can be covered by a single vehicle
        if self.vehCap < iRoute.demand + jRoute.demand: return False
        # else, merging is feasible
        return True
    
    def getDepotEdge(self, aRoute, aNode):
        ''' returns the edge in aRoute that contains aNode and the depot 
        (it will be the dirst or the last one) '''
        # check if first edge in aRoute contains aNode and self.depot
        origin = aRoute.edges[0].origin
        end = aRoute.edges[0].end
        if ((origin == aNode and end == self.depot) or
            (origin == self.depot and end == aNode)):
            return aRoute.edges[0]
        else: # retunr last edge in aRoute
            return aRoute.edges[-1]
    
    def edgeSelectionRoutingMerging(self, savingsList):
        """ Perform the edge-selection & routing-merging iterative process """
        while len(savingsList) > 0: # list is not empty
            ijEdge = savingsList.pop(0) # select the next edge from the list
            # determine the nodes i < j that define the edge
            iNode = ijEdge.origin
            jNode = ijEdge.end
            # determine the routes associated with each node
            iRoute = iNode.inRoute
            jRoute = jNode.inRoute
            # check if merge is possible
            isMergeFeasible = self.checkMergingConditions(iNode, jNode, iRoute, jRoute)
            # if all necessary coditions are satisfied, merge
            if isMergeFeasible == True:
                # iRoute will contain either edge (self.depot, 1) or edge (1, self.depot)
                iEdge = self.getDepotEdge(iRoute, iNode) # iEdge is either (0,1) or (1,0)
                # remove iEdge from iRoute and update iRoute cost
                iRoute.edges.remove(iEdge)
                iRoute.cost -= iEdge.cost
                # if there are multiple edges in iRoute, then i will be interior
                if len(iRoute.edges) > 1: iNode.isInterior = True
                # if new iRoute does not start at 0 it must be reversed
                if iRoute.edges[0].origin != self.depot: iRoute.reverse()
                # jRoute will contain either edge (self.depot, j) or  edge (j, self.depot)
                jEdge = self.getDepotEdge(jRoute, jNode) # jEdge is either (0, j) or (j,0)
                # remove jEdge from jRoute and update jRoute cost
                jRoute.edges.remove(jEdge)
                jRoute.cost -= jEdge.cost
                # if there are multiple edges in jRute, the j will be interior
                if len(jRoute.edges) > 1: jNode.isInterior = True
                # if  new jRoute  starts at 0 it must be reverse()
                if jRoute.edges[0].origin == self.depot: jRoute.reverse()
                # add ijEdge to iRoute
                iRoute.edges.append(ijEdge)
                iRoute.cost += ijEdge.cost
                iRoute.demand += jNode.demand
                jNode.inRoute = iRoute
                # add jRoute to new iRoute
                for edge in jRoute.edges:
                    iRoute.edges.append(edge)
                    iRoute.cost += edge.cost
                    iRoute.demand += edge.end.demand
                    edge.end.inRoute = iRoute
                # delete jRoute from emeging solution
                self.sol.cost -= ijEdge.savings
                self.sol.routes.remove(jRoute)
                
                #populate bestRoutes
                if( self.enabledRouteCacheUsage ):
                    #create hash to search for the route
                    routeKey = self.getRouteHash( iRoute )
                        
                    if routeKey in self.bestRoutes.keys():
                        if self.bestRoutes[routeKey].cost > iRoute.cost:
                            self.bestRoutes[routeKey] = copy.deepcopy( iRoute )
                    else:
                        self.bestRoutes[routeKey] =  copy.deepcopy( iRoute )
  
    def getCost(self):
        return self.sol.cost
    
    def printCost(self):
        print('Instance: '+ self.instanceName)
        print('Cost of C&W savings sol=', "{:{}f}".format(self.sol.cost, 2))
        
    def printRoute(self, route):
        s = str(0)
        for edge in route.edges:
            s = s + '-' + str(edge.end.ID)
        print('Route: ' + s + ' || cost = ' + "{:{}f}".format(route.cost,2))
    
    def printRouteCosts(self):
        print('Instance: '+ self.instanceName)
        for route in self.sol.routes:
            s = str(0)
            for edge in route.edges:
                s = s + '-' + str(edge.end.ID)
            print('Route: ' + s + ' || cost = ' + "{:{}f}".format(route.cost,2))
    
    def printBestRoutes(self):
        print( "-------------------------------------" )
        print( "Instance: " + self.instanceName )
        for route in sorted( self.bestRoutes.keys() ):
            s = str(0)
            for edge in self.bestRoutes[route].edges:
                s = s + '-' + str(edge.end.ID)
            print('Route ' + str(route) + ': ' + s + ' || cost = ' + "{:{}f}".format( self.bestRoutes[route].cost,2 ) )
        
        print( "-------------------------------------" )
    
    def printRouteCostsBestSolution(self, sol):
        print('Instance: '+ self.instanceName)
        for route in sol.routes:
            s = str(0)
            for edge in route.edges:
                s = s + '-' + str(edge.end.ID)
            print('Route: ' + s + ' || cost = ' + "{:{}f}".format(route.cost,2))
        print( "Solution cost = " + "{:{}f}".format(sol.cost,2))
            
    def getSolution(self):
        return self.sol
    
    def plotGraph(self):
        G = nx.Graph()
        fnode = self.sol.routes[0].edges[0].origin
        G.add_node(fnode.ID, coord=(fnode.x, fnode.y))
        coord = nx.get_node_attributes(G, 'coord')
        fig, ax = plt.subplots() #Add axes
        nx.draw_networkx(G, coord, node_size = 60, node_color='white', ax = ax)
        nx.draw_networkx(G, coord)
        
        plt.title('Instance: '+ self.instanceName)
        
        j=0
        for route in self.sol.routes:
            #Assign random colors in RGB
            c1 = int(random.uniform(0, 255)) if (j%3 == 2) else (j%3)*int(random.uniform(0, 255))
            c2 = int(random.uniform(0, 255)) if ((j+1)%3 == 2) else ((j+1)%3)*int(random.uniform(0, 255))
            c3 = int(random.uniform(0, 255)) if ((j+2)%3 == 2) else ((j+2)%3)*int(random.uniform(0, 255))
            for edge in route.edges:
                G.add_edge(edge.origin.ID, edge.end.ID)
                G.add_node(edge.end.ID, coord=(edge.end.x, edge.end.y))
                coord = nx.get_node_attributes(G, 'coord')
                nx.draw_networkx_nodes(G, coord, node_size = 60, node_color='white', ax=ax)
                nx.draw_networkx_edges(G, coord, edge_color='#%02x%02x%02x' % (c1,c2,c3))
                nx.draw_networkx_labels(G, coord, font_size = 9)
                G.remove_node(edge.origin.ID)
            j +=1
        limits=plt.axis('on') #Turn on axes)
        ax.tick_params(left=True, bottom =True, labelleft=True, labelbottom=True)
        plt.show()






