# coding=utf-8
import pickle
import random
import unittest
import math

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from submission import PriorityQueue, a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, uniform_cost_search, haversine_dist_heuristic, \
    tridirectional_upgraded, custom_heuristic
from visualize_graph import plot_search



class TestBasicSearch(unittest.TestCase):
    """Test the simple search algorithms: BFS, UCS, A*"""

    def setUp(self):
        """Romania map data from Russell and Norvig, Chapter 3."""
        with open('romania_graph.pickle', 'rb') as rom:
            romania = pickle.load(rom)
        print("***")
        print(type(romania))
        print("***")
        
        """
        create graph
        """
        """
        currency_map = [('STNR', 'ISBL', 0.99),
                        ('STNR', 'GBP', 0.0645),
                        ('STNR', 'EUR', 0.0465),
                        ('STNR', 'BTC', 0.0610),
                        ('BTC', 'AUD', 0.005),
                        ('EUR', 'AUD', 0.0650),
                        ('EUR', 'CNY', 0.0120),
                        ('GBP', 'AUD', 0.0493),
                        ('GBP', 'CNY', 0.0571),
                        ('AUD', 'TRY', 0.0621),
                        ('AUD', 'UGX', 0.0023),
                        ('AUD', 'INR', 0.0260),
                        ('CNY', 'TRY', 0.0170),
                        ('CNY', 'UGX', 0.0892),
                        ('CNY', 'INR', 0.0400),
                        ('INR', 'ISBL', 0.0847),
                        ('UGX', 'ISBL', 0.0100),
                        ('TRY', 'ISBL', 0.0124)
                        ]
        
        """
        
        land_map = [('St', 'J', 238),
                    ('St', 'E', 106),
                    ('E', 'Bo', 113),
                    ('Bo', 'N', 145),
                    ('Bo', 'L', 123),
                    ('N', 'Se', 115),
                    ('L', 'H', 123),
                    ('H', 'P', 134),
                    ('Se', 'T', 212), #211 on map
                    ('Se', 'P', 244),
                    ('I', 'P', 124),
                    ('I', 'T', 153),
                    ('J', 'Ba', 155), #140 on map
                    ('Ba', 'T', 168)]
        


        pos_ = {'I' : (220.2, 382.1), 
               'P' : (154.5, 350.4),
               'T' : (360.4, 323.2),
               'J' : (248.3 , 287.1),
               'N' : (148.7, 240.5),
               'H' : (87.5, 300.5),
               'L' : (63.2, 211.7),
               'Bo' : (111.3, 132.6),
               'E' : (182.9, 92.0),
               'Se' : (206.0, 178.6),
               'Ba': (365.3, 185.6),
               'St' : (246.4, 49.3)}
        G = networkx.Graph()
        
        """
        for i in range(len(currency_map)):
            G.add_weighted_edges_from([currency_map[i]]) 
        """
        for i in range(len(land_map)):
            G.add_weighted_edges_from([land_map[i]]) 
        
        networkx.set_node_attributes(G,  pos_, 'pos')
        
        #edge_labels = networkx.get_edge_attributes(G, 'weight')
        #weight = G.get_edge_data('Washington', 'Boise')['weight']
        
        #print("edge_labels = ", weight)
        
        #                start=start, goal=goal, path=path)
    
        
        self.romania = ExplorableGraph(G)
        #print("Graph values")
        
        #self.draw_graph(self.romania, node_positions=pos_)
        
        node_list = list(G.nodes(data = True))
        print("node_list = ", node_list[0][0])
        for i in range(len(node_list)):
            d = self.euclidean_dist(self.romania, node_list[i][0], 'St' )
            print("euclidean distnace [", node_list[i][0], "] = ", d)
        
        self.romania.reset_search()

    @staticmethod
    def euclidean_dist(graph, v, goal):
        x1,y1 = graph.nodes[v]['pos']
        x2,y2 = graph.nodes[goal]['pos']
    
        #print(x1, ", ", y1, ", ", x2, ", ", y2)
    
        distance = math.sqrt(math.pow((x2 - x1),2) + math.pow((y2 - y1),2))
   
        return int(distance)
    
    @staticmethod
    def get_path_cost(graph, path):
        cost = 0
        for i in range(0, len(path)):
            if(i+1 >= len(path)):
                break
            cost = cost + graph.get_edge_weight(path[i], path[i+1])
        return cost
        #Test for 2a and 2b
    """
    def test_ucs(self):
        #Test and visualize uniform-cost search
        start = 'ISBL'
        goal = 'STNR'

        #node_positions = {n: self.romania.nodes[n]['pos'] for n in
        #                  self.romania.nodes.keys()}
        
        
        self.romania.reset_search()
        path = uniform_cost_search(self.romania, start, goal)
        
        print("path = " , path)
        print("cost = ", get_path_cost(self.romania, path))
   

        #self.draw_graph(self.romania, node_positions=node_positions,
        #                start=start, goal=goal, path=path)
    
    """
    def test_astar_path(self):
        start = "I"
        goal = "St"
        
        path = uniform_cost_search(self.romania, start, goal)
        print("Path using ucs = ", path)
        
        print("*** starting a-star***")
        path = a_star(self.romania, start, goal)
        print("Path using a-star = ", path)
        print("Path cost =", self.get_path_cost(self.romania, path))
        
    
    """
    def test_best_path(self):
        start_list = ['Washington', 'Oregon', 'Stanford', 'UCLA']
        goal_list = ['Brown', 'MIT', 'Georgetown', 'Duke'] 
        
        for i in start_list:
            for j in goal_list:
                self.bi_ucs(i, j)
              
    def test_astar_path(self):
        start = "Gatech"
        goal = "OSU"
        
        path = uniform_cost_search(self.romania, start, goal)
        print("Path using ucs = ", path)
        
        path = a_star(self.romania, start, goal)
        print("Path using a-star = ", path)
        
    """           
    
    @staticmethod
    def draw_graph(graph, node_positions=None, start=None, goal=None,
                   path=None):
        """Visualize results of graph search"""
        explored = [key for key in graph.explored_nodes if graph.explored_nodes[key] > 0]

        labels = {}
        for node in graph:
            labels[node] = node

        if node_positions is None:
            node_positions = networkx.spring_layout(graph)

        networkx.draw_networkx_nodes(graph, node_positions)
        networkx.draw_networkx_edges(graph, node_positions, style='dashed')
        networkx.draw_networkx_labels(graph, node_positions, labels)

        networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
                                     node_color='g')
        edge_labels = networkx.get_edge_attributes(graph, 'weight')
        networkx.draw_networkx_edge_labels(graph, node_positions, edge_labels=edge_labels)
        
        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
            networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
                                         edge_color='b')

        if start:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[start], node_color='b')

        if goal:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[goal], node_color='y')

        plt.plot()
        plt.show()





if __name__ == '__main__':
     
    unittest.main()
