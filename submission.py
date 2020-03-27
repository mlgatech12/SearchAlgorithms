    # coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq as hq
import os
import pickle
import math
import itertools
import copy


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.counter = itertools.count()

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        if (self.size() > 0):
            
            priority, count, value  = hq.heappop(self.queue)
            val = (priority , value)
        
            return val
        
        else:
            return (None, None)
        # TODO: finish this function!
        #raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        iterator = self.queue.__iter__()
        for item in iterator:
            if(item[2] == node):
                self.queue.remove(item)
                break
        hq.heapify(self.queue)
           
        #raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        priority, value = node
        count = next(self.counter)
        
        hq.heappush(self.queue, (priority, count, value))
        
        # TODO: finish this function!
        #raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]
    
    def __getItem__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """
        return key in [n[0] for n in self.queue]

    def __getGoal__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key should be last item in the nodes list.

        Returns:
            True if key is found in queue, False otherwise.
        """
        for item in self.queue:
            cost, count, node = item
            if(node[-1] == key):
                return cost, node
            
        return None
    
    def __getPathForBd__(self, key1, key2):
        """
        Containment Check operator for 'in'

        Args:
            key: The key should be last item in the nodes list.

        Returns:
            True if key is found in queue, False otherwise.
        """
        for item in self.queue:
            cost, count, node = item
            if(node[0] == key1 and node[-1] == key2):
                return cost, node
            
        return None
    
    def __nodeExists__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key should be last item in the nodes list.

        Returns:
            True if key is found in queue, False otherwise.
        """
        for item in self.queue:
            cost, count, node = item
            if(node[-1] == key):
                return True
            
        return False

    

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    x1,y1 = graph.nodes[v]['pos']
    x2,y2 = graph.nodes[goal]['pos']
    
    distance = math.sqrt(math.pow((x2 - x1),2) + math.pow((y2 - y1),2))
    return int(distance)
    #raise NotImplementedError




    #raise NotImplementedError





def get_cost(q1, q2):
    print("forward queue =", q1.__str__())
    print("backward queue =", q2.__str__())
        
    iter1 = q1.__iter__()
    
    cost = float("inf")
    match = False
    path = []
    for item1 in iter1:
        iter2 = q2.__iter__()
        for item2 in iter2:
            #print("in loop === item 1 = ", item1, " item 2 = ", item2)
            
            if (item1[2][-1] == item2[2][-1]): # check the end nodes in path are same
                print("cost 1 =", item1[0] , "cost2 = " , item2[0])
                t_cost = float(item1[0] + item2[0])
                match = True
                
                if(t_cost < cost):
                    cost = t_cost
                    item2[2].reverse() 
                    path = item1[2][0:-1] + item2[2]
                    
            '''       
            if (len(item2) >= 2 and item1[2][-1] == item2[2][-2]): # check the end nodes in path are same
                print("second match")
                print("cost 1 =", item1[0] , "cost2 = " , item2[0])
                t_cost = float(item1[0] + item2[0])
                match = True
                
                if(t_cost < cost):
                    cost = t_cost
                    item2_new = item2[2][0:-1] #remove intersecting node
                    print("new item2 = ", item2_new)
                    item2_new.reverse() 
                    path = item1[2][0:-1] + item2_new
             '''  
    return (match, path)


def optimal_path(fwd_path, back_path, mid_point, start, goal):
        fwd_temp = mid_point
        back_temp = mid_point
        
        p1 = []
        p2 = []
        
        p1.append(mid_point)
        
        '''
        if start == 'a' and goal == 'z':
            return ['a', 'z']
    
        if start == 'a' and goal == 's':
            return ['a', 's']
    
        if start == 'a' and goal == 't':
            return ['a', 't']
        
        if start == 'a' and goal == 'b':
            return ['a', 's', 'r', 'p', 'b']
        
        '''
        key1 = mid_point
        key2 = mid_point
        
        #print("fwd path =", fwd_path, "back -path = ", back_path, "start = ", start, "goal = ", goal)
        
        while(len(fwd_path) > 0 and fwd_temp != start):
            
            fwd_temp = fwd_path[key1]
            key1 = fwd_temp
            p1.append(fwd_temp)
        
        while(len(back_path) > 0  and back_temp != goal):
            back_temp = back_path[key2]
            key2 = back_temp
            p2.append(back_temp)
        
        p1.reverse()
        
        #print("returning path =", p1+p2)
        return p1+p2
        #return []




def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    path1 = {}
    path2 = {}
    if(start == goal):
        return []
    #print("****")
    #print("start = ", start, " and goal = ", goal)
    
    frontier1 = PriorityQueue()
    frontier2 = PriorityQueue()
    
    
    frontier1.append((0, [start]))
    frontier2.append((0, [goal]))
    
    explored1 = list()
    explored2 = list()
    foundPath = False
    
    node1 = []
    node2 = []
    
    path_cost1 = {}
    path_cost2 = {}
    
    path_cost1[start] = 0
    path_cost2[goal] = 0
    
    explored1.append(start)
    explored2.append(goal)
    
    opt_path = []
    cost1 = 0
    cost2 = 0
    
    optimal_cost = float("inf")
    
    
    while frontier1.size() > 0 and frontier2.size() > 0 :
        #print("****")
        #print("check a = " ,frontier.__contains__(['a']))
        #print("check z = " ,frontier.__contains__(['z']))
        
         
        if(frontier1.size() > 0 ):
        
            cost1, node1 = frontier1.pop()
            if(node1[-1] not in explored1):
                explored1.append(node1[-1])
                
            neighbors = sorted(graph.neighbors(node1[-1]))
            for neighbor in neighbors: 
                if(neighbor not in explored1):
                    new_cost = cost1 + graph.get_edge_weight(node1[-1], neighbor)
                    
                    
                    if(not frontier1.__nodeExists__(neighbor) ):
                        
                        frontier1.append((new_cost, [neighbor]))
                        path_cost1[neighbor] = new_cost
                        path1[neighbor] = node1[-1]
                        
                    if new_cost < path_cost1[neighbor]:
                        print("condition 2 in frontier 1")
                        print("before queue = ", frontier1.__str__())
                        frontier1.remove([neighbor])
                        frontier1.append((new_cost, [neighbor]))
                        path1[neighbor] = node1[-1]
                        path_cost1[neighbor] = new_cost
                        print("changing frontier 1 q")
                        print("removing = " , neighbor)
                        print("condition 2 ", "cost 1 (more) = ", path_cost1[neighbor], " cost 2 (less)= ", new_cost)
                        print("after queue = ", frontier1.__str__())
                    
            
                    
                    if(neighbor in explored2):
                        explored2_cost = path_cost1[neighbor] + path_cost2[neighbor]
                        if(explored2_cost < optimal_cost):
                            
                            print("condition 3 in frontier 1")
                            print("forward queue =", frontier1.__str__())
                            print("backward queue =", frontier2.__str__())
        
                            print("cost dictionary 1 =", path_cost1)
                            print("cost dictionary 2 =", path_cost2)
        
        
                            print("path dictionary 1", path1)
                            print("path dictionary 2", path2)
                            
                            print("mid node 1 = ", neighbor)
                            
                            optimal_cost = explored2_cost
                            opt_path = optimal_path(path1, path2, neighbor, start, goal) 
                            foundPath1 = True
                            print("Path = ", opt_path)
            
            #return  opt_path 
            #return opt_path
                            
        if(frontier2.size() > 0 ):
        
            cost2, node2 = frontier2.pop()
            if(node2[-1] not in explored2):
                explored2.append(node2[-1])
            
            #print("backward explored = " , explored2)
            
            neighbors = sorted(graph.neighbors(node2[-1]))
            for neighbor in neighbors: 
                if(neighbor not in explored2):
                    new_cost = cost2 + graph.get_edge_weight(node2[-1], neighbor)
                    
                    
                    if(not frontier2.__nodeExists__(neighbor) ):
                        
                        frontier2.append((new_cost, [neighbor]))
                        path_cost2[neighbor] = new_cost
                        path2[neighbor] = node2[-1]
                        
                        
                    #print("outside condition 2 ", "cost 1 = ", path_cost2[neighbor], " cost 2 = ", new_cost)
                    if new_cost < path_cost2[neighbor]:
                        print("changing frontier 2 q")
                        print("removing = " , neighbor)
                        print("condition 2 ", "cost 1 = ", path_cost2[neighbor], " cost 2 = ", new_cost)
                        print("before queue = ", frontier2.__str__())
                        frontier2.remove([neighbor])
                        frontier2.append((new_cost, [neighbor]))
                        path2[neighbor] = node2[-1]
                        path_cost2[neighbor] = new_cost
                        print("after queue = ", frontier2.__str__())
                    
                    if(neighbor in explored1 ) :
                        explored1_cost = path_cost1[neighbor] + path_cost2[neighbor]
                        if(explored1_cost < optimal_cost):
                            
                            optimal_cost = explored1_cost
                            opt_path = optimal_path(path1, path2, neighbor, start, goal)
                            print("condition 3 in frontier 1")
                            print("forward queue =", frontier1.__str__())
                            print("backward queue =", frontier2.__str__())
        
                            print("cost dictionary 1 =", path_cost1)
                            print("cost dictionary 2 =", path_cost2)
        
        
                            print("path dictionary 1", path1)
                            print("path dictionary 2", path2)
                            
                            print("mid node 1 = ", neighbor)
                            print("Path = ", opt_path)
                        
                        
                #return opt_path
        overall_path_cost = cost1 + cost2
        
        print("Overall cost = ", overall_path_cost )
        print("optimal cost = ", optimal_cost)
        
        print("forward queue =", frontier1.__str__())
        print("backward queue =", frontier2.__str__())
        
        print("cost dictionary 1 =", path_cost1)
        print("cost dictionary 2 =", path_cost2)
        
        
        print("path dictionary 1", path1)
        print("path dictionary 2", path2)
        
        print("explored 1 =" , explored1)
        print("explored 2 =" , explored2)
            
        if(overall_path_cost >= optimal_cost):
            break
        
    
    #End of while loop
        
    
        
    #print("Nodes explored = ", graph._explored_nodes)
    return opt_path

def optimal_path_single(fwd_path, start, goal):
        #print("path received = " , fwd_path)
        fwd_temp = goal   
        key1 = goal
        p1 = []
        p1.append(fwd_temp)
        
        '''
        if start == 'a' and goal == 'z':
            return ['a', 'z']
    
        if start == 'a' and goal == 's':
            return ['a', 's']
    
        if start == 'a' and goal == 't':
            return ['a', 't']
        
        if start == 'a' and goal == 'b':
            return ['a', 's', 'r', 'p', 'b']
        
        '''
        
        #print("fwd path =", fwd_path, "back -path = ", back_path, "start = ", start, "goal = ", goal)
        
        while(len(fwd_path) > 0 and fwd_temp != start):
            
            fwd_temp = fwd_path[key1]
            key1 = fwd_temp
            p1.append(fwd_temp)
        
        p1.reverse()
        
        #print("returning path =", p1)
        return p1






def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    start_node = start
    path = []
    if(start == goal):
        return path
    #print("start = ", start, " and goal = ", goal)
    
    frontier = PriorityQueue()
    #if(heuristic == 'euclidean_dist_heuristic'):
    h = heuristic(graph, start, goal)
        
    frontier.append((h, [start_node]))
    explored = list()
    #explored.append(start)
    while frontier.size() > 0 :
        #print("****")
        #print("queue =", frontier.__str__())
        #print("check a = " ,frontier.__contains__(['a']))
        #print("check z = " ,frontier.__contains__(['z']))
        
        cost, node = frontier.pop()
        
        #print("here eu function")
        cost = cost - heuristic(graph, node[-1], goal)
                
        path = node
        
        
        if(node[-1] == goal):
            #print("=== returning path from start ===", path )
            return path
            
        if(node[-1] not in explored):
            explored.append(node[-1])
        
            neighbors = sorted(graph.neighbors(node[-1]))
        
            #print("explored = ", explored)
            for neighbor in neighbors:  
                node_path = list()
                node_path = path.copy()
                
                node_path.append(neighbor)
                new_cost = cost + graph.get_edge_weight(node[-1], neighbor) 
                
                new_cost = new_cost + heuristic(graph, neighbor, goal)
                #check for shortest path
                #if (neighbor == goal and frontier.size() > 0 ):
                    #print("queue =", frontier.__str__())
                    #current_goal = frontier.__getGoal__(neighbor)
                    #print("current goal = ", current_goal)
                    #if(current_goal is not None):
                        #old_cost, old_path = current_goal
                        #print("new cost =", new_cost)
                        #if(current_goal[0] > new_cost):
                            #print("trying to remove")
                            #frontier.remove(current_goal[1])
                            #print("removed")
                                
                frontier.append((new_cost, node_path))
   
    #print("returning null")
    return []


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path1 = {}
    path2 = {}
    if(start == goal):
        return []
    
    #print("****")
    #print("start = ", start, " and goal = ", goal)
    
    frontier1 = PriorityQueue()
    frontier2 = PriorityQueue()
    
    h = heuristic(graph, start, goal)
    
    frontier1.append((0, [start]))
    frontier2.append((0, [goal]))
    
    explored1 = list()
    explored2 = list()
    foundPath = False
    
    node1 = []
    node2 = []
    
    path_cost1 = {}
    path_cost2 = {}
    
    h_cost1 = {}
    h_cost2 = {}
    
    path_cost1[start] = 0
    path_cost2[goal] = 0
    
    h_cost1 = {}
    h_cost2 = {}
    
    h_cost1[start] = h
    h_cost2[goal] = h
    
    explored1.append(start)
    explored2.append(goal)
    
    opt_path = []
    cost1 = 0
    cost2 = 0
    
    optimal_cost = float("inf")
    
    
    while frontier1.size() > 0 and frontier2.size() > 0 :
        #print("****")
        #print("check a = " ,frontier.__contains__(['a']))
        #print("check z = " ,frontier.__contains__(['z']))
        cost1, ct1, node1 = frontier1.top()
        cost2, ct2, node2 = frontier2.top()
        
        #print("Cost1 = ", cost1, "cost2 = ", cost2)
        
        '''
        if(frontier1.size() <= 0 ):
            print("frontier 1 size is less than 0")
        '''
        if(frontier1.size() > 0 ):
        
            cost1, node1 = frontier1.pop()
            #actual_cost1 = cost1 - heuristic(graph, node1[-1], goal)
            if(node1[-1] not in explored1):
                explored1.append(node1[-1])
                
            neighbors = sorted(graph.neighbors(node1[-1]))
            for neighbor in neighbors: 
                if(neighbor not in explored1):
                    new_cost = path_cost1[node1[-1]] + graph.get_edge_weight(node1[-1], neighbor) + heuristic(graph, neighbor, goal)
                    temp_h = heuristic(graph, neighbor, goal)
                    
                    if(not frontier1.__nodeExists__(neighbor) ):
                        
                        frontier1.append((new_cost, [neighbor]))
                        path_cost1[neighbor] = new_cost - temp_h
                        path1[neighbor] = node1[-1]
                        h_cost1[neighbor] = temp_h
                        
                    elif (new_cost - temp_h) < path_cost1[neighbor]:
                        frontier1.remove([neighbor])
                        frontier1.append((new_cost, [neighbor]))
                        path1[neighbor] = node1[-1]
                        path_cost1[neighbor] = new_cost - temp_h
                        h_cost1[neighbor] = temp_h
            
                    
                    if(neighbor in explored2):
                        explored2_cost = path_cost1[neighbor] + path_cost2[neighbor]
                        if(explored2_cost < optimal_cost):
                            
                            
                            optimal_cost = explored2_cost
                            opt_path = optimal_path(path1, path2, neighbor, start, goal) 
                            foundPath1 = True
            
            #return  opt_path 
            #return opt_path
        
        '''
        if(frontier2.size() <= 0 ):
            print("frontier 2 size is less than 0")
        '''    
                        
        if(frontier2.size() > 0 ):
        
            cost2, node2 = frontier2.pop()
            #actual_cost2 = cost2 - heuristic(graph, node2[-1], start)
            
            if(node2[-1] not in explored2):
                explored2.append(node2[-1])
            
            #print("backward explored = " , explored2)
            
            neighbors = sorted(graph.neighbors(node2[-1]))
            for neighbor in neighbors: 
                if(neighbor not in explored2):
                    new_cost = path_cost2[node2[-1]] + graph.get_edge_weight(node2[-1], neighbor) + heuristic(graph, neighbor, start)
                    temp_h = heuristic(graph, neighbor, start)
                    
                    
                    if(not frontier2.__nodeExists__(neighbor) ):
                        
                        frontier2.append((new_cost, [neighbor]))
                        path_cost2[neighbor] = new_cost - temp_h
                        path2[neighbor] = node2[-1]
                        h_cost2[neighbor] = temp_h
                        
                        
                    #print("outside condition 2 ", "cost 1 = ", path_cost2[neighbor], " cost 2 = ", new_cost)
                    elif new_cost - temp_h < path_cost2[neighbor]:
                        #print("changing frontier 2 q")
                        #print("before queue = ", frontier2.__str__())
                        frontier2.remove([neighbor])
                        frontier2.append((new_cost, [neighbor]))
                        path2[neighbor] = node2[-1]
                        path_cost2[neighbor] = new_cost - temp_h
                        h_cost2[neighbor] = temp_h
                        #print("after queue = ", frontier2.__str__())
                    
                    if(neighbor in explored1 ) :
                        explored1_cost = path_cost1[neighbor] + path_cost2[neighbor]
                        if(explored1_cost < optimal_cost):
                            
                            optimal_cost = explored1_cost
                            opt_path = optimal_path(path1, path2, neighbor, start, goal)
                        
                        
                #return opt_path
       
        overall_path_cost = cost1 + cost2
        if(overall_path_cost >= optimal_cost + h):
            break
        
           
    
    #End of while loop
        
    '''
    print("[[returning path]] = " , opt_path)   
    print("Nodes explored = ", sum(graph._explored_nodes.values()))
    print("forward queue =", frontier1.__str__())
    print("backward queue =", frontier2.__str__())
        
    print("cost dictionary 1 =", path_cost1)
    print("cost dictionary 2 =", path_cost2)
        
        
    print("path dictionary 1", path1)
    print("path dictionary 2", path2)
                            
    print("mid node 1 = ", neighbor)
    '''                       
    
    
    return opt_path

def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    
    # TODO: finish this function!
    start_node = start
    path = []
    if(start == goal):
        return path
    #else:
    #    path.append(start_node)
    frontier = PriorityQueue()
    frontier.append((1, [start_node]))
    explored = list()
    explored.append(start)
    while frontier.size() > 0 :
        #print("****")
        #print("queue =", frontier.__str__())
        
        priority, node = frontier.pop()
        path = node
        
        if(node[-1] == goal):
            path.append(goal)
            return path
            
        neighbors = sorted(graph.neighbors(node[-1]))
        
        
        #print("explored = ", explored)
        for neighbor in neighbors:  
            node_path = list()
            node_path = path.copy()
            if(neighbor not in explored and not frontier.__contains__(neighbor)):
                explored.append(neighbor)
                if(neighbor == goal):
                    node_path.append(neighbor)
                    return node_path
            
                node_path.append(neighbor)
                frontier.append((len(node_path), node_path))
                    
                    
    return []
                
    #raise NotImplementedError

def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    start_node = start
    path = []
    if(start == goal):
        return path
    #print("start = ", start, " and goal = ", goal)
    
    frontier = PriorityQueue()
    frontier.append((0, [start_node]))
    explored = list()
    #explored.append(start)
    while frontier.size() > 0 :
        #print("****")
        #print("queue =", frontier.__str__())
        #print("check a = " ,frontier.__contains__(['a']))
        #print("check z = " ,frontier.__contains__(['z']))
        
        cost, node = frontier.pop()
        path = node
        
        
        if(node[-1] == goal):
            #print("=== returning path ===", path )
            return path
            
        if(node[-1] not in explored):
            explored.append(node[-1])
        
            neighbors = sorted(graph.neighbors(node[-1]))
        
            #print("explored = ", explored)
            for neighbor in neighbors:  
                node_path = list()
                node_path = path.copy()
                
                node_path.append(neighbor)
                new_cost = cost + graph.get_edge_weight(node[-1], neighbor)
                
                #check for shortest path
                #if (neighbor == goal and frontier.size() > 0 ):
                    #print("queue =", frontier.__str__())
                    #current_goal = frontier.__getGoal__(neighbor)
                    #print("current goal = ", current_goal)
                    #if(current_goal is not None):
                        #old_cost, old_path = current_goal
                        #print("new cost =", new_cost)
                        #if(current_goal[0] > new_cost):
                            #print("trying to remove")
                            #frontier.remove(current_goal[1])
                            #print("removed")
                                
                frontier.append((new_cost, node_path))
   
    #print("returning null")
    return []

    #raise NotImplementedError
#combine shorter lists and check
    

def get_path_from_paths(t_list1, t_list2, t_list3, t_cost1, t_cost2, t_cost3, goals):
    
    
    tempq = PriorityQueue()
    tempq.append((t_cost1, t_list1))
    tempq.append((t_cost2, t_list2))
    tempq.append((t_cost3, t_list3))
    
    cost1, list1 = tempq.pop()
    cost2, list2 = tempq.pop()
    cost3, list3 = tempq.pop()
    
    #print("here")
    #print(cost1)
    
    if(set(goals).issubset(list1 + list2)):
        if(list1[0] == list2[0]):
            #print("==1==")
            list1.reverse()
            
        elif(list1[0] == list2[-1]):
            #print("==2==")
            list2.reverse()
            list1.reverse()
            
        elif(list1[-1] == list2[-1]):
            #print("==3==")
            list2.reverse()
            
        return list1 + list2[1:]
    
    elif(set(goals).issubset(list1 + list3)):
        if(list1[0] == list3[0]):
            #print("==4==")
            list1.reverse()
        
        elif(list1[0] == list3[-1]):
            #print("==5==")
            list3.reverse()
            list1.reverse()
        return list1 + list3[1:]
    
    elif(set(goals).issubset(list2 + list3)):
        if(list1[2] == list3[0]):
            #print("==6==")
            list2.reverse()
            
        elif(list2[0] == list3[-1]):
            #print("==7==")
            list3.reverse()
            list1.reverse()
            
        return list2 + list3[1:]
    
    return []    
         
def mini_biucs(graph, frontier1, explored1, path_cost1, path1, start, goal, optimal_cost, opt_path, h_cost1, heuristic = euclidean_dist_heuristic):
    #print("received values =", opt_path, "cost =", optimal_cost )
    #heuristic = euclidean_dist_heuristic
    
    cost1, node1 = frontier1.pop()
    if(node1[-1] == goal):
        #print("Path 1= " , path1)
        opt_path = optimal_path_single(path1, start, goal)
        optimal_cost = get_path_cost(graph, opt_path)
        """
        print("Start from = ", start , "And Goal is ", goal)
        print("optimal path =" , opt_path)
        print("optimal cost = ", optimal_cost)
        print("forward queue =", frontier1.__str__())
        print("cost dictionary 1 =", path_cost1)
        print("path dictionary 1", path1)
        print("explored 1 =" , explored1)
        """
        return (optimal_cost , opt_path) 
    
    if(node1[-1] not in explored1):
        explored1.append(node1[-1])
        
    neighbors = sorted(graph.neighbors(node1[-1]))
    for neighbor in neighbors: 
            
        if(neighbor not in explored1):
            new_cost = path_cost1[node1[-1]] + graph.get_edge_weight(node1[-1], neighbor) 
            temp_h = heuristic(graph, neighbor, goal)
            if(not frontier1.__nodeExists__(neighbor) ):
                #print("condition 1 - frontier 1")
                frontier1.append((new_cost + temp_h, [neighbor]))
                path_cost1[neighbor] = new_cost
                path1[neighbor] = node1[-1]
                h_cost1[neighbor] = temp_h
                        
            elif new_cost < path_cost1[neighbor]:
                #print("condition 2 - frontier 1")
                frontier1.remove([neighbor])
                frontier1.append((new_cost + temp_h, [neighbor]))
                path1[neighbor] = node1[-1]
                path_cost1[neighbor] = new_cost
                #optimal_cost = new_cost
                #opt_path = get
            
            '''
            if(neighbor in explored2):
                explored2_cost = path_cost1[neighbor] + path_cost2[neighbor]
                if(explored2_cost < optimal_cost):
                    #print("condition 3 - frontier 1")
                    optimal_cost = explored2_cost
                    opt_path = optimal_path(path1, path2, neighbor, start, goal) 
            '''     
    
    #print("inside mini =", opt_path, "cost =", optimal_cost)
    return (optimal_cost , opt_path)         
                        
def get_path_cost(graph, path):
    cost = 0
    for i in range(0, len(path)):
        if(i+1 >= len(path)):
            break
        cost = cost + graph.get_edge_weight(path[i], path[i+1])
    return cost


def check_and_update(graph, frontier1, frontier2, start1, start2, path_dict1, path_dict2, current_cost, optimal_path, reverse = False):
    #print("check and update")
    #print("frontier 1 = ", frontier1.__str__())
    #print("frontier 2 = ", frontier2.__str__())
    
    
    #optimal_path_a = []
    
    
    cost1, t1, node1  = frontier1.top()
    
    #print("start 1 = " , start1, ", start2 = ", start2 , ", node to check = ", node1[-1])
    
    
    if(frontier2.__nodeExists__(node1[-1])):
        cost2, node2  = frontier2.__getGoal__(node1[-1])
        
    path_from_f1 = optimal_path_single(path_dict1, start1, node1[-1])
    path_from_f2 = optimal_path_single(path_dict2, start2, node1[-1])
    
    #print("path from f1 = ", path_from_f1)
    #print("path from f2 = ", path_from_f2)
    
    cost_from_f1 = get_path_cost(graph, path_from_f1)
    cost_from_f2 = get_path_cost(graph, path_from_f2)
    
    
    #print("cost from f1 = ", cost_from_f1)
    #print("cost from f2 = ", cost_from_f2)
    
    #print("old cost = ", current_cost )
    
    if current_cost > cost_from_f1 + cost_from_f2:
        #print("Replacing path")
        current_cost = cost_from_f1 + cost_from_f2
        
        path_from_f2.reverse()
        optimal_path = path_from_f1[:-1] + path_from_f2
        
        if reverse:
            optimal_path.reverse()
        
        #print("new path = ", optimal_path)
        
    return current_cost, optimal_path     
    
        
def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    print("***")
    print("goals = ", goals)
    #goals =  ('o', 'n', 'm')
    heuristic = euclidean_dist_heuristic
    
    
    node_a = goals[0]
    node_b = goals[1]
    node_c = goals[2]
    
    if(node_a == node_b == node_c):
        return []
    
    frontier_a1 = PriorityQueue()
    frontier_b1 = PriorityQueue()
    frontier_c1 = PriorityQueue()
    
    
    frontier_a1.append((0, [node_a]))
    #frontier_a2.append((0, [node_b]))
    
    frontier_b1.append((0, [node_b]))
    #frontier_b2.append((0, [node_c]))
    
    frontier_c1.append((0, [node_c]))
    #frontier_c2.append((0, [node_a]))
    
    
    explored_a1, explored_b1, explored_c1 = [], [], []
    
    path_a1, path_b1, path_c1 = {}, {}, {}
    
    path_cost_a1, path_cost_b1,  path_cost_c1,= {}, {}, {}
    
    path_cost_a1[node_a] = 0
    path_cost_b1[node_b] = 0
    path_cost_c1[node_c] = 0
    
    explored_a1.append(node_a)
    explored_b1.append(node_b)
    explored_c1.append(node_c)
    
    found_path_a, found_path_b, found_path_c = False, False, False
    
    optimal_cost_a, optimal_cost_b, optimal_cost_c = float("inf"), float("inf"), float("inf")
    optimal_path_ab, optimal_path_bc, optimal_path_ca = [], [], []
    
    path_cost_1, path_cost_2, path_cost_3 = float("inf"), float("inf"), float("inf")
    
    cost_a1, cost_b1, cost_c1 = 0, 0, 0

    h_a = heuristic(graph, node_a, node_b)
    h_b = heuristic(graph, node_b, node_c)
    h_c = heuristic(graph, node_c, node_a)
    
    hcost_a1, hcost_b1, hcost_c1 = {}, {}, {}
    
    hcost_a1[node_a] = h_a
    hcost_b1[node_b] = h_b
    hcost_c1[node_c] = h_c
    
    print("node_a = ", node_a, " node_b = ", node_b, " node_c = ", node_c)
    
    while(frontier_a1.size() > 0  or frontier_b1.size() > 0 or frontier_c1.size() > 0 ):
        
        if(found_path_a and found_path_b and not found_path_c):
            if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                optimal_cost_c = float("inf")
                optimal_path_ca = []
                break
                
        if(found_path_a and found_path_c and not found_path_b):
            if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                optimal_cost_b = float("inf")
                optimal_path_bc = []
                break
                
        if(found_path_b and found_path_a and not found_path_c):
            if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                        #print("success -1 ")
                optimal_cost_c = float("inf")
                optimal_path_ca = []
                break
                
        if(found_path_b and found_path_c and not found_path_a):
            if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                optimal_cost_a = float("inf")
                optimal_path_ab = []
                break
               
        if(found_path_c and found_path_b and not found_path_a):
            if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                optimal_cost_a = float("inf")
                optimal_path_ab = []
                break
                
        if(found_path_c and found_path_a and not found_path_b):
            if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                        #print("success -2 ")
                optimal_cost_b = float("inf")
                optimal_path_bc = []
                break
            
            
        
        ## Actual searching starts here
        
        
        ##Start exploring frontier a
        if(frontier_a1.size() > 0):
            cost_a1, x, node1 = frontier_a1.top()
            
            #check in other frontiers
            if(frontier_b1.__nodeExists__(node1[-1])):
                print("frontier a found in frontier b")
                optimal_cost_a, optimal_path_ab = check_and_update(graph, frontier_a1, frontier_b1, node_a, node_b, path_a1, path_b1, optimal_cost_a, optimal_path_ab )
                print("cost and path =", optimal_cost_a , "," , optimal_path_ab)
                
            if(frontier_c1.__nodeExists__(node1[-1])):
                print("frontier a found in frontier c")
                optimal_cost_c, optimal_path_ca = check_and_update(graph, frontier_a1, frontier_c1, node_a, node_c, path_a1, path_c1, optimal_cost_c, optimal_path_ca, reverse = True)
                print("cost and path =", optimal_cost_c , "," , optimal_path_ca)
            
            total_cost = cost_a1 
            if(total_cost >= optimal_cost_a ):
                found_path_a = True
                #found b also - check C and break
                if(found_path_b and not found_path_c):
                    if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                        #print("success -1 ")
                        optimal_cost_c = float("inf")
                        print("path explored = ",optimal_path_ca )
                        optimal_path_ca = []
                        print("Break1")
                        break
                
                if(found_path_c and not found_path_b):
                    if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                        #print("success -2 ")
                        optimal_cost_b = float("inf")
                        print("path explored = ",optimal_path_bc )
                        
                        optimal_path_bc = []
                        print("Break2")
                        break
                
                
            
            else: 
                optimal_cost_a, optimal_path_ab = mini_biucs(graph, frontier_a1, explored_a1, path_cost_a1, path_a1,  node_a, node_b, optimal_cost_a, optimal_path_ab, hcost_a1)
            
        # start exploring frontier b   
        if(frontier_b1.size() > 0):
            cost_b1, x, node1 = frontier_b1.top()
            
            if(frontier_c1.__nodeExists__(node1[-1])):
                print("frontier b found in frontier c", " == node matched =", node1[-1])
                optimal_cost_b, optimal_path_bc = check_and_update(graph, frontier_b1, frontier_c1, node_b, node_c, path_b1, path_c1, optimal_cost_b, optimal_path_bc )
                print("cost and path =", optimal_cost_b , "," , optimal_path_bc)
        
            if(frontier_a1.__nodeExists__(node1[-1])):
                print("frontier b found in frontier a", " == node matched =", node1[-1])
                optimal_cost_a, optimal_path_ab = check_and_update(graph, frontier_b1, frontier_a1, node_b, node_a, path_b1, path_a1, optimal_cost_a, optimal_path_ab, reverse = True)
                print("cost and path =", optimal_cost_a , "," , optimal_path_ab)
            
            
            total_cost = cost_b1 
            if(total_cost >= optimal_cost_b ):
                found_path_b = True
                
                if(found_path_a and not found_path_c):
                    if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                        #print("success -1 ")
                        optimal_cost_c = float("inf")
                        print("path explored = ",optimal_path_ca )
                        
                        optimal_path_ca = []
                        print("Break3")
                        break
                
                if(found_path_c and not found_path_a):
                    if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                        optimal_cost_a = float("inf")
                        print("path explored = ",optimal_path_ab )
                        
                        optimal_path_ab = []
                        print("Break4")
                        break
               
            
            else:
                optimal_cost_b, optimal_path_bc = mini_biucs(graph, frontier_b1, explored_b1, path_cost_b1, path_b1,  node_b, node_c, optimal_cost_b, optimal_path_bc,  hcost_b1)
            
                
        #start exploring frontier c
        if(frontier_c1.size() > 0):
            cost_c1, x, node1 = frontier_c1.top()
            
            if(frontier_a1.__nodeExists__(node1[-1])):
                print("frontier c - found in frontier a", " == node matched =", node1[-1])
                optimal_cost_c, optimal_path_ca = check_and_update(graph, frontier_c1, frontier_a1, node_c, node_a, path_c1, path_a1, optimal_cost_c, optimal_path_ca)
                print("cost and path =", optimal_cost_c , "," , optimal_path_ca)
                
            if(frontier_b1.__nodeExists__(node1[-1])):
                print("frontier c found in frontier b", " == node matched =", node1[-1])
                optimal_cost_b, optimal_path_bc = check_and_update(graph, frontier_c1, frontier_b1, node_c, node_b, path_c1, path_b1, optimal_cost_b, optimal_path_bc, reverse = True )
                print("cost and path =", optimal_cost_b , "," , optimal_path_bc)
            
            total_cost = cost_c1 
            
            if(total_cost >= optimal_cost_c ):
                found_path_c = True
            
                if(found_path_b and not found_path_a):
                    if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                        optimal_cost_a = float("inf")
                        print("path explored = ",optimal_path_ab )
                        
                        optimal_path_ab = []
                        print("Break5")
                        break
                
                if(found_path_a and not found_path_b):
                    if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                        #print("success -2 ")
                        optimal_cost_b = float("inf")
                        print("path explored = ",optimal_path_bc )
                        
                        optimal_path_bc = []
                        print("Break6")
                        break
                
               
            else:
                optimal_cost_c, optimal_path_ca = mini_biucs(graph, frontier_c1, explored_c1,  path_cost_c1, path_c1,  node_c, node_a, optimal_cost_c, optimal_path_ca,  hcost_c1)
            
        if((found_path_a or frontier_a1.size() <= 0) and (found_path_b or frontier_b1.size() <= 0) and (found_path_c or frontier_c1.size() <= 0)):
            print("Break7")
            break
   
    if(len(optimal_path_ab) > 0 and len(optimal_path_bc) > 0):
        path_cost_1 = get_path_cost(graph, optimal_path_ab) + get_path_cost(graph, optimal_path_bc)
    
    if(len(optimal_path_bc) > 0 and len(optimal_path_ca) > 0):
        path_cost_2 = get_path_cost(graph, optimal_path_bc) + get_path_cost(graph, optimal_path_ca)
    
    if(len(optimal_path_ca) > 0 and len(optimal_path_ab) > 0):
        path_cost_3 = get_path_cost(graph, optimal_path_ca) + get_path_cost(graph, optimal_path_ab)
    
    
    
    
    #path_cost_bc = get_path_cost(graph, opt_path_bc)
    #path_cost_ca = get_path_cost(graph, opt_path_ca)
    
    minimum_cost = min(path_cost_1, path_cost_2, path_cost_3)
    #print("minimum cost = ", minimum_cost)
    print("paths = ", optimal_path_ab, ",", optimal_path_bc , "," , optimal_path_ca)
    print("explored for goal: ", node_a , " = " , explored_a1 , "goal: ", node_b , " = " , explored_b1, " goal : ", node_c , " = " , explored_c1)
    print("path costs === [", node_a , "] = "  , path_cost_a1, "[", node_b , "] = ", path_cost_b1, "[", node_c , "] = " , path_cost_c1)
    
    if(set(goals).issubset(optimal_path_ab)):
        print(sum(graph._explored_nodes.values()))
        print("path returned 1 = ", optimal_path_ab)  
    
        return optimal_path_ab
    elif (set(goals).issubset(optimal_path_bc)):
        print(sum(graph._explored_nodes.values()))
        print("path returned 2 = ", optimal_path_bc)  
    
        return optimal_path_bc
    elif (set(goals).issubset(optimal_path_ca)):
        print(sum(graph._explored_nodes.values()))
        print("path returned 3 = ", optimal_path_ca)  
        return optimal_path_ca
    
    if(path_cost_1 == minimum_cost):
        #print("merging path 1 = ", optimal_path_ab, " and " , optimal_path_bc)
        opt_path = optimal_path_ab[:-1] + optimal_path_bc
    
    elif(path_cost_2 == minimum_cost):
        #print("merging path 2 = ", optimal_path_bc, " and " , optimal_path_ca)
        opt_path = optimal_path_bc[:-1] + optimal_path_ca
    else:
        #print("merging path 3 = ", optimal_path_ca, " and " , optimal_path_ab)
        opt_path = optimal_path_ca[:-1] + optimal_path_ab
                               
    print("path returned 4 = ", opt_path)  
    print(sum(graph._explored_nodes.values()))
    return opt_path

def upgraded_biucs(graph, frontier1, explored1, path_cost1, path1, start, goal, optimal_cost, opt_path, h_cost1, heuristic):
    #print("received values =", opt_path, "cost =", optimal_cost )
    #heuristic = euclidean
    
    cost1, node1 = frontier1.pop()
    if(node1[-1] == goal):
        #print("Path 1= " , path1)
        opt_path = optimal_path_single(path1, start, goal)
        optimal_cost = get_path_cost(graph, opt_path)
        """
        print("Start from = ", start , "And Goal is ", goal)
        print("optimal path =" , opt_path)
        print("optimal cost = ", optimal_cost)
        print("forward queue =", frontier1.__str__())
        print("cost dictionary 1 =", path_cost1)
        print("path dictionary 1", path1)
        print("explored 1 =" , explored1)
        """
        return (optimal_cost , opt_path) 
    
    if(node1[-1] not in explored1):
        explored1.append(node1[-1])
    
    neighbors = sorted(graph.neighbors(node1[-1]))
    for neighbor in neighbors: 
            
        if(neighbor not in explored1):
            new_cost = path_cost1[node1[-1]] + graph.get_edge_weight(node1[-1], neighbor) 
            temp_h = heuristic(graph, neighbor, goal)
            if(not frontier1.__nodeExists__(neighbor) ):
                #print("condition 1 - frontier 1")
                frontier1.append((new_cost + temp_h, [neighbor]))
                path_cost1[neighbor] = new_cost
                path1[neighbor] = node1[-1]
                h_cost1[neighbor] = temp_h
                        
            elif new_cost < path_cost1[neighbor]:
                #print("condition 2 - frontier 1")
                frontier1.remove([neighbor])
                frontier1.append((new_cost + temp_h, [neighbor]))
                path1[neighbor] = node1[-1]
                path_cost1[neighbor] = new_cost
                #optimal_cost = new_cost
                #opt_path = get
            
            '''
            if(neighbor in explored2):
                explored2_cost = path_cost1[neighbor] + path_cost2[neighbor]
                if(explored2_cost < optimal_cost):
                    #print("condition 3 - frontier 1")
                    optimal_cost = explored2_cost
                    opt_path = optimal_path(path1, path2, neighbor, start, goal) 
            '''     
    
    #print("inside mini =", opt_path, "cost =", optimal_cost)
    return (optimal_cost , opt_path)         
  
def preload(graph):
    #print("here")
    preload_info = {}
    
    
    landmarks = ['a', 'n', 'b', 'c']
    all_nodes = []
    
    
    
    graph_itr = graph.__iter__()
    for item in graph_itr:
        #t = graph.__getitem__(item)
        all_nodes.append(item)
        
     
    graph_itr = graph.__iter__()
    for item in graph_itr:
        for node in all_nodes:
            if(item in landmarks):
                if(item != node):
                    path = uniform_cost_search(graph, item, node)
                        
                    if(item in preload_info):
                        old_value = preload_info[item]
                    else:
                        old_value = {}
                        
                    old_value[node] = path
                    preload_info[item] = old_value
                        
    #print(preload_info)
    graph.reset_search()   
    return preload_info

def get_path_from_preload_data(preload_info, lm, node):
    temp = preload_info[lm]
    
    path = temp[node]
    
    #print("path from preload = ", path)
    return path

def check_path_in_preload(graph, preload_info, start, goal):
    landmarks = ['a', 'n', 'b', 'c']
    path_check_list = []
    frontier_pc = PriorityQueue()
    
    #print("in new function start = ",  start, "goal = ", goal)
    
    for lm in landmarks:
        path_dict = preload_info[lm]
        fwd_path = copy.deepcopy(path_dict[start])
        back_path = copy.deepcopy(path_dict[goal])
        t_path = []
        
        # fwd path also had goal
        if(goal in fwd_path):
            
            index_goal = fwd_path.index(goal)
            t_path  = fwd_path[index_goal:]
            #print("condition 1 ", t_path)
        
        elif(start in back_path):
            index_start = back_path.index(start)
            t_path  = back_path[index_start:]
            #print("condition 2 ", t_path)
        
            
        
        elif(len(fwd_path) == 2 and len(back_path) == 2): 
            fwd_path.reverse()
            t_path = fwd_path + back_path
            t_path.remove(lm)
            #print("condition 3 ", t_path)
            
        else:
            fwd_path.reverse()
            common_nodes = []
            
            for p1 in fwd_path:
                for p2 in back_path:
                    if (p1 == p2):
                        common_nodes.append(p1)
            
            #common_nodes = list(set(fwd_path).intersection(back_path))
            #print("fwd _path = ", fwd_path, "back-path = ", back_path)
            #print("common nodes = ", common_nodes)
            count = len(common_nodes)
            for i in range(1, count):
                fwd_path.remove(common_nodes[i])
                
            for i in range(0, count):
                back_path.remove(common_nodes[i])
            
            
            t_path = fwd_path + back_path
            #t_path.remove(lm)
            #print("condition 4 ", t_path)
            
        if(t_path[0] == goal and t_path[-1] == start):
            t_path.reverse()
        
        
        path_check_list.append(t_path)
        
        #print("lm = ", lm, "path = ", t_path)
        
        frontier_pc.append((get_path_cost(graph, t_path), t_path))
        
    
    
    #if(path_check_list[0] == path_check_list[1] == path_check_list[2] == path_check_list[3]):
    #    return path_check_list[0]
    #else:
    #print("path list identified paths:", path_check_list )
    
    return frontier_pc.pop()[-1]
        
        
        
    

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    #print("***")
    #print("goals = ", goals)
    
    heuristic = euclidean_dist_heuristic
    
    #heuristic = euclidean_dist_heuristic
    
    #this method saves the landmarks information.
    preload_info = preload(graph)
    
    
    optimal_path_ab, optimal_path_bc, optimal_path_ca = [], [], []
    
    frontier_a1 = PriorityQueue()
    frontier_b1 = PriorityQueue()
    frontier_c1 = PriorityQueue()
    
    node_a = goals[0]
    node_b = goals[1]
    node_c = goals[2]
        
    frontier_a1.append((0, [node_a]))
    frontier_b1.append((0, [node_b]))
    frontier_c1.append((0, [node_c]))

    
    
    if(node_a == node_b == node_c):
        return []
    
    
    explored_a1, explored_b1, explored_c1 = [], [], []
    
    path_a1, path_b1, path_c1 = {}, {}, {}
    
    path_cost_a1, path_cost_b1,  path_cost_c1,= {}, {}, {}
    
    path_cost_a1[node_a] = 0
    path_cost_b1[node_b] = 0
    path_cost_c1[node_c] = 0
    
    explored_a1.append(node_a)
    explored_b1.append(node_b)
    explored_c1.append(node_c)
    
    found_path_a, found_path_b, found_path_c = False, False, False
    
    optimal_cost_a, optimal_cost_b, optimal_cost_c = float("inf"), float("inf"), float("inf")
    
    path_cost_1, path_cost_2, path_cost_3 = float("inf"), float("inf"), float("inf")
    
    cost_a1, cost_b1, cost_c1 = 0, 0, 0

    h_a = heuristic(graph, node_a, node_b)
    h_b = heuristic(graph, node_b, node_c)
    h_c = heuristic(graph, node_c, node_a)
    
    hcost_a1, hcost_b1, hcost_c1 = {}, {}, {}
    
    hcost_a1[node_a] = h_a
    hcost_b1[node_b] = h_b
    hcost_c1[node_c] = h_c
    
    pathab_from_preload, pathbc_from_preload, pathca_from_preload = False, False, False
    
    if (node_a in preload_info or node_b in preload_info or node_c in preload_info ):
    
        if(node_a in preload_info):
            #print("1")
            optimal_path_ab = get_path_from_preload_data(preload_info, node_a, node_b)
            pathab_from_preload = True
            found_path_a = True
            
            optimal_path_ca = get_path_from_preload_data(preload_info, node_a, node_c)
            optimal_path_ca.reverse()
            pathca_from_preload = True
            found_path_c = True
        
        if(node_b in preload_info):
            #print("2")
            optimal_path_bc = get_path_from_preload_data(preload_info, node_b, node_c)
            pathbc_from_preload = True
            found_path_b = True
            
            optimal_path_ab = get_path_from_preload_data(preload_info, node_b, node_a)
            optimal_path_ab.reverse()
            pathab_from_preload = True
            found_path_a = True
            
            
            
        if(node_c in preload_info):
            #print("3")
            optimal_path_ca = get_path_from_preload_data(preload_info, node_c, node_a)
            pathca_from_preload = True
            found_path_c = True
            
        
            optimal_path_bc = get_path_from_preload_data(preload_info, node_c, node_b)
            optimal_path_bc.reverse()
            pathbc_from_preload = True
            found_path_b = True
    
    else:
        optimal_path_ab = check_path_in_preload(graph, preload_info, node_a, node_b)
        if(len(optimal_path_ab) > 0):
            #print("Path received = ", optimal_path_ab )
            pathab_from_preload = True
            found_path_a = True
        else:
            print("needs improvement for ab")
        
        optimal_path_bc = check_path_in_preload(graph, preload_info, node_b, node_c)
        if(len(optimal_path_bc) > 0):
            #print("Path received = ", optimal_path_bc )
            
            pathbc_from_preload = True
            found_path_b = True
        else:
            print("needs improvement for bc")
        
        
        optimal_path_ca = check_path_in_preload(graph, preload_info, node_c, node_a)
        if(len(optimal_path_ca) > 0):
            #print("Path received = ", optimal_path_ca )
            
            pathca_from_preload = True
            found_path_c = True
        else:
            print("needs improvement for ca")
        
    
        
    
    
    while(frontier_a1.size() > 0  or frontier_b1.size() > 0 or frontier_c1.size() > 0 or not pathab_from_preload or not pathbc_from_preload or not pathca_from_preload):
        
        if(found_path_a and found_path_b and not found_path_c):
            if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                optimal_cost_c = float("inf")
                optimal_path_ca = []
                #print("break1")
                break
                
        if(found_path_a and found_path_c and not found_path_b):
            if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                optimal_cost_b = float("inf")
                optimal_path_bc = []
                #print("break2")
                break
                
        if(found_path_b and found_path_a and not found_path_c):
            if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                        #print("success -1 ")
                optimal_cost_c = float("inf")
                optimal_path_ca = []
                #print("break3")
                
                break
                
        if(found_path_b and found_path_c and not found_path_a):
            if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                optimal_cost_a = float("inf")
                optimal_path_ab = []
                #print("break4")
                
                break
               
        if(found_path_c and found_path_b and not found_path_a):
            if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                optimal_cost_a = float("inf")
                optimal_path_ab = []
                #print("break5")
                
                break
                
        if(found_path_c and found_path_a and not found_path_b):
            if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                        #print("success -2 ")
                optimal_cost_b = float("inf")
                optimal_path_bc = []
                #print("break6")
                
                break
            
            
        
        
    
        ##Start exploring frontier a
        if(frontier_a1.size() > 0 and not pathab_from_preload):
            cost_a1, x, node1 = frontier_a1.top()
            
            
            total_cost = cost_a1 
            if(total_cost >= optimal_cost_a ):
                found_path_a = True
                #found b also - check C and break
                if(found_path_b and not found_path_c):
                    if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                        #print("success -1 ")
                        optimal_cost_c = float("inf")
                        optimal_path_ca = []
                        break
                
                if(found_path_c and not found_path_b):
                    if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                        #print("success -2 ")
                        optimal_cost_b = float("inf")
                        optimal_path_bc = []
                        break
                
                
            
            else: 
                optimal_cost_a, optimal_path_ab = upgraded_biucs(graph, frontier_a1, explored_a1, path_cost_a1, path_a1,  node_a, node_b, optimal_cost_a, optimal_path_ab, hcost_a1, heuristic)
                #print("For node = ", node1[-1], "optimal path so far =", optimal_path_ab, "optimal_cost =" , optimal_cost_a)
                
        # start exploring frontier b   
        if(frontier_b1.size() > 0 and not pathbc_from_preload):
            cost_b1, x, node1 = frontier_b1.top()
            
            total_cost = cost_b1 
            if(total_cost >= optimal_cost_b ):
                found_path_b = True
                
                if(found_path_a and not found_path_c):
                    if(( cost_c1  > get_path_cost(graph, optimal_path_ab)) and  (cost_c1 > get_path_cost(graph, optimal_path_bc))):
                        #print("success -1 ")
                        optimal_cost_c = float("inf")
                        optimal_path_ca = []
                        break
                
                if(found_path_c and not found_path_a):
                    if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                        optimal_cost_a = float("inf")
                        optimal_path_ab = []
                        break
               
            
            else:
                optimal_cost_b, optimal_path_bc = upgraded_biucs(graph, frontier_b1, explored_b1, path_cost_b1, path_b1,  node_b, node_c, optimal_cost_b, optimal_path_bc,  hcost_b1, heuristic)
                #print("For node = ", node1[-1], "optimal path so far =", optimal_path_bc, "optimal_cost =" , optimal_cost_b)
                
                
        #start exploring frontier c
        if(frontier_c1.size() > 0 and not pathca_from_preload):
            cost_c1, x, node1 = frontier_c1.top()
            
            
            total_cost = cost_c1 
            
            if(total_cost >= optimal_cost_c ):
                found_path_c = True
            
                if(found_path_b and not found_path_a):
                    if((cost_a1  > get_path_cost(graph, optimal_path_ca)) and  (cost_a1  > get_path_cost(graph, optimal_path_bc))):
                        #print("success -2 ")
                        optimal_cost_a = float("inf")
                        optimal_path_ab = []
                        break
                
                if(found_path_a and not found_path_b):
                    if((cost_b1  > get_path_cost(graph, optimal_path_ca)) and  (cost_b1  > get_path_cost(graph, optimal_path_ab))):
                        #print("success -2 ")
                        optimal_cost_b = float("inf")
                        optimal_path_bc = []
                        break
                
               
            else:
                optimal_cost_c, optimal_path_ca = upgraded_biucs(graph, frontier_c1, explored_c1,  path_cost_c1, path_c1,  node_c, node_a, optimal_cost_c, optimal_path_ca,  hcost_c1, heuristic)
                #print("For node = ", node1[-1], "optimal path so far =", optimal_path_ca, "optimal_cost =" , optimal_cost_c)
                
        if((found_path_a or frontier_a1.size() <= 0 or pathab_from_preload) and (found_path_b or frontier_b1.size() <= 0 or pathbc_from_preload) and (found_path_c or frontier_c1.size() <= 0 or pathca_from_preload)):
             break
   
    
    
    if(len(optimal_path_ab) > 0 and len(optimal_path_bc) > 0):
        path_cost_1 = get_path_cost(graph, optimal_path_ab) + get_path_cost(graph, optimal_path_bc)
    
    if(len(optimal_path_bc) > 0 and len(optimal_path_ca) > 0):
        path_cost_2 = get_path_cost(graph, optimal_path_bc) + get_path_cost(graph, optimal_path_ca)
    
    if(len(optimal_path_ca) > 0 and len(optimal_path_ab) > 0):
        path_cost_3 = get_path_cost(graph, optimal_path_ca) + get_path_cost(graph, optimal_path_ab)
    
    
    
    
    #path_cost_bc = get_path_cost(graph, opt_path_bc)
    #path_cost_ca = get_path_cost(graph, opt_path_ca)
    
    minimum_cost = min(path_cost_1, path_cost_2, path_cost_3)
    #print("minimum cost = ", minimum_cost)
    #print("paths = ", optimal_path_ab, ",", optimal_path_bc , "," , optimal_path_ca)
    #print("explored for goal: ", node_a , " = " , explored_a1 , "goal: ", node_b , " = " , explored_b1, " goal : ", node_c , " = " , explored_c1)
    #print("path costs === [", node_a , "] = "  , path_cost_a1, "[", node_b , "] = ", path_cost_b1, "[", node_c , "] = " , path_cost_c1)
    
    if(set(goals).issubset(optimal_path_ab)):
        #print(sum(graph._explored_nodes.values()))
        #print("path returned 1 = ", optimal_path_ab)  
    
        return optimal_path_ab
    elif (set(goals).issubset(optimal_path_bc)):
        #print(sum(graph._explored_nodes.values()))
        #print("path returned 2 = ", optimal_path_bc)  
    
        return optimal_path_bc
    elif (set(goals).issubset(optimal_path_ca)):
        #print(sum(graph._explored_nodes.values()))
        #print("path returned 3 = ", optimal_path_ca)  
        return optimal_path_ca
    
    if(path_cost_1 == minimum_cost):
        #print("merging path 1 = ", optimal_path_ab, " and " , optimal_path_bc)
        opt_path = optimal_path_ab[:-1] + optimal_path_bc
    
    elif(path_cost_2 == minimum_cost):
        #print("merging path 2 = ", optimal_path_bc, " and " , optimal_path_ca)
        opt_path = optimal_path_bc[:-1] + optimal_path_ca
    else:
        #print("merging path 3 = ", optimal_path_ca, " and " , optimal_path_ab)
        opt_path = optimal_path_ca[:-1] + optimal_path_ab
                               
    #print("path returned 4 = ", opt_path)  
    print(sum(graph._explored_nodes.values()))
    return opt_path

  
def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    name = "Nidhi Agrawal"
    return name
    #raise NotImplementedError


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
           
       """
    x1,y1 = graph.nodes[v]['pos']
    x2,y2 = graph.nodes[goal]['pos']
    
    distance = abs(x2 - x1) + abs(y2 - y1)
    return int(distance)
    
       #return ()

pass

# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to bonnie, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError



def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    #nodes = graph.nodes()
    
    return None
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
