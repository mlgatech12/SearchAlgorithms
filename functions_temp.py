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
        return path1
    print("start = ", start, " and goal = ", goal)
    
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
    
    min_cost = float("inf")
    
    
    while frontier1.size() > 0 and frontier2.size() > 0 :
        print("****")
        #print("check a = " ,frontier.__contains__(['a']))
        #print("check z = " ,frontier.__contains__(['z']))
        if(frontier1.size() > 0 ):
        
            cost1, node1 = frontier1.pop()
            if(node1[-1] not in explored1):
                explored1.append(node1[-1])
                
            print("forward explored = " , explored1)
            
            neighbors = sorted(graph.neighbors(node1[-1]))
            for neighbor in neighbors: 
                if(neighbor not in explored1):
                    new_cost = cost1 + graph.get_edge_weight(node1[-1], neighbor)
                    
                    
                    if(not frontier1.__nodeExists__(neighbor) ):
                        
                        frontier1.append((new_cost, [neighbor]))
                        path_cost1[neighbor] = new_cost
                        path1[neighbor] = node1[-1]
                        
                    elif new_cost < path_cost1[neighbor]:
                        print("changing frontier 1 q")
                        frontier1.remove([neighbor])
                        frontier1.append((new_cost, [neighbor]))
                        path1[neighbor] = node1[-1]
                    
                    if(neighbor in explored2):
                        explored2_cost = path_cost1[neighbor] + path_cost2[neighbor]
                        if(explored2_cost < min_cost):
                            #print("forward queue =", frontier1.__str__())
                            #print("backward queue =", frontier2.__str__())
        
                            #print("cost dictionary 1 =", path_cost1)
                            #print("cost dictionary 2 =", path_cost2)
        
        
                            #print("path dictionary 1", path1)
                            #print("path dictionary 2", path2)
                            
                            #print("mid node 1 = ", neighbor)
                            min_cost = explored2_cost
                            opt_path = optimal_path(path1, path2, neighbor, start, goal) 
                            foundPath1 = True
            
            #return  opt_path 
                        
                            
        if(frontier2.size() > 0 ):
        
            cost2, node2 = frontier2.pop()
            if(node2[-1] not in explored2):
                explored2.append(node2[-1])
            
            print("backward explored = " , explored2)
            
            neighbors = sorted(graph.neighbors(node2[-1]))
            for neighbor in neighbors: 
                if(neighbor not in explored2):
                    new_cost = cost2 + graph.get_edge_weight(node2[-1], neighbor)
                    
                    
                    if(not frontier2.__nodeExists__(neighbor) ):
                        
                        frontier2.append((new_cost, [neighbor]))
                        path_cost2[neighbor] = new_cost
                        path2[neighbor] = node2[-1]
                        
                        
                    elif new_cost < path_cost2[neighbor]:
                        print("changing frontier 2 q")
                        frontier2.remove([neighbor])
                        frontier2.append((new_cost, [neighbor]))
                        path2[neighbor] = node2[-1]
                    
                    if(neighbor in explored1) :
                        explored1_cost = path_cost1[neighbor] + path_cost2[neighbor]
                        if(explored1_cost < min_cost):
                            #print("forward queue =", frontier1.__str__())
                            #print("backward queue =", frontier2.__str__())
        
                            #print("cost dictionary 1 =", path_cost1)
                            #print("cost dictionary 2 =", path_cost2)
        
        
                            #print("path dictionary 1", path1)
                            #print("path dictionary 2", path2)
                            
                            #print("mid node 2 = ", neighbor)
                            min_cost = explored1_cost
                            opt_path = optimal_path(path1, path2, neighbor, start, goal)
                        
                        
            #return opt_path
                        
                   
            
    
    #End of while loop
        
    
        
    
    return opt_path

    #raise NotImplementedError
    
    
'''

def bidirectional_ucs(graph, start, goal):
    """Run bidirectional uniform-cost search
    between start and goal"""
    #print("*****")
    print("start = ", start, " and goal = ", goal)
    
    if start == goal:
        return []
    
    frt1 = PriorityQueue()
    frt2 = PriorityQueue()
    
    prev1 = {}
    prev2 = {}
    
    explored1 = set()
    explored2 = set()
    
    costs1 = {start: 0}
    costs2 = {goal: 0}
    
    frt1.append((0, start))
    frt2.append((0, goal))
    
    def get_path(prev, start, goal, reverse=False):
        print("[[ Inside getting path = ", "reverse = ", reverse, " start = " , start, "goal = ", goal)
        print(prev)
        path = []
        if start == goal:
            return path
        if not reverse:
            prev_node = goal
            while prev_node != start:
                path = [prev_node] + path
                prev_node = prev[prev_node]
            path = [start] + path
        else:
            next_node = start
            while next_node != goal:
                path.append(next_node)
                next_node = prev[next_node]
            path.append(goal)
        #print("from this fn path returned = ", path)
        return path
        
    def combine_path(node):
        if node == start:
            return get_path(prev2, node, goal, True)
        if node == goal:
            return get_path(prev1, start, node)
        
        #print("[[start = ", start, "center point = ", node, "goal = ", goal)
        #print("Costs dictionaries]]")
        #print(costs1)
        #print(costs2)
        #print("Path dictionaries")
        #print(prev1)
        #print(prev2)
        path1 = get_path(prev1, start, node)
        path2 = get_path(prev2, node, goal, True)
        return path1[:-1] + path2
        
    min_cost = float('Inf')
    path = []
    
    def add_neighbors(node, cost, frt, explored, other_explored, costs, other_costs, prev, min_cost, path):
        neighbors = sorted(graph.neighbors(node))
        for ngbr in neighbors:
            #print("explored = ", explored)
            #print("queue = ", frt.__str__())
            #print("other explored = ",  other_explored)
            #print("checking neighbor = " ,ngbr)
            if ngbr not in explored:
                if ngbr not in frt:
                    #print("condition 1")
                    costs[ngbr] = cost + graph.get_edge_weight(node, ngbr)
                    frt.append((costs[ngbr], ngbr))
                    prev[ngbr] = node
                    #print("previous neigbor =" , prev[ngbr])
                if ngbr in frt and cost + graph.get_edge_weight(node, ngbr) < costs[ngbr]:
                    #print("condition 2 ", "cost 1 = ", cost + graph.get_edge_weight(node, ngbr), " cost 2 = ", costs[ngbr])
                    frt.remove(ngbr)
                    costs[ngbr] = cost + graph.get_edge_weight(node, ngbr)
                    frt.append((costs[ngbr], ngbr))
                    prev[ngbr] = node
                    #print("previous neigbor =" , prev[ngbr])
                
                if ngbr in other_explored and costs[ngbr] + other_costs[ngbr] < min_cost:
                    #print("condition 3", " cost 1 = ", costs[ngbr] + other_costs[ngbr] , "min cost = ", min_cost )
                    min_cost = costs[ngbr] + other_costs[ngbr]
                    path = combine_path(ngbr)
                    #print("path = " , path)
        
        return min_cost, path
        
    while frt1.size() > 0 and frt2.size() > 0:
        cost1, node1 = frt1.pop()
        cost2, node2 = frt2.pop()
        
        if cost1 + cost2 >= min_cost:
            break
        
        explored1.add(node1)
        explored2.add(node2)
        
        min_cost, path = add_neighbors(node1, cost1, frt1, explored1, explored2, costs1, costs2, prev1, min_cost, path)
        #print("==== second call ====")
        min_cost, path = add_neighbors(node2, cost2, frt2, explored2, explored1, costs2, costs1, prev2, min_cost, path)

    
    #print("path returned = ", path)
    return path   


'''

def bidirectional_ucs(graph, start, goal):
    """Run bidirectional uniform-cost search
    between start and goal"""
    #print("*****")
    print("start = ", start, " and goal = ", goal)
    
    if start == goal:
        return []
    
    frt1 = PriorityQueue()
    frt2 = PriorityQueue()
    
    prev1 = {}
    prev2 = {}
    
    explored1 = set()
    explored2 = set()
    
    costs1 = {start: 0}
    costs2 = {goal: 0}
    
    frt1.append((0, start))
    frt2.append((0, goal))
    
    def get_path(prev, start, goal, reverse=False):
        print("[[ Inside getting path = ", "reverse = ", reverse, " start = " , start, "goal = ", goal)
        print(prev)
        path = []
        if start == goal:
            return path
        if not reverse:
            prev_node = goal
            while prev_node != start:
                path = [prev_node] + path
                prev_node = prev[prev_node]
            path = [start] + path
        else:
            next_node = start
            while next_node != goal:
                path.append(next_node)
                next_node = prev[next_node]
            path.append(goal)
        print("from this fn path returned = ", path)
        return path
        
    def combine_path(node):
        if node == start:
            return get_path(prev2, node, goal, True)
        if node == goal:
            return get_path(prev1, start, node)
        
        print("[[start = ", start, "center point = ", node, "goal = ", goal)
        print("Costs dictionaries]]")
        print(costs1)
        print(costs2)
        print("Path dictionaries")
        print(prev1)
        print(prev2)
        path1 = get_path(prev1, start, node)
        path2 = get_path(prev2, node, goal, True)
        return path1[:-1] + path2
        
    min_cost = float('Inf')
    path = []
    
    def add_neighbors(node, cost, frt, explored, other_explored, costs, other_costs, prev, min_cost, path):
        neighbors = sorted(graph.neighbors(node))
        for ngbr in neighbors:
            print("explored = ", explored)
            print("queue = ", frt.__str__())
            print("other explored = ",  other_explored)
            print("checking neighbor = " ,ngbr)
            if ngbr not in explored:
                if ngbr not in frt:
                    print("condition 1")
                    costs[ngbr] = cost + graph.get_edge_weight(node, ngbr)
                    frt.append((costs[ngbr], ngbr))
                    prev[ngbr] = node
                    print("previous neigbor =" , prev[ngbr])
                if ngbr in frt and cost + graph.get_edge_weight(node, ngbr) < costs[ngbr]:
                    print("condition 2 ", "cost 1 = ", cost + graph.get_edge_weight(node, ngbr), " cost 2 = ", costs[ngbr])
                    frt.remove(ngbr)
                    costs[ngbr] = cost + graph.get_edge_weight(node, ngbr)
                    frt.append((costs[ngbr], ngbr))
                    prev[ngbr] = node
                    print("previous neigbor =" , prev[ngbr])
                
                if ngbr in other_explored and costs[ngbr] + other_costs[ngbr] < min_cost:
                    print("condition 3", " cost 1 = ", costs[ngbr] + other_costs[ngbr] , "min cost = ", min_cost )
                    min_cost = costs[ngbr] + other_costs[ngbr]
                    path = combine_path(ngbr)
                    print("path = " , path)
        
        return min_cost, path
        
    while frt1.size() > 0 and frt2.size() > 0:
        cost1, node1 = frt1.pop()
        cost2, node2 = frt2.pop()
        
        if cost1 + cost2 >= min_cost:
            break
        
        explored1.add(node1)
        explored2.add(node2)
        
        min_cost, path = add_neighbors(node1, cost1, frt1, explored1, explored2, costs1, costs2, prev1, min_cost, path)
        print("==== second call ====")
        min_cost, path = add_neighbors(node2, cost2, frt2, explored2, explored1, costs2, costs1, prev2, min_cost, path)

    
    print("path returned = ", path)
    return path   

   
