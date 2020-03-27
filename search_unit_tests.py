# coding=utf-8
import itertools
import pickle
import random
import unittest

import networkx

from explorable_graph import ExplorableGraph
from submission import a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, euclidean_dist_heuristic, \
    null_heuristic, haversine_dist_heuristic, tridirectional_search, tridirectional_upgraded, \
    uniform_cost_search, custom_heuristic

#from submission import tridirectional_search


def is_valid(graph, path, start, goal):
    """
    Test whether a path is valid or not
    """
    if start == goal:
        return path == []
    else:
        if path[0] != start or path[-1] != goal:
            return False
    for i in range(len(path) -1):
        if path[i + 1] not in graph.neighbors(path[i]):
            return False
    return True

class SearchUnitTests(unittest.TestCase):
    """
    Error Diagnostic code courtesy one of our former students -  Mac Chan

    The following unit tests will check for all pairs on romania and random
    points on atlanta.
    Comment out any tests that you haven't implemented yet.

    If you failed on bonnie because of non-optimal path, make sure you pass
    all the local tests.
    Change test_count=-1 if you failed the path test on bonnie, it will run
    tests on atlanta until it finds a set of points that fail.

    If you failed on bonnie because of your explored set is too large,
    there is no easy way to test without a reference implementation.
    But you can read the pdf slides for the optimized terminal condition.

    To run,
    nosetests --nocapture -v search_unit_tests.py:SearchUnitTests
    nosetests --nocapture -v
                        search_unit_tests.py:SearchUnitTests.test_bfs_romania
    """

    def setUp(self):
        """Setup both atlanta and romania graph data."""

        with (open("romania_graph.pickle", "rb")) as romFile:
            romania = pickle.load(romFile)
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()

        with (open("atlanta_osm.pickle", "rb")) as atlFile:
            atlanta = pickle.load(atlFile)
        self.atlanta = ExplorableGraph(atlanta)
        self.atlanta.reset_search()

        self.margin_of_error = 1.0e-6

    def reference_path(self, graph, src_node, dst_node, weight='weight'):
        """
        Path as generated by networkx shortest path.

        Args:
            graph (ExplorableGraph): Undirected graph to search.
            src_node (node): Key for the start node.
            dst_node (node): Key for the end node.
            weight (:obj:`str`):
                If None, every edge has weight/distance/cost 1.
                If a string, use this edge attribute as the edge weight.
                Any edge attribute not present defaults to 1.

        Returns:
            Tuple with (cost of path, path as list).
        """

        graph.reset_search()
        path = networkx.shortest_path(graph, src_node, dst_node, weight=weight)
        cost = self.sum_weight(graph, path)

        return cost, path

    def reference_bfs_path(self, graph, src_node, dst_node):
        """
        Breadth First Search as generated by networkx shortest path.

        Args:
            graph (ExplorableGraph): Undirected graph to search.
            src_node (node): Key for the start node.
            dst_node (node): Key for the end node.

        Returns:

        """
        return self.reference_path(graph, src_node, dst_node, weight=None)

    @staticmethod
    def sum_weight(graph, path):
        """
        Calculate the total cost of a path by summing edge weights.

        Args:
            graph (ExplorableGraph): Graph that contains path.
            path (list(nodes)): List of nodes from src to dst.

        Returns:
            Sum of edge weights in path.
        """
        pairs = zip(path, path[1:])

        return sum([graph.get_edge_data(a, b)['weight'] for a, b in pairs])

    def run_romania_data(self, ref_method, method, **kwargs):
        """
        Run the test search against the Romania data.

        Args:
            ref_method (func): Reference search function to compare test search
            method (func): Test search function.
            kwargs: Keyword arguments.

        Asserts:
            True if the path from the test search is equivalent to the
            reference search.
        """

        keys = self.romania.nodes.keys()
        pairs = itertools.permutations(keys, 2)
        for src, dst in pairs:
            self.romania.reset_search()
            path = method(self.romania, src, dst, **kwargs)
            ref_len, ref_path = ref_method(self.romania, src, dst)

            if path != ref_path:
                print (src, dst)

            self.assertEqual(ref_path, path)

    def run_romania_tri(self, method, **kwargs):
        """
        Run the tridirectional test search against the Romania data.

        Args:
            method (func): Test search function.
            kwargs: Keyword arguments.

        Asserts:
            True if the path from the test search is equivalent to the
            reference search.
        """

        keys = self.romania.nodes.keys()
        triplets = itertools.permutations(keys, 3)
        for goals in triplets:
            self.romania.reset_search()
            path = method(self.romania, goals, **kwargs)
            path_len = self.sum_weight(self.romania, path)
            s1len, _ = self.reference_path(self.romania, goals[0], goals[1])
            s2len, _ = self.reference_path(self.romania, goals[2], goals[1])
            s3len, _ = self.reference_path(self.romania, goals[0], goals[2])
            min_len = min(s1len + s2len, s1len + s3len, s3len + s2len)

            if path_len != min_len:
                print (goals)

            self.assertEqual(min_len, path_len)

    def run_atlanta_data(self, method, test_count=10, **kwargs):
        """
        Run the bidirectional test search against the Atlanta data.

        In the interest of time and memory, this is not an exhaustive search of
        all possible pairs in the graph.

        Args:
            method (func): Test search function.
            test_count (int): Number of tests to run. Default is 10.
            kwargs: Keyword arguments.

        Asserts:
            True if the path from the test search is equivalent to the
            reference search.
        """

        keys = list(networkx.connected_components(self.atlanta).__next__())
        random.shuffle(keys)
        for src, dst in list(zip(keys, keys[1:]))[::2]:
            self.atlanta.reset_search()
            path = method(self.atlanta, src, dst, **kwargs)
            path_len = self.sum_weight(self.atlanta, path)
            ref_len, ref_path = self.reference_path(self.atlanta, src, dst)
            if abs(path_len - ref_len) > self.margin_of_error:
                print (src, dst)

            self.assertAlmostEqual(path_len, ref_len,
                                   delta=self.margin_of_error)
            test_count -= 1

            if test_count == 0:
                break

    def run_atlanta_tri(self, method, test_count=10, **kwargs):
        """
        Run the tridirectional test search against the Atlanta data.

        In the interest of time and memory, this is not an exhaustive search of
        all possible triplets in the graph.

        Args:
            method (func): Test search function.
            test_count (int): Number of tests to run. Default is 10.
            kwargs: Keyword arguments.

        Asserts:
            True if the path from the test search is equivalent to the
            reference search.
        """

        keys = list(next(networkx.connected_components(self.atlanta)))
        random.shuffle(keys)
        for goals in list(zip(keys, keys[1:], keys[2:]))[::3]:
            self.atlanta.reset_search()
            path = method(self.atlanta, goals, **kwargs)
            path_len = self.sum_weight(self.atlanta, path)
            s1len, _ = self.reference_path(self.atlanta, goals[0], goals[1])
            s2len, _ = self.reference_path(self.atlanta, goals[2], goals[1])
            s3len, _ = self.reference_path(self.atlanta, goals[0], goals[2])
            min_len = min(s1len + s2len, s1len + s3len, s3len + s2len)

            if abs(path_len - min_len) > self.margin_of_error:
                print (goals)
            self.assertAlmostEqual(path_len, min_len,
                                   delta=self.margin_of_error)
            test_count -= 1
            if test_count == 0:
                break

    def same_node_bi(self, graph, method, test_count=10, **kwargs):
        """
        Run the a bidirectional test search using same start and end node.

        Args:
            graph (ExplorableGraph): Graph that contains path.
            method (func): Test search function.
            test_count (int): Number of tests to run. Default is 10.
            kwargs: Keyword arguments.

        Asserts:
            True if the path between the same start and end node is empty.
        """

        keys = list(networkx.connected_components(graph).__next__())
        random.shuffle(keys)

        for i in range(test_count):
            path = method(graph, keys[i], keys[i], **kwargs)

            self.assertFalse(path)

    def test_same_node_bi(self):
        """
        Test bidirectional search using the same start and end nodes.

        Searches Tested:
            breadth_first_search
            uniform_cost_search
            a_star, null_heuristic
            a_star, euclidean_dist_heuristic
            bidirectional_ucs
            bidirectional_a_star, null_heuristic
            bidirectional_a_star, euclidean_dist_heuristic
        """

        self.same_node_bi(self.romania, breadth_first_search)
        self.same_node_bi(self.romania, uniform_cost_search)
        self.same_node_bi(self.romania, a_star, heuristic=null_heuristic)
        self.same_node_bi(self.romania, a_star,
                          heuristic=euclidean_dist_heuristic)
        self.same_node_bi(self.romania, bidirectional_ucs)
        self.same_node_bi(self.romania, bidirectional_a_star,
                          heuristic=null_heuristic)
        self.same_node_bi(self.romania, bidirectional_a_star,
                          heuristic=euclidean_dist_heuristic)

    def same_node_tri_test(self, graph, method, test_count=10, **kwargs):
        """
        Run the tridirectional test search using same start and end nodes

        Args:
            graph (ExplorableGraph): Graph that contains path.
            method (func): Test search function.
            test_count (int): Number of tests to run. Default is 10.
            kwargs: Keyword arguments.

        Asserts:
            True if the path between the same start and end node is empty.
        """

        keys = list(next(networkx.connected_components(graph)))
        random.shuffle(keys)
        for i in range(test_count):
            path = method(graph, [keys[i], keys[i], keys[i]], **kwargs)
            self.assertFalse(path)

    '''
    def test_same_node_tri(self):
        """
        Test bidirectional search using the same start and end nodes.

        Searches Tested:
            tridirectional_search
            tridirectional_upgraded, null_heuristic
            tridirectional_upgraded, euclidean_dist_heuristic
        """

        self.same_node_tri_test(self.romania, tridirectional_search)
        self.same_node_tri_test(self.romania, tridirectional_upgraded,
                                heuristic=null_heuristic)
        self.same_node_tri_test(self.romania, tridirectional_upgraded,
                                heuristic=euclidean_dist_heuristic)

    
    def test_bfs_romania(self):
        """Test breadth first search with Romania data."""


        keys = self.romania.nodes.keys()
        pairs = itertools.permutations(keys, 2)
        for src in keys:
            for dst in keys:
                self.romania.reset_search()
                path = breadth_first_search(self.romania, src, dst)
                ref_len, ref_path = self.reference_bfs_path(self.romania, src, dst)
                self.assertTrue(is_valid(self.romania, path, src, dst),
                     msg="path %s for start '%s' and goal '%s' is not valid" % (path, src, dst))
                if src != dst: # we want path == [] if src == dst
                    self.assertTrue(len(path) == len(ref_path), msg="Path is too long. Real path: %s, your path: %s" % (ref_path, path))

    
    def test_ucs_romania(self):
        """Test uniform cost search with Romania data."""

        self.run_romania_data(self.reference_path, uniform_cost_search)

    '''
    def test_a_star_null_romania(self):
        #Test A* search with Romania data and the Null heuristic."""

        self.run_romania_data(self.reference_path, a_star,
                              heuristic=null_heuristic)
    '''
    def test_a_star_euclidean_romania(self):
        """Test A* search with Romania data and the Euclidean heuristic."""

        self.run_romania_data(self.reference_path, a_star,
                              heuristic=euclidean_dist_heuristic)
    
    
    def test_bi_ucs_romania(self):
        """Test Bi-uniform cost search with Romania data."""

        self.run_romania_data(self.reference_path, bidirectional_ucs)
    
    
    
    
    def test_bi_ucs_atlanta(self):
        """
        Test Bi-uniform cost search with Atlanta data.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_data(bidirectional_ucs, test_count=10)

    
    def test_bi_a_star_null_romania(self):
        """Test Bi-A* search with Romania data and the Null heuristic."""

        self.run_romania_data(self.reference_path, bidirectional_a_star,
                              heuristic=null_heuristic)

    
    def test_bi_a_star_null_atlanta(self):
        """
        Test Bi-A* search with Atlanta data and the Null heuristic.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_data(bidirectional_a_star, heuristic=null_heuristic,
                              test_count=10)
    
    def test_bi_a_star_euclidean_romania(self):
        """Test Bi-A* search with Romania data and the Euclidean heuristic."""

        self.run_romania_data(self.reference_path, bidirectional_a_star,
                              heuristic=euclidean_dist_heuristic)
    
    def test_bi_a_star_euclidean_atlanta(self):
        """
        Test Bi-A* search with Atlanta data and the Euclidean heuristic.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_data(bidirectional_a_star,
                              heuristic=euclidean_dist_heuristic,
                              test_count=10)

    def test_bi_a_star_haversine_atlanta(self):
        """
        Test Bi-A* search with Atlanta data and the Haversine heuristic.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_data(bidirectional_a_star,
                              heuristic=haversine_dist_heuristic,
                              test_count=10)

    
    def test_tri_ucs_romania(self):
        """Test Tri-UC search with Romania data."""

        self.run_romania_tri(tridirectional_search)
    
    def test_tri_ucs_atlanta(self):
        """
        Test Tri-UC search with Atlanta data.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_tri(tridirectional_search, test_count=10)
    
    '''
    def test_tri_upgraded_null_romania(self):
        """
        Test upgraded tri search with Romania data and the Null heuristic.
        """

        self.run_romania_tri(tridirectional_upgraded, heuristic=null_heuristic)

    '''
    def test_tri_upgraded_null_atlanta(self):
        """
        Test upgraded tri search with Atlanta data and the Null heuristic.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_tri(tridirectional_upgraded, test_count=10,
                             heuristic=null_heuristic)
    
    def test_tri_upgraded_euclidean_romania(self):
        """
        Test upgraded tri search with Romania data and the Euclidean heuristic.
        """

        self.run_romania_tri(tridirectional_upgraded,
                             heuristic=euclidean_dist_heuristic)
    
    def test_tri_upgraded_euclidean_atlanta(self):
        """
        Test upgraded tri search with Atlanta data and the Euclidean heuristic.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_tri(tridirectional_upgraded, test_count=10,
                             heuristic=euclidean_dist_heuristic)

    def test_tri_upgraded_haversine_atlanta(self):
        """
        Test upgraded tri search with Atlanta data and the Haversine heuristic.

        To loop test forever, set test_count to -1
        """

        self.run_atlanta_tri(tridirectional_upgraded, test_count=10,
                             heuristic=haversine_dist_heuristic)

    '''
if __name__ == '__main__':
    unittest.main()
